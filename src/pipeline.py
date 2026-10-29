"""Feature groups, preprocessing and model factories shared by every experiment."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import *

CATEGORICAL = [
    "1st_Road_Class", "2nd_Road_Class", "Junction_Detail", "Junction_Control",
    "Light_Conditions", "Road_Surface_Conditions", "Road_Type", "Weather_Conditions",
    "Urban_or_Rural_Area", "Carriageway_Hazards", "Special_Conditions_at_Site",
    "Pedestrian_Crossing-Human_Control", "Pedestrian_Crossing-Physical_Facilities",
    "la_district", "Police_Force", "season", "speed_band",
]
TIME_NUM = ["hour", "month", "dow", "is_weekend", "is_rush_hour", "is_night", "hour_sin", "hour_cos"]
SCENE_NUM = ["speed_limit", "n_vehicles", "is_urban", "wet_surface", "bad_weather",
             "dark_and_unlit", "junction_at_night", "high_speed_rural"]
GEO_NUM = ["lat", "lon"]
HISTORY_NUM = ["h_cell_n", "h_cell_ksi_rate", "h_cell_score", "h_la_n", "h_la_ksi_rate",
               "h_la_score", "h_dist_hotspot_m", "h_in_hotspot"]
ROAD_HISTORY = ["h_road_n", "h_road_ksi_rate", "h_road_score"]   # tested in the ablation only
LEAKY = ["Number_of_Casualties", "Did_Police_Officer_Attend_Scene_of_Accident"]

TIME_CATS = ["season"]


def feature_set(name, extra_numeric=None, extra_categorical=None):
    """Named feature sets used in the ablation study. Returns (categorical, numeric)."""
    cats, nums = list(CATEGORICAL), TIME_NUM + SCENE_NUM + GEO_NUM
    if name == "full":
        nums = nums + HISTORY_NUM
    elif name == "scene_only":                      # no history, no coordinates
        nums = [c for c in nums if c not in GEO_NUM]
    elif name == "no_history":
        pass
    elif name == "no_time":
        cats = [c for c in cats if c not in TIME_CATS]
        nums = [c for c in nums if c not in TIME_NUM] + HISTORY_NUM
    elif name == "no_geo":
        nums = [c for c in nums if c not in GEO_NUM] + HISTORY_NUM
    elif name == "with_road_history":
        nums = nums + HISTORY_NUM + ROAD_HISTORY
    elif name == "with_leaky":
        nums = nums + HISTORY_NUM + LEAKY
    else:
        raise ValueError(name)
    if extra_numeric:
        nums = nums + list(extra_numeric)
    if extra_categorical:
        cats = cats + list(extra_categorical)
    return cats, nums


def make_preprocessor(cats, nums, scale=False):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=300,
                                                sparse_output=True, dtype=np.float32))]), cats),
        ("num", Pipeline(num_steps), nums),
    ], sparse_threshold=1.0)


def make_classifier(name, n_classes=3, class_weight=None, **kw):
    import lightgbm as lgb
    import xgboost as xgb
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return LogisticRegression(max_iter=300, C=0.5, class_weight=class_weight,
                                  solver="lbfgs", n_jobs=2)
    if name == "tree":
        return DecisionTreeClassifier(max_depth=12, min_samples_leaf=50,
                                      class_weight=class_weight, random_state=SEED)
    if name == "rf":
        return RandomForestClassifier(n_estimators=150, max_depth=16, min_samples_leaf=20,
                                      max_features="sqrt", class_weight=class_weight,
                                      n_jobs=2, random_state=SEED)
    if name == "lgbm":
        p = dict(n_estimators=400, learning_rate=0.06, num_leaves=63, min_child_samples=80,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                 class_weight=class_weight, n_jobs=2, random_state=SEED, verbose=-1)
        p.update(kw)
        return lgb.LGBMClassifier(**p)
    if name == "xgb":
        p = dict(n_estimators=300, learning_rate=0.08, max_depth=7, min_child_weight=5,
                 subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, tree_method="hist",
                 n_jobs=2, random_state=SEED, verbosity=0)
        p.update(kw)
        return xgb.XGBClassifier(**p)
    raise ValueError(name)


def sample_weights(y, mode="balanced"):
    """Inverse-frequency weights (balanced) or their square root (softer)."""
    y = np.asarray(y)
    counts = np.bincount(y)
    w = len(y) / (len(counts) * counts)
    if mode == "sqrt":
        w = np.sqrt(w)
    return w[y]


def load_splits(sample=None, random_state=SEED):
    """Time-based split. 2005 only seeds the history features, so training starts in 2006."""
    df = pd.read_parquet(PROC_DIR / "accidents_clean.parquet")
    tr = df[df.year.between(2006, max(TRAIN_YEARS))]
    va = df[df.year.isin(VAL_YEARS)]
    te = df[df.year.isin(TEST_YEARS)]
    if sample:
        tr = tr.groupby("severity").sample(frac=min(1, sample / len(tr)), random_state=random_state)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)
