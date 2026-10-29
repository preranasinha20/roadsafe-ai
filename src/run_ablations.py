"""Step 6: cross-validation, statistical comparison and ablation studies.

Run:  python -m src.run_ablations
 * 5-fold stratified CV (on a 150k training sample) for every model family, with a paired t-test
   of the best model against each competitor.
 * Ablations of the final model: which feature groups matter, and what leakage would do.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import *
from .metrics import evaluate, tune_class_scales
from .pipeline import feature_set, load_splits, make_classifier, make_preprocessor

warnings.filterwarnings("ignore")


def cross_validate_models():
    tr, _, _ = load_splits(sample=150_000)
    cats, nums = feature_set("full")
    X, y = tr[cats + nums], tr.severity.values
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models = {"Logistic Regression": ("logreg", "balanced"), "Decision Tree": ("tree", "balanced"),
              "Random Forest": ("rf", "balanced"), "XGBoost": ("xgb", None), "LightGBM": ("lgbm", "balanced")}
    scores = {}
    for name, (m, cw) in models.items():
        fold = []
        for tr_i, te_i in skf.split(X, y):
            pipe = Pipeline([("prep", make_preprocessor(cats, nums, scale=(m == "logreg"))),
                             ("clf", make_classifier(m, class_weight=cw))])
            fit_kw = {}
            if m == "xgb":
                counts = np.bincount(y[tr_i])
                fit_kw["clf__sample_weight"] = (len(tr_i) / (3 * counts))[y[tr_i]]
            pipe.fit(X.iloc[tr_i], y[tr_i], **fit_kw)
            p = pipe.predict_proba(X.iloc[te_i])
            fold.append(f1_score(y[te_i], p.argmax(1), average="macro"))
        scores[name] = fold
        print(f"CV {name:20s} macro-F1 {np.mean(fold):.4f} +/- {np.std(fold):.4f}", flush=True)
    best = max(scores, key=lambda k: np.mean(scores[k]))
    rows = []
    for k, v in scores.items():
        t, p = (np.nan, np.nan) if k == best else ttest_rel(scores[best], v)
        rows.append({"model": k, "cv_macro_f1_mean": np.mean(v), "cv_macro_f1_std": np.std(v),
                     **{f"fold{i + 1}": s for i, s in enumerate(v)},
                     "paired_t_vs_best": t, "p_value_vs_best": p, "best": k == best})
    pd.DataFrame(rows).to_csv(TAB_DIR / "cv_results.csv", index=False)
    return rows


def ablations():
    res = json.load(open(TAB_DIR / "final_results.json"))
    model = "lgbm" if res["lgbm"]["val_tuned"]["macro_f1"] >= res["xgb"]["val_tuned"]["macro_f1"] else "xgb"
    params, wp = res[model]["params"], res[model]["weight_power"]
    tr, va, te = load_splits(sample=400_000)
    rows = []
    variants = [("Full model (scene + time + location + spatial history)", "full"),
                ("- spatial history features", "no_history"),
                ("- time features", "no_time"),
                ("- raw coordinates (lat/lon)", "no_geo"),
                ("Scene only (no history, no coordinates)", "scene_only"),
                ("+ road-level history", "with_road_history"),
                ("+ LEAKAGE: casualties & police attendance", "with_leaky")]
    for label, fs in variants:
        cats, nums = feature_set(fs)
        cols = cats + nums
        prep = make_preprocessor(cats, nums)
        A, V, T = prep.fit_transform(tr[cols]), prep.transform(va[cols]), prep.transform(te[cols])
        y = tr.severity.values
        counts = np.bincount(y)
        w = ((len(y) / (3 * counts)) ** wp)[y]
        clf = make_classifier(model, **params)
        clf.fit(A, y, sample_weight=w)
        pv, pt = clf.predict_proba(V), clf.predict_proba(T)
        sc = tune_class_scales(pv, va.severity.values)
        m = evaluate(te.severity.values, pt, sc)
        rows.append({"variant": label, "feature_set": fs, **{k: m[k] for k in
                     ["macro_f1", "accuracy", "balanced_accuracy", "recall_Fatal", "recall_Serious",
                      "roc_auc_ovr", "pr_auc_Fatal", "mcc"]}})
        print(f"{label:60s} macroF1 {m['macro_f1']:.4f}  AUC {m['roc_auc_ovr']:.4f}  "
              f"acc {m['accuracy']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(TAB_DIR / "ablations.csv", index=False)


if __name__ == "__main__":
    t = time.time()
    cross_validate_models()
    ablations()
    print(f"done {(time.time() - t) / 60:.1f} min")
