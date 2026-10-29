"""Step 8: export everything the website needs (static JSON, model trees, figures).

Run:  python -m src.export_web
The deployed site is fully static: the gradient-boosted trees, calibration and decision scales are
exported to JSON and evaluated in the browser (app/src/lib/model.js).
"""
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score, roc_curve)

from .modelio import load_model
from .config import *
from .metrics import confusion
from .pipeline import CATEGORICAL, load_splits

DATA_OUT = APP_DATA
DATA_OUT.mkdir(parents=True, exist_ok=True)
NICE_MODEL = {"lgbm": "LightGBM", "xgb": "XGBoost"}


def dump(name, obj):
    with open(DATA_OUT / f"{name}.json", "w") as f:
        json.dump(obj, f, separators=(",", ":"), default=float)


def thin(x, y, n=90):
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return [[round(float(x[i]), 4), round(float(y[i]), 4)] for i in idx]


# ───────────────────────── model trees ─────────────────────────
def lgbm_trees(clf, n_classes):
    out = []
    for info in clf.booster_.dump_model()["tree_info"]:
        f, t, l, r, v, c, mi = [], [], [], [], [], [], []

        def walk(node):
            i = len(f)
            f.append(-1); t.append(0.0); l.append(-1); r.append(-1); v.append(0.0); c.append(1.0)
            if "leaf_value" in node:
                v[i] = node["leaf_value"]; c[i] = max(float(node.get("leaf_count", 1)), 1.0)
                return i
            assert node["decision_type"] == "<="
            f[i] = node["split_feature"]; t[i] = node["threshold"]
            c[i] = max(float(node.get("internal_count", 1)), 1.0)
            l[i] = walk(node["left_child"]); r[i] = walk(node["right_child"])
            return i

        walk(info["tree_structure"])
        k = info["tree_index"] % n_classes
        leaf = [i for i in range(len(f)) if f[i] < 0]
        base = sum(v[i] * c[i] for i in leaf) / c[0]
        out.append({"k": k, "f": f, "t": [float(np.float64(x)) for x in t], "l": l, "r": r,
                    "v": [round(x, 7) for x in v], "c": c, "base": base})
    return out


def xgb_trees(clf, n_classes):
    booster = clf.get_booster()
    out = []
    for idx, js in enumerate(booster.get_dump(dump_format="json", with_stats=True)):
        root = json.loads(js)
        f, t, l, r, v, c, mi = [], [], [], [], [], [], []

        def walk(node):
            i = len(f)
            f.append(-1); t.append(0.0); l.append(-1); r.append(-1); v.append(0.0); c.append(1.0); mi.append(-1)
            c[i] = max(float(node.get("cover", 1)), 1e-6)
            if "leaf" in node:
                v[i] = node["leaf"]
                return i
            kids = {ch["nodeid"]: ch for ch in node["children"]}
            f[i] = int(node["split"][1:])
            # xgboost sends x < thr to "yes"; the browser engine uses x <= thr
            t[i] = float(np.nextafter(np.float32(node["split_condition"]), np.float32(-np.inf)))
            l[i] = walk(kids[node["yes"]]); r[i] = walk(kids[node["no"]])
            mi[i] = l[i] if node["missing"] == node["yes"] else r[i]
            return i

        walk(root)
        leaf = [i for i in range(len(f)) if f[i] < 0]
        base = sum(v[i] * c[i] for i in leaf) / c[0]
        out.append({"k": idx % n_classes, "f": f, "t": t, "l": l, "r": r, "m": mi, "v": [round(x, 7) for x in v], "c": c, "base": base})
    return out


def group_of(name, cats):
    body = name[5:]
    return next(c for c in sorted(cats, key=len, reverse=True) if body.startswith(c + "_")), body


def export_model(bundle, df_hist):
    pipe, cal = bundle["pipeline"], bundle["calibrator"]
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    cats = bundle["cats"]
    names = list(prep.get_feature_names_out())
    columns = []
    for n in names:
        if n.startswith("num__"):
            columns.append({"type": "num", "col": n[5:]})
        else:
            col, body = group_of(n, cats)
            cat = body[len(col) + 1:]
            columns.append({"type": "cat", "col": col, "cat": "__infrequent__" if cat == "infrequent_sklearn" else cat})
    trees = lgbm_trees(clf, 3) if bundle["model"] == "lgbm" else xgb_trees(clf, 3)
    calib = [{"x": [round(float(a), 6) for a in m.X_thresholds_], "y": [round(float(a), 6) for a in m.y_thresholds_]}
             for m in cal.iso]
    bias = [0.0, 0.0, 0.0]
    if bundle["model"] == "xgb":     # xgboost keeps a per-class starting margin outside the trees
        import re
        cfg = json.loads(clf.get_booster().save_config())["learner"]["learner_model_param"]["base_score"]
        bias = [float(x) for x in re.findall(r"-?[\d.]+(?:E-?\d+)?", cfg)]
    model = {"classes": SEVERITY_ORDER, "bias": bias, "trees": trees, "calibration": calib,
             "scales": [float(s) for s in bundle["scales"]], "algorithm": bundle["model"]}
    dump("model", model)
    return model, columns, names


def export_meta(columns, df):
    """Option lists, local-authority table, location history grid and hotspot centroids."""
    cats_ui = ["Road_Type", "Junction_Detail", "Junction_Control", "Light_Conditions", "Weather_Conditions",
               "Road_Surface_Conditions", "1st_Road_Class"]
    options = {}
    for c in cats_ui:
        vc = df[c].dropna().value_counts()
        options[c] = [k for k, v in vc.items() if v >= 300 and k not in ("Unknown",)]
    prior = float(df.ksi.mean())
    la = df.groupby("la_district").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("ksi", "size"),
                                       k=("ksi", "sum"), score=("sev_weight", "sum"),
                                       force=("Police_Force", lambda s: s.mode().iat[0])).reset_index()
    la["ksi_rate"] = (la.k + 50 * prior) / (la.n + 50)
    la_list = [{"name": r.la_district, "lat": round(r.lat, 4), "lon": round(r.lon, 4), "n": int(r.n),
                "ksi_rate": round(float(r.ksi_rate), 5), "score": int(r.score), "force": r.force} for r in la.itertuples()]
    cell = (np.floor(df.lat / GRID_DEG).astype(int).astype(str) + "_" + np.floor(df.lon / GRID_DEG).astype(int).astype(str))
    g = df.assign(cell=cell).groupby("cell").agg(n=("ksi", "size"), k=("ksi", "sum"), s=("sev_weight", "sum"),
                                               la=("lat", "mean"), lo=("lon", "mean"))
    keep = g[g.n >= 3]
    cells = {k: [int(r.n), int(r.k), int(r.s)] for k, r in keep.iterrows()}
    ok = g[g.n >= 5]
    hot = ok[ok.s >= ok.s.quantile(0.98)]
    hot_list = [[round(a, 4), round(b, 4)] for a, b in zip(hot.la, hot.lo)]
    meta = {"encoder": {"columns": columns}, "options": options, "la": la_list, "cell_deg": GRID_DEG,
            "cells": cells, "hot": hot_list, "prior_ksi": prior,
            "test_ksi_rate": float(df[df.year.isin(TEST_YEARS)].ksi.mean())}
    dump("meta", meta)
    print("meta:", len(la_list), "LAs,", len(cells), "cells,", len(hot_list), "hotspot cells")
    return meta


def export_parity(bundle, n=80):
    """Reference rows so the JavaScript engine can be checked against scikit-learn/SHAP."""
    import shap
    _, _, te = load_splits()
    rows = te.sample(n, random_state=SEED).reset_index(drop=True)
    pipe, cal = bundle["pipeline"], bundle["calibrator"]
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    X = prep.transform(rows[bundle["features"]])
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    if bundle["model"] == "xgb":
        X = np.where(X == 0, np.nan, X)     # sparse training matrix: zero == missing
    probs = cal.transform(pipe.predict_proba(rows[bundle["features"]]))
    sv = shap.TreeExplainer(clf).shap_values(X)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv
    names = list(prep.get_feature_names_out())
    cols = [group_of(c, bundle["cats"])[0] if c.startswith("cat__") else c[5:] for c in names]
    grouped = []
    for i in range(n):
        g = {}
        for j, c in enumerate(cols):
            g.setdefault(c, np.zeros(3))
            g[c] = g[c] + sv[i, j, :]
        grouped.append({k: v.tolist() for k, v in g.items()})
    feats = rows[bundle["features"]].astype(object).where(rows[bundle["features"]].notna(), None)
    out = [{"row": {k: (v.item() if hasattr(v, "item") else v) for k, v in feats.iloc[i].items()},
            "probs": probs[i].tolist(), "shap": grouped[i]} for i in range(n)]
    p = ROOT / "app" / "scripts"
    p.mkdir(exist_ok=True)
    json.dump(out, open(p / "parity.json", "w"), default=float)


# ───────────────────────── results ─────────────────────────
def export_results(bundle, res):
    _, _, te = load_splits()
    y = te.severity.values
    P = np.load(MODELS_DIR / "test_proba.npy")
    scales = np.asarray(bundle["scales"])
    pred = (P * scales).argmax(1)
    cm = confusion(y, P, scales)

    curves = {}
    for k, name in enumerate(SEVERITY_ORDER):
        yy = (y == k).astype(int)
        fpr, tpr, _ = roc_curve(yy, P[:, k])
        pr, rc, _ = precision_recall_curve(yy, P[:, k])
        curves[name] = {"roc": thin(fpr, tpr), "pr": thin(rc[::-1], pr[::-1]),
                        "auc": float(roc_auc_score(yy, P[:, k])), "ap": float(average_precision_score(yy, P[:, k]))}

    # reliability for Fatal, before vs after calibration
    raw = bundle["pipeline"].predict_proba(te[bundle["features"]])
    def rel(p):
        q = pd.qcut(p[:, 2], 10, duplicates="drop")
        d = pd.DataFrame({"p": p[:, 2], "o": (y == 2).astype(int), "q": q}).groupby("q", observed=True).mean()
        return [{"p": round(a, 5), "o": round(b, 5)} for a, b in zip(d.p, d.o)]

    # error analysis by subgroup
    def sub(mask, name):
        m = mask.values if hasattr(mask, "values") else mask
        yy, pp = y[m], pred[m]
        return {"group": name, "n": int(m.sum()), "fatal_n": int((yy == 2).sum()),
                "recall_Fatal": float(((pp == 2) & (yy == 2)).sum() / max((yy == 2).sum(), 1)),
                "recall_KSI": float(((pp >= 1) & (yy >= 1)).sum() / max((yy >= 1).sum(), 1)),
                "accuracy": float((pp == yy).mean())}
    errs = [sub(np.ones(len(te), bool), "All test accidents"), sub(te.is_night == 1, "Night"), sub(te.is_night == 0, "Daytime"),
            sub(te.is_urban == 1, "Urban"), sub(te.is_urban == 0, "Rural"),
            sub(te.speed_limit <= 30, "Speed limit ≤ 30 mph"), sub(te.speed_limit.between(40, 50), "Speed limit 40-50 mph"),
            sub(te.speed_limit >= 60, "Speed limit ≥ 60 mph"),
            sub(te.Junction_Detail != "Not at junction or within 20 metres", "At a junction"),
            sub(te.Junction_Detail == "Not at junction or within 20 metres", "Not at a junction")]

    winner = bundle["model"]
    r = res[winner]
    base_acc = float((y == 0).mean())
    rs, bk = res["random_split"], res["binary_ksi"]
    dummy_f1 = float(pd.read_csv(TAB_DIR / "experiment_grid.csv").query("model=='dummy'").test_macro_f1.iloc[0])
    note = (f"On the 2016-17 test years {pct(base_acc)} of accidents are Slight, so a model that never predicts a fatality scores {pct(base_acc)} accuracy. "
            f"Our tuned model scores {pct(r['test_tuned']['accuracy'])} accuracy because it deliberately flags more serious cases ({pct(r['test_tuned']['recall_KSI'], 0)} of killed-or-seriously-injured accidents caught). "
            f"With plain argmax it reaches {pct(r['test_argmax']['accuracy'])}, and on a stratified random split, accuracy is {pct(rs['unweighted_argmax']['accuracy'])} against a {pct(rs['majority_accuracy'])} majority baseline.")
    cv = pd.read_csv(TAB_DIR / "cv_results.csv") if (TAB_DIR / "cv_results.csv").exists() else pd.DataFrame()
    abl = pd.read_csv(TAB_DIR / "ablations.csv") if (TAB_DIR / "ablations.csv").exists() else pd.DataFrame()
    protocols = [
        {"name": "Time-based test 2016-17 · tuned scales (headline)", "accuracy": r["test_tuned"]["accuracy"], "majority": base_acc,
         "macro_f1": r["test_tuned"]["macro_f1"], "auc": r["test_tuned"]["roc_auc_ovr"]},
        {"name": "Time-based test 2016-17 · plain argmax", "accuracy": r["test_argmax"]["accuracy"], "majority": base_acc,
         "macro_f1": r["test_argmax"]["macro_f1"], "auc": r["test_argmax"]["roc_auc_ovr"]},
        {"name": "Stratified random split (70/15/15) · unweighted argmax", "accuracy": rs["unweighted_argmax"]["accuracy"],
         "majority": rs["majority_accuracy"], "macro_f1": rs["unweighted_argmax"]["macro_f1"], "auc": rs["unweighted_argmax"]["roc_auc_ovr"]},
        {"name": "Stratified random split · tuned scales", "accuracy": rs["tuned"]["accuracy"], "majority": rs["majority_accuracy"],
         "macro_f1": rs["tuned"]["macro_f1"], "auc": rs["tuned"]["roc_auc_ovr"]},
        {"name": "Binary Fatal/Serious vs Slight · threshold 0.5", "accuracy": bk["argmax"]["accuracy"], "majority": bk["majority_accuracy"],
         "macro_f1": bk["argmax"]["macro_f1"], "auc": bk["argmax"]["roc_auc"]},
        {"name": "Binary Fatal/Serious vs Slight · tuned threshold", "accuracy": bk["tuned"]["accuracy"], "majority": bk["majority_accuracy"],
         "macro_f1": bk["tuned"]["macro_f1"], "auc": bk["tuned"]["roc_auc"]},
    ]
    final = {
        "model": NICE_MODEL[winner], "params": r["params"],
        "test": {"tuned": r["test_tuned"], "argmax": r["test_argmax"]},
        "baseline": {"accuracy": base_acc, "macro_f1": dummy_f1},
        "confusion": cm.tolist(),
        "confusion_note": "Rows are the true class. The tuned decision scales trade some Slight precision for much higher recall on Serious and Fatal accidents.",
        "curves": curves, "ece": {"raw": r["ece_test_raw"], "calibrated": r["ece_test_calibrated"]},
        "reliability": {"raw": rel(raw), "calibrated": rel(P)},
        "errors": errs, "accuracy_note": note, "protocols": protocols,
        "cv": [{"model": x.model, "mean": x.cv_macro_f1_mean, "std": x.cv_macro_f1_std, "p": 0.0 if pd.isna(x.p_value_vs_best) else x.p_value_vs_best,
                "best": bool(x.best)} for x in cv.itertuples()] if len(cv) else [],
        "ablations": abl[["variant", "macro_f1", "roc_auc_ovr", "accuracy", "recall_Fatal"]].to_dict("records") if len(abl) else [],
    }
    dump("final", final)
    grid = pd.read_csv(TAB_DIR / "experiment_grid.csv")
    keep = ["model", "imbalance", "decision", "fit_seconds", "infer_ms_per_row", "val_macro_f1", "test_macro_f1", "test_accuracy",
            "test_balanced_accuracy", "test_mcc", "test_recall_Fatal", "test_recall_Serious", "test_roc_auc_ovr", "test_pr_auc_Fatal"]
    dump("experiments", grid[keep].round(5).to_dict("records"))
    return final


# ───────────────────────── EDA + data pages ─────────────────────────
def export_eda(df):
    sev = df.severity.map(dict(enumerate(SEVERITY_ORDER)))
    def share(col, order=None, rename=None):
        t = pd.crosstab(df[col], sev)[SEVERITY_ORDER]
        t = (t.div(t.sum(1), axis=0) * 100).round(2)
        if order is not None:
            t = t.reindex(order)
        t.index = [rename.get(i, i) if rename else i for i in t.index]
        return [{"name": str(i), **row} for i, row in t.iterrows()]
    hour = pd.crosstab(df.hour, sev)[SEVERITY_ORDER]
    hs = (hour.div(hour.sum(1), axis=0) * 100).round(2)
    yr = pd.crosstab(df.year, sev)[SEVERITY_ORDER]
    ys = (yr.div(yr.sum(1), axis=0) * 100).round(2)
    short = {"Darkness - lights lit": "Dark, lit", "Darkness - no lighting": "Dark, unlit", "Darkness - lighting unknown": "Dark, unknown",
             "Darkness - lights unlit": "Dark, lights off", "Daylight": "Daylight"}
    road = df.Road_Type.fillna("Unknown")
    keep_roads = road.value_counts().index[:6]
    tmp = pd.crosstab(road[road.isin(keep_roads)], sev[road.isin(keep_roads)])[SEVERITY_ORDER]
    tmp = (tmp.div(tmp.sum(1), axis=0) * 100).round(2)
    eda = {
        "hour": [{"name": f"{h:02d}", **{k: int(v) for k, v in r.items()}} for h, r in hour.iterrows()],
        "hour_share": [{"name": f"{h:02d}", "Serious": r.Serious, "Fatal": r.Fatal} for h, r in hs.iterrows()],
        "speed": share("speed_limit", rename={i: f"{i}" for i in [20, 30, 40, 50, 60, 70]}),
        "light": share("Light_Conditions", rename=short),
        "road": [{"name": i, **r} for i, r in tmp.iterrows()],
        "dow": share("dow", order=range(7), rename=dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))),
        "year": [{"name": str(y), "Serious": r.Serious, "Fatal": r.Fatal} for y, r in ys.iterrows()],
        "urban": share("Urban_or_Rural_Area"),
    }
    dump("eda", eda)


DICT = {
    "Accident_Index": ("Unique accident identifier (used only to join/trace rows)", "id", "Kept as key, never a feature"),
    "year": ("Calendar year of the accident", "int", "Drives the time-based split"),
    "date": ("Accident date", "date", "Parsed; month, weekday and season derived"),
    "severity": ("Target: 0 Slight, 1 Serious, 2 Fatal", "target", "Mapped from text labels"),
    "ksi": ("Killed or seriously injured (1) vs slight (0)", "target", "Derived from severity for the binary task"),
    "sev_weight": ("EPDO weight: Fatal 10, Serious 5, Slight 1", "weight", "Used for black-spot scoring only"),
    "lat": ("Latitude (WGS84)", "float", "Rows with missing/outside UK box dropped"),
    "lon": ("Longitude (WGS84)", "float", "Rows with missing/outside UK box dropped"),
    "Number_of_Casualties": ("People injured in the accident", "int", "EXCLUDED: a consequence of severity (leakage)"),
    "Did_Police_Officer_Attend_Scene_of_Accident": ("Whether police attended", "cat", "EXCLUDED: recorded after the event and tied to severity"),
    "1st_Road_Class": ("Class of the first road (Motorway, A, B, C, Unclassified)", "cat", "One-hot"),
    "2nd_Road_Class": ("Class of the intersecting road", "cat", "Missing → 'None'; one-hot"),
    "Junction_Detail": ("Junction layout (T, crossroads, roundabout …)", "cat", "One-hot"),
    "Junction_Control": ("How the junction is controlled", "cat", "'Data missing' → Unknown; one-hot"),
    "Light_Conditions": ("Daylight or darkness (lit / unlit)", "cat", "One-hot"),
    "Road_Surface_Conditions": ("Dry, wet, frost, snow, flood", "cat", "One-hot"),
    "Road_Type": ("Single / dual carriageway, roundabout, one-way, slip road", "cat", "One-hot"),
    "Weather_Conditions": ("Weather at the time", "cat", "One-hot"),
    "Urban_or_Rural_Area": ("Urban or rural", "cat", "'Unallocated' rows dropped (160)"),
    "Carriageway_Hazards": ("Object or animal on the road", "cat", "98% missing → 'None' level"),
    "Special_Conditions_at_Site": ("Roadworks, defects, oil …", "cat", "97% missing → 'None' level"),
    "Pedestrian_Crossing-Human_Control": ("Crossing controlled by a person", "cat", "Numeric code → category"),
    "Pedestrian_Crossing-Physical_Facilities": ("Crossing facilities (zebra, pelican …)", "cat", "Numeric code → category"),
    "la_district": ("Local authority district (416)", "cat", "One-hot with rare levels grouped"),
    "Police_Force": ("Reporting police force (51)", "cat", "One-hot with rare levels grouped"),
    "season": ("Winter / spring / summer / autumn", "cat", "Derived from month"),
    "speed_band": ("Low ≤30, medium 40-50, high ≥60 mph", "cat", "Derived from speed limit"),
    "hour": ("Hour of day 0-23", "int", "Parsed from time; 156 missing rows dropped"),
    "month": ("Month 1-12", "int", "Derived from date"),
    "dow": ("Day of week (0 = Monday)", "int", "Mapped from text"),
    "is_weekend": ("Saturday or Sunday", "flag", "Derived"),
    "is_rush_hour": ("07-09 or 16-19", "flag", "Derived"),
    "is_night": ("Any darkness category", "flag", "Derived from light conditions"),
    "speed_limit": ("Posted speed limit (mph)", "int", "Only 20-70 kept (invalid codes dropped)"),
    "n_vehicles": ("Vehicles involved, capped at 6", "int", "Outliers capped"),
    "is_urban": ("Urban = 1", "flag", "Derived"),
    "wet_surface": ("Wet or flooded surface", "flag", "Interaction flag"),
    "bad_weather": ("Rain, snow or fog", "flag", "Interaction flag"),
    "dark_and_unlit": ("Darkness with no working lighting", "flag", "Interaction flag"),
    "junction_at_night": ("At a junction and dark", "flag", "Interaction flag"),
    "high_speed_rural": ("≥60 mph limit on a rural road", "flag", "Interaction flag"),
    "hour_sin": ("sin(2π·hour/24)", "float", "Cyclical encoding"),
    "hour_cos": ("cos(2π·hour/24)", "float", "Cyclical encoding"),
    "h_cell_n": ("Earlier accidents in the same ~550 m grid cell", "float", "Built from years < accident year only"),
    "h_cell_ksi_rate": ("Smoothed share of earlier accidents there that were Serious/Fatal", "float", "Bayesian smoothing (m = 20)"),
    "h_cell_score": ("Earlier severity-weighted score of the cell", "float", "Years < accident year only"),
    "h_la_n": ("Earlier accidents in the district", "float", "Years < accident year only"),
    "h_la_ksi_rate": ("Smoothed earlier KSI rate in the district", "float", "Bayesian smoothing (m = 50)"),
    "h_la_score": ("Earlier severity-weighted score of the district", "float", "Years < accident year only"),
    "h_dist_hotspot_m": ("Metres to the nearest earlier hotspot cell (top 2%)", "float", "Haversine BallTree; years < accident year"),
    "h_in_hotspot": ("Within 300 m of an earlier hotspot", "flag", "Derived from the distance"),
}
INCLUSION = [
    ("Number_of_Casualties", "int", False, False, "Consequence of the accident: more casualties go hand in hand with a higher severity class."),
    ("Did_Police_Officer_Attend_Scene_of_Accident", "cat", False, False, "Recorded after the event; serious accidents are far more likely to be attended."),
    ("Accident_Severity", "target", False, False, "The label itself."),
    ("Location_Easting/Northing, LSOA, Local_Authority_(Highway), InScotland", "various", True, False, "Redundant with latitude/longitude, district and police force."),
    ("Road, junction, light, weather, surface, speed limit, urban/rural", "cat/int", True, True, "Scene description known at the time of the accident."),
    ("Date, time, weekday, month", "date/int", True, True, "Known at the time; used for temporal flags."),
    ("Number_of_Vehicles", "int", True, True, "Known from the scene, capped at 6."),
    ("Latitude / longitude", "float", True, True, "Location of the accident."),
    ("h_* spatial history", "float", True, True, "Only uses accidents from earlier years, so no outcome of the row itself leaks in."),
]
ENGINEERED = [
    {"group": "Time", "items": ["hour, month, weekday", "is_weekend, is_rush_hour (07-09, 16-19)", "is_night from light conditions", "season", "hour_sin / hour_cos (cyclical)"]},
    {"group": "Scene interactions", "items": ["dark_and_unlit", "wet_surface and bad_weather", "junction_at_night", "high_speed_rural", "speed_band (low / medium / high)"]},
    {"group": "Spatial history (leak-free)", "items": ["h_cell_n, h_cell_ksi_rate, h_cell_score", "h_la_n, h_la_ksi_rate, h_la_score", "h_dist_hotspot_m and h_in_hotspot", "all from years strictly before the accident"]},
]


def export_data(df, final):
    log = pd.read_csv(TAB_DIR / "cleaning_log.csv")
    dd = pd.read_csv(TAB_DIR / "data_dictionary_processed.csv").set_index("column")
    dictionary = []
    for col, (meaning, typ, handling) in DICT.items():
        if col in dd.index:
            dictionary.append({"feature": col, "meaning": meaning, "type": typ, "example": str(dd.loc[col, "example"])[:30],
                               "missing": f"{dd.loc[col, 'missing_pct']:.2f}", "handling": handling})
    ntr = int(df.year.between(2006, 2014).sum()); nva = int((df.year == 2015).sum()); nte = int(df.year.isin(TEST_YEARS).sum())
    shares = df.severity.value_counts(normalize=True)
    facts = [[f"{len(df):,}", "accidents after cleaning"], ["2005–2017", "13 years of police-reported injury accidents"],
             ["51 / 416", "police forces / local authorities"], [f"{shares[0]:.1%} / {shares[1]:.1%} / {shares[2]:.1%}", "Slight / Serious / Fatal"],
             [str(len(CATEGORICAL) + 27), "model input features"], ["Time-based", "split: past → future"],
             ["0.02%", "of rows lost in cleaning"], ["0", "post-accident columns used"]]
    dump("data", {"facts": facts, "funnel": log.to_dict("records"), "dictionary": dictionary,
                  "inclusion": [{"feature": a, "type": b, "pre": c, "used": d, "reason": e} for a, b, c, d, e in INCLUSION],
                  "engineered": ENGINEERED,
                  "splits": [{"name": "Train", "years": "2006-2014", "n": ntr, "use": "fit models (2005 only seeds history features)"},
                             {"name": "Validation", "years": "2015", "n": nva, "use": "tuning, calibration, decision scales"},
                             {"name": "Test", "years": "2016-2017", "n": nte, "use": "scored once at the end"}]})


def export_summary(final, df):
    bs = json.load(open(TAB_DIR / "blackspot_summary.json"))
    t = final["test"]["tuned"]
    dump("summary", {
        "dataset": {"rows": int(len(df))},
        "headline": {"macro_f1": t["macro_f1"], "recall_KSI": t["recall_KSI"], "roc_auc": t["roc_auc_ovr"],
                     "accuracy": t["accuracy"], "recall_Fatal": t["recall_Fatal"], "model": final["model"]},
        "blackspots": {"persistence_pct": f"{bs['top100_persistence_rate'] * 100:.0f}%", "kde_overlap": bs["kde_overlap_top50"],
                       "ksi_share_top100": bs["top100_mean_ksi_share"], "ksi_share_all": bs["overall_ksi_share"], "n": bs["n_black_spots"],
                       "dbscan_overlap": bs["dbscan_overlap_top50"], "top100_share": bs["top100_share_of_all_accidents"],
                       "control_pct": f"{bs['control_rank1000_persistence'] * 100:.0f}%"},
    })


def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def main():
    bundle = load_model()
    res = json.load(open(TAB_DIR / "final_results.json"))
    df = pd.read_parquet(PROC_DIR / "accidents_clean.parquet")
    model, columns, names = export_model(bundle, df)
    export_meta(columns, df)
    export_parity(bundle)
    final = export_results(bundle, res)
    export_eda(df)
    export_data(df, final)
    export_summary(final, df)
    figs = APP_DATA.parent / "figures"
    figs.mkdir(exist_ok=True)
    for p in FIG_DIR.glob("*.png"):
        shutil.copy(p, figs / p.name)
    size = sum(f.stat().st_size for f in DATA_OUT.glob("*.json")) / 1e6
    print(f"exported web data: {size:.1f} MB total, {len(model['trees'])} trees")


if __name__ == "__main__":
    main()
