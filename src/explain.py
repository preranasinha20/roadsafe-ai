"""Step 7: SHAP and LIME explanations of the final model.

Run:  python -m src.explain
Outputs figures to reports/figures, tables to reports/tables and JSON for the web app.
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pyproj import Transformer
from scipy.stats import spearmanr
from sklearn.neighbors import KDTree

from .modelio import load_model
from .config import *
from .pipeline import load_splits

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 10})
COL = {"Slight": "#34d399", "Serious": "#fbbf24", "Fatal": "#f43f5e"}
NICE = {"speed_limit": "Speed limit", "hour": "Hour of day", "lat": "Latitude", "lon": "Longitude",
        "h_cell_ksi_rate": "Local KSI history", "h_cell_n": "Local accident count",
        "h_cell_score": "Local severity score", "h_la_ksi_rate": "District KSI history",
        "h_la_n": "District accident count", "h_la_score": "District severity score",
        "h_dist_hotspot_m": "Distance to hotspot", "h_in_hotspot": "Inside hotspot",
        "n_vehicles": "Vehicles involved", "dow": "Day of week", "month": "Month",
        "Light_Conditions": "Light conditions", "Road_Type": "Road type", "Junction_Detail": "Junction type",
        "Junction_Control": "Junction control", "Weather_Conditions": "Weather",
        "Road_Surface_Conditions": "Road surface", "1st_Road_Class": "Road class",
        "2nd_Road_Class": "2nd road class", "Urban_or_Rural_Area": "Urban / rural",
        "la_district": "Local authority", "Police_Force": "Police force", "season": "Season",
        "speed_band": "Speed band", "is_night": "Night", "is_weekend": "Weekend",
        "is_rush_hour": "Rush hour", "wet_surface": "Wet surface", "bad_weather": "Bad weather",
        "dark_and_unlit": "Dark & unlit", "junction_at_night": "Junction at night",
        "high_speed_rural": "High-speed rural", "is_urban": "Urban", "hour_sin": "Hour (sin)",
        "hour_cos": "Hour (cos)", "Carriageway_Hazards": "Carriageway hazards",
        "Special_Conditions_at_Site": "Special conditions",
        "Pedestrian_Crossing-Human_Control": "Ped. crossing (human)",
        "Pedestrian_Crossing-Physical_Facilities": "Ped. crossing (physical)"}
nice = lambda c: NICE.get(c, c)


def group_of(names, cats, nums):
    """Map transformed column names back to their original feature."""
    out = []
    for n in names:
        if n.startswith("num__"):
            out.append(n[5:])
        else:
            body = n[5:]
            out.append(next(c for c in sorted(cats, key=len, reverse=True) if body.startswith(c + "_")))
    return np.array(out)


def main():
    bundle = load_model()
    pipe, cal, scales = bundle["pipeline"], bundle["calibrator"], bundle["scales"]
    cats, nums, cols = bundle["cats"], bundle["nums"], bundle["features"]
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    names = prep.get_feature_names_out()
    groups = group_of(names, cats, nums)
    ug = list(dict.fromkeys(groups))

    _, _, te = load_splits()
    samp = te.sample(6000, random_state=SEED)
    fat = te[te.severity == 2].sample(400, random_state=SEED)
    samp = pd.concat([samp, fat]).drop_duplicates("Accident_Index").reset_index(drop=True)
    Xs = prep.transform(samp[cols])
    Xd = Xs.toarray() if hasattr(Xs, "toarray") else np.asarray(Xs)
    if bundle["model"] == "xgb":
        Xd = np.where(Xd == 0, np.nan, Xd)      # trained on a sparse matrix: zero means missing
    print("computing SHAP on", len(samp), "rows,", Xd.shape[1], "columns", flush=True)
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(Xd)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv            # (n, f, 3)
    base = np.atleast_1d(expl.expected_value)

    # grouped (original-feature) SHAP
    G = np.zeros((len(samp), len(ug), 3))
    for j, g in enumerate(ug):
        G[:, j, :] = sv[:, groups == g, :].sum(1)

    # ---------- global importance ----------
    imp = {SEVERITY_ORDER[k]: np.abs(G[:, :, k]).mean(0) for k in range(3)}
    imp_df = pd.DataFrame(imp, index=ug)
    imp_df["overall"] = imp_df.mean(1)
    imp_df = imp_df.sort_values("overall", ascending=False)
    imp_df.to_csv(TAB_DIR / "shap_importance.csv")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5.6), sharey=False)
    for ax, k in zip(axs, SEVERITY_ORDER):
        d = imp_df[k].sort_values(ascending=False).head(12)[::-1]
        ax.barh([nice(i) for i in d.index], d.values, color=COL[k])
        ax.set_title(f"{k}: mean |SHAP|")
    plt.tight_layout(); plt.savefig(FIG_DIR / "20_shap_importance_per_class.png", bbox_inches="tight"); plt.close()

    # beeswarm for Fatal on the grouped features (value = raw numeric or encoded rank)
    plt.figure(figsize=(8, 6))
    top = list(imp_df.sort_values("Fatal", ascending=False).index[:14])
    fvals = []
    for g in top:
        v = samp[g]
        if v.dtype.kind in "OSU" or str(v.dtype) == "category":
            v = v.astype("category").cat.codes.astype(float)
        fvals.append(v.values.astype(float))
    ex = shap.Explanation(values=G[:, [ug.index(g) for g in top], 2],
                          base_values=np.full(len(samp), base[2]),
                          data=np.column_stack(fvals), feature_names=[nice(g) for g in top])
    shap.plots.beeswarm(ex, max_display=14, show=False, plot_size=(8, 6.2))
    plt.title("SHAP summary (Fatal class, grouped features)")
    plt.savefig(FIG_DIR / "21_shap_beeswarm_fatal.png", bbox_inches="tight"); plt.close()

    # ---------- dependence plots ----------
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    dep_json = {}
    for ax, g in zip(axs, ["speed_limit", "hour", "h_cell_ksi_rate"]):
        j = ug.index(g)
        x, y = samp[g].values.astype(float), G[:, j, 2]
        ax.scatter(x, y, s=5, alpha=.35, c=samp.is_night, cmap="coolwarm")
        ax.set_xlabel(nice(g)); ax.set_ylabel("SHAP value (Fatal)"); ax.set_title(f"Dependence: {nice(g)}")
        ix = np.random.RandomState(SEED).choice(len(x), 700, replace=False)
        dep_json[g] = {"label": nice(g), "x": x[ix].round(4).tolist(), "y": y[ix].round(4).tolist(),
                       "night": samp.is_night.values[ix].tolist()}
    plt.tight_layout(); plt.savefig(FIG_DIR / "22_shap_dependence_fatal.png", bbox_inches="tight"); plt.close()

    # ---------- local explanations ----------
    P = cal.transform(pipe.predict_proba(samp[cols]))
    pred = (P * scales).argmax(1)
    y = samp.severity.values
    cases = {
        "Fatal, correctly flagged": np.where((y == 2) & (pred == 2))[0],
        "Fatal, missed (predicted Slight)": np.where((y == 2) & (pred == 0))[0],
        "Serious, correctly flagged": np.where((y == 1) & (pred == 1))[0],
        "Slight, correctly predicted": np.where((y == 0) & (pred == 0))[0],
    }
    local_json = []
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (title, idx) in zip(axs.ravel(), cases.items()):
        i = int(idx[np.argmax(P[idx, 2])]) if "Fatal, correctly" in title else int(idx[0])
        k = int(pred[i]) if "missed" not in title else 2
        contrib = pd.Series(G[i, :, k], index=ug).sort_values(key=np.abs, ascending=False)
        show = contrib.head(8)[::-1]
        ax.barh([f"{nice(g)} = {samp.loc[i, g] if not isinstance(samp.loc[i, g], float) else round(samp.loc[i, g], 2)}"
                 for g in show.index], show.values, color=[COL["Fatal"] if v > 0 else "#6366f1" for v in show.values])
        ax.set_title(f"{title}\nP(Slight/Serious/Fatal) = {P[i].round(2).tolist()}", fontsize=9.5)
        ax.set_xlabel(f"SHAP contribution to {SEVERITY_ORDER[k]} (log-odds)")
        local_json.append({"title": title, "class": SEVERITY_ORDER[k], "probs": P[i].round(4).tolist(),
                           "contrib": [{"feature": nice(g), "value": str(samp.loc[i, g]), "shap": float(v)}
                                       for g, v in contrib.head(8).items()]})
    plt.tight_layout(); plt.savefig(FIG_DIR / "23_shap_local_cases.png", bbox_inches="tight"); plt.close()

    # ---------- LIME vs SHAP ----------
    from lime.lime_tabular import LimeTabularExplainer
    tr_s, _, _ = load_splits(sample=20_000)
    cat_idx = [cols.index(c) for c in cats]
    cat_names = {}
    enc_df = tr_s[cols].copy()
    for c in cats:
        enc_df[c] = enc_df[c].fillna("Unknown").astype("category")
        cat_names[cols.index(c)] = list(enc_df[c].cat.categories)
        enc_df[c] = enc_df[c].cat.codes
    lime_x = enc_df.values.astype(float)
    ex_lime = LimeTabularExplainer(lime_x, feature_names=cols, categorical_features=cat_idx,
                                   categorical_names=cat_names, class_names=SEVERITY_ORDER,
                                   discretize_continuous=True, random_state=SEED, mode="classification")

    def predict_fn(arr):
        d = pd.DataFrame(arr, columns=cols)
        for c in cats:
            cl = cat_names[cols.index(c)]
            d[c] = [cl[int(round(v))] if 0 <= int(round(v)) < len(cl) else "Unknown" for v in d[c]]
        return cal.transform(pipe.predict_proba(d))

    lime_rows, stab_rows = [], []
    chosen = [int(idx[np.argmax(P[idx, 2])]) for t, idx in list(cases.items())[:1]] + \
             [int(cases["Serious, correctly flagged"][0]), int(cases["Slight, correctly predicted"][0])]
    for i in chosen:
        k = int(pred[i])
        row = samp.loc[i, cols].copy()
        arr = row.copy()
        for c in cats:
            cl = cat_names[cols.index(c)]
            v = row[c] if isinstance(row[c], str) else "Unknown"
            arr[c] = cl.index(v) if v in cl else 0
        arr = arr.values.astype(float)
        shap_top = pd.Series(G[i, :, k], index=ug).abs().sort_values(ascending=False).index[:5].tolist()
        tops = []
        for seed in range(5):
            e = LimeTabularExplainer(lime_x, feature_names=cols, categorical_features=cat_idx,
                                     categorical_names=cat_names, class_names=SEVERITY_ORDER,
                                     discretize_continuous=True, random_state=seed, mode="classification")
            exp = e.explain_instance(arr, predict_fn, num_features=8, labels=[k], num_samples=1500)
            names_l = []
            for desc, w in exp.as_list(label=k):
                for c in cols:
                    if c in desc:
                        names_l.append(c); break
            tops.append(names_l[:5])
        base_top = tops[0]
        overlap_shap = len(set(base_top) & set(shap_top)) / 5
        jac = np.mean([len(set(base_top) & set(t)) / len(set(base_top) | set(t)) for t in tops[1:]])
        lime_rows.append({"case": int(i), "explained_class": SEVERITY_ORDER[k],
                          "shap_top5": ", ".join(nice(c) for c in shap_top),
                          "lime_top5": ", ".join(nice(c) for c in base_top),
                          "shap_lime_top5_overlap": overlap_shap, "lime_stability_jaccard": jac})
        print("LIME case", i, lime_rows[-1], flush=True)
    pd.DataFrame(lime_rows).to_csv(TAB_DIR / "lime_vs_shap.csv", index=False)

    # ---------- inside vs outside black spots ----------
    spots = pd.read_json(APP_DATA / "blackspots.json")
    tf = Transformer.from_crs(4326, 27700, always_xy=True)
    se, sn = tf.transform(spots.lon.values, spots.lat.values)
    pe, pn = tf.transform(samp.lon.values.astype(float), samp.lat.values.astype(float))
    tree = KDTree(np.c_[se, sn])
    dist, ix = tree.query(np.c_[pe, pn], k=1)
    r = np.minimum(spots.radius_m.values[ix[:, 0]], 400) + 50
    inside = dist[:, 0] <= r
    print("inside black spots:", inside.mean())
    io = pd.DataFrame({"inside": np.abs(G[inside, :, 2]).mean(0), "outside": np.abs(G[~inside, :, 2]).mean(0)},
                      index=ug)
    io["inside_share"] = io.inside / io.inside.sum()
    io["outside_share"] = io.outside / io.outside.sum()
    io = io.sort_values("outside_share", ascending=False)
    io.to_csv(TAB_DIR / "shap_inside_vs_outside.csv")
    top = io.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    yy = np.arange(len(top))
    ax.barh(yy - .2, top.outside_share * 100, .4, label="outside black spots", color="#6366f1")
    ax.barh(yy + .2, top.inside_share * 100, .4, label="inside black spots", color="#f43f5e")
    ax.set_yticks(yy); ax.set_yticklabels([nice(i) for i in top.index]); ax.legend(frameon=False)
    ax.set_xlabel("share of total |SHAP| for Fatal (%)"); ax.set_title("What drives Fatal risk: inside vs outside black spots")
    plt.tight_layout(); plt.savefig(FIG_DIR / "24_shap_inside_vs_outside.png", bbox_inches="tight"); plt.close()

    # ---------- JSON for the web ----------
    web = {
        "importance": {k: [{"feature": nice(g), "value": float(imp_df.loc[g, k])} for g in
                           imp_df[k].sort_values(ascending=False).index[:14]] for k in SEVERITY_ORDER},
        "overall": [{"feature": nice(g), "value": float(v)} for g, v in imp_df.overall.head(14).items()],
        "dependence": dep_json, "local": local_json,
        "inside_outside": [{"feature": nice(g), "inside": float(r_.inside_share), "outside": float(r_.outside_share)}
                           for g, r_ in io.head(10).iterrows()],
        "inside_share_of_accidents": float(inside.mean()),
        "inside_ksi_rate": float((samp.ksi.values[inside]).mean()), "outside_ksi_rate": float((samp.ksi.values[~inside]).mean()),
        "lime": lime_rows,
    }
    json.dump(web, open(APP_DATA / "explain.json", "w"))
    np.save(MODELS_DIR / "shap_sample_idx.npy", samp.index.values)
    print("explain done")


if __name__ == "__main__":
    main()
