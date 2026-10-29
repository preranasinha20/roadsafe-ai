"""Step 4: black-spot detection, validation and sensitivity analysis.

Run:  python -m src.blackspots

A black spot is a compact site (a few hundred metres) with an unusually high, severity-weighted
concentration of injury accidents. Detection window = 2005-2014, validation window = 2015-2017.

Methods
  A. Sliding-window peak detection (PRIMARY). Every accident is scored by the EPDO-weighted
     accidents within R metres (Fatal 10, Serious 5, Slight 1). Peaks are picked greedily and
     neighbours within 2R are suppressed, giving disjoint, compact sites like those used in UK
     practice.
  B. DBSCAN / HDBSCAN on the coordinates. Informative, but on road data they chain along
     corridors into huge clusters, which the sensitivity table documents.
  C. Weighted KDE on a 100 m grid, used as an independent cross-check.
Coordinates are projected to the British National Grid so that distances are real metres.
"""
import json
import warnings

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

from .config import *

warnings.filterwarnings("ignore")
DETECT = (2005, 2014)
VALID = (2015, 2017)
FINAL = dict(radius=150, min_accidents=15)     # chosen after the sensitivity study


def load_points():
    cols = ["year", "lat", "lon", "severity", "sev_weight", "la_district", "Light_Conditions",
            "Road_Type", "speed_limit", "Junction_Detail", "hour", "is_night", "Weather_Conditions",
            "Urban_or_Rural_Area", "1st_Road_Class"]
    df = pd.read_parquet(PROC_DIR / "accidents_clean.parquet", columns=cols)
    tf = Transformer.from_crs(4326, 27700, always_xy=True)
    e, n = tf.transform(df.lon.values.astype("float64"), df.lat.values.astype("float64"))
    df["e"], df["n"] = e, n
    return df


# ───────────── Method A: sliding-window peaks ─────────────
def window_scores(xy, w, radius, chunk=60_000):
    """Count and EPDO score of accidents within `radius` metres of every accident."""
    tree = KDTree(xy)
    cnt = np.zeros(len(xy), dtype=np.int32)
    score = np.zeros(len(xy), dtype=np.float32)
    for s in range(0, len(xy), chunk):
        nb = tree.query_radius(xy[s:s + chunk], r=radius)
        sizes = np.fromiter((len(a) for a in nb), dtype=np.int64, count=len(nb))
        flat = np.concatenate(nb)
        cnt[s:s + chunk] = sizes
        score[s:s + chunk] = np.add.reduceat(w[flat], np.r_[0, np.cumsum(sizes)[:-1]])
    return cnt, score, tree


WEIGHT_SCHEMES = {"10/5/1 (final)": (10, 5, 1), "5/2/1": (5, 2, 1), "20/5/1": (20, 5, 1), "1/1/1 (counts only)": (1, 1, 1)}


def detect_peaks(pts, radius=150, min_accidents=15, max_spots=6000, weights=(10, 5, 1)):
    xy = pts[["e", "n"]].values
    # severity codes are 0 Slight, 1 Serious, 2 Fatal; weights are given as (Fatal, Serious, Slight)
    w = np.array([weights[2], weights[1], weights[0]], dtype="float32")[pts.severity.values]
    cnt, score, tree = window_scores(xy, w, radius)
    cand = np.where(cnt >= min_accidents)[0]
    order = cand[np.argsort(-score[cand], kind="stable")]
    free = np.ones(len(pts), dtype=bool)
    spots = []
    for i in order:
        if not free[i]:
            continue
        near = tree.query_radius(xy[i:i + 1], r=radius)[0]
        centre = xy[near].mean(0)
        members = tree.query_radius(centre[None, :], r=radius)[0]
        spots.append((centre[0], centre[1], members))
        free[tree.query_radius(xy[i:i + 1], r=2 * radius)[0]] = False
        if len(spots) >= max_spots:
            break
    return spots


def summarise_spots(pts, spots, years, radius):
    rows = []
    for k, (ce, cn, idx) in enumerate(spots):
        d = pts.iloc[idx]
        rows.append({
            "cluster_id": k, "accidents": len(d), "score": int(d.sev_weight.sum()),
            "fatal": int((d.severity == 2).sum()), "serious": int((d.severity == 1).sum()),
            "e": ce, "n": cn, "lat": d.lat.mean(), "lon": d.lon.mean(),
            "night_share": d.is_night.mean(), "peak_hour": int(d.hour.mode().iat[0]),
            "speed_limit": int(d.speed_limit.mode().iat[0]), "road_type": d.Road_Type.mode().iat[0],
            "junction": d.Junction_Detail.mode().iat[0], "road_class": d["1st_Road_Class"].mode().iat[0],
            "light": d.Light_Conditions.mode().iat[0], "urban": d.Urban_or_Rural_Area.mode().iat[0],
            "la": d.la_district.mode().iat[0]})
    out = pd.DataFrame(rows)
    out["ksi_share"] = (out.fatal + out.serious) / out.accidents
    out["per_year"] = out.accidents / years
    out["radius_m"] = radius
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def validate(spots_df, valid_pts, years_valid, radius, top_k=None):
    s = spots_df if top_k is None else spots_df.head(top_k)
    tree = KDTree(valid_pts[["e", "n"]].values)
    w = valid_pts.sev_weight.values
    nb = tree.query_radius(s[["e", "n"]].values, r=radius)
    s = s.copy()
    s["valid_accidents"] = [len(a) for a in nb]
    s["valid_score"] = [w[a].sum() for a in nb]
    s["valid_per_year"] = s.valid_accidents / years_valid
    s["ratio"] = s.valid_per_year / s.per_year
    # persistent: still >= 1 accident a year AND at least half of the earlier annual rate
    s["persistent"] = (s.valid_per_year >= 1) & (s.ratio >= 0.5)
    return s


# ───────────── Method B: DBSCAN / HDBSCAN (comparison) ─────────────
def dbscan_stats(pts, eps, ms):
    loc = pts.assign(e_=pts.e.round(0), n_=pts.n.round(0)).groupby(["e_", "n_"]).agg(
        c=("severity", "size")).reset_index()
    lab = DBSCAN(eps=eps, min_samples=ms, algorithm="kd_tree", n_jobs=2).fit_predict(
        loc[["e_", "n_"]].values, sample_weight=loc.c.values)
    size = loc.c.groupby(lab).sum().drop(-1, errors="ignore")
    ext = loc.assign(l=lab).query("l >= 0").groupby("l").agg(
        de=("e_", lambda s: s.max() - s.min()), dn=("n_", lambda s: s.max() - s.min()))
    span = np.hypot(ext.de, ext.dn)
    return {"clusters": int(len(size)), "noise_share": float(loc.c[lab < 0].sum() / loc.c.sum()),
            "median_cluster_accidents": float(size.median()), "largest_cluster_accidents": int(size.max()),
            "median_span_km": float(span.median() / 1000), "p90_span_km": float(span.quantile(.9) / 1000),
            "max_span_km": float(span.max() / 1000)}, lab, loc


def hdbscan_stats(pts):
    import hdbscan
    sample = pts.sample(500_000, random_state=SEED)
    lab = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=15, cluster_selection_method="leaf",
                          algorithm="boruvka_kdtree", core_dist_n_jobs=2).fit_predict(sample[["e", "n"]].values)
    d = sample.assign(l=lab).query("l >= 0")
    span = np.hypot(d.groupby("l").e.agg(lambda s: s.max() - s.min()), d.groupby("l").n.agg(lambda s: s.max() - s.min()))
    return {"clusters": int(lab.max() + 1), "noise_share": float((lab < 0).mean()),
            "median_span_km": float(span.median() / 1000), "p90_span_km": float(span.quantile(.9) / 1000),
            "max_span_km": float(span.max() / 1000)}


# ───────────── Method C: KDE ─────────────
def kde_grid(pts, cell=100, sigma_cells=2, quantile=0.9995):
    e0, n0 = pts.e.min(), pts.n.min()
    ix = ((pts.e - e0) // cell).astype(int).values
    iy = ((pts.n - n0) // cell).astype(int).values
    H = np.zeros((ix.max() + 1, iy.max() + 1), dtype="float32")
    np.add.at(H, (ix, iy), pts.sev_weight.values.astype("float32"))
    K = gaussian_filter(H, sigma=sigma_cells)
    return K, np.quantile(K[K > 0], quantile), (e0, n0, cell)


def kde_overlap(spots_df, K, thr, origin, top_k):
    e0, n0, cell = origin
    top = spots_df.head(top_k)
    hit = []
    for e, n in zip(top.e, top.n):
        a = int(np.clip((e - e0) // cell, 0, K.shape[0] - 1))
        b = int(np.clip((n - n0) // cell, 0, K.shape[1] - 1))
        hit.append(K[max(a - 2, 0):a + 3, max(b - 2, 0):b + 3].max() >= thr)
    return float(np.mean(hit))


def main():
    pts = load_points()
    det = pts[pts.year.between(*DETECT)].reset_index(drop=True)
    val = pts[pts.year.between(*VALID)].reset_index(drop=True)
    ny_d, ny_v = DETECT[1] - DETECT[0] + 1, VALID[1] - VALID[0] + 1
    print(f"detection {len(det):,} accidents | validation {len(val):,}", flush=True)

    # ---- sensitivity of the primary method: radius, then severity weights ----
    rows, kept = [], {}

    def run(radius, scheme):
        spots = detect_peaks(det, radius, FINAL["min_accidents"], weights=WEIGHT_SCHEMES[scheme])
        sdf = summarise_spots(det, spots, ny_d, radius)
        # rank by the scheme's own score so the top-100 reflects that weighting
        w = np.array([WEIGHT_SCHEMES[scheme][2], WEIGHT_SCHEMES[scheme][1], WEIGHT_SCHEMES[scheme][0]])
        sdf["scheme_score"] = sdf.fatal * w[2] + sdf.serious * w[1] + (sdf.accidents - sdf.fatal - sdf.serious) * w[0]
        sdf = sdf.sort_values("scheme_score", ascending=False).reset_index(drop=True)
        v = validate(sdf, val, ny_v, radius, top_k=100)
        kept[(radius, scheme)] = sdf
        rows.append({"radius_m": radius, "weights_fatal_serious_slight": scheme, "sites_detected": len(sdf),
                     "median_site_accidents": float(sdf.accidents.median()),
                     "top100_mean_ksi_share": float(sdf.head(100).ksi_share.mean()),
                     "top100_persistence_rate": float(v.persistent.mean()),
                     "top100_median_ratio": float(v.ratio.median()),
                     "spearman_score_vs_later": float(spearmanr(v.score, v.valid_score)[0])})
        print(rows[-1], flush=True)

    for radius in (100, 150, 200, 300):
        run(radius, "10/5/1 (final)")
    for scheme in ("5/2/1", "20/5/1", "1/1/1 (counts only)"):
        run(FINAL["radius"], scheme)
    pd.DataFrame(rows).to_csv(TAB_DIR / "blackspot_sensitivity.csv", index=False)

    # ---- ranking stability: how many of the final top-20 sites reappear under other settings ----
    base = kept[(FINAL["radius"], "10/5/1 (final)")].head(20)
    bt = KDTree(base[["e", "n"]].values)
    stab = []
    for (radius, scheme), sdf in kept.items():
        d, _ = bt.query(sdf.head(20)[["e", "n"]].values, k=1)
        stab.append({"radius_m": radius, "weights": scheme, "top20_shared_with_final": float((d[:, 0] < 400).mean())})
    pd.DataFrame(stab).to_csv(TAB_DIR / "blackspot_rank_stability.csv", index=False)

    # ---- comparison with DBSCAN / HDBSCAN: chaining on road networks ----
    comp = []
    for eps, ms in [(100, 25), (150, 30), (200, 40), (300, 60)]:
        st, _, _ = dbscan_stats(det, eps, ms)
        comp.append({"method": "DBSCAN", "params": f"eps={eps} m, min_samples={ms}", **st})
        print(comp[-1], flush=True)
    comp.append({"method": "HDBSCAN (leaf)", "params": "min_cluster=30, min_samples=15, 500k sample", **hdbscan_stats(det)})
    print(comp[-1], flush=True)
    pd.DataFrame(comp).to_csv(TAB_DIR / "blackspot_method_comparison.csv", index=False)

    # ---- final configuration ----
    R = FINAL["radius"]
    ranked = kept[(R, "10/5/1 (final)")]
    v100 = validate(ranked, val, ny_v, R, top_k=100)
    K, thr, origin = kde_grid(det)
    overlap = kde_overlap(ranked, K, thr, origin, 50)

    # DBSCAN agreement: how many of the top-50 sites sit inside a DBSCAN cluster (eps=150 m, 30)
    _, lab, loc = dbscan_stats(det, 150, 30)
    in_cl = loc[lab >= 0]
    ct = KDTree(in_cl[["e_", "n_"]].values)
    d, _ = ct.query(ranked.head(50)[["e", "n"]].values, k=1)
    dbscan_overlap = float((d[:, 0] <= 150).mean())

    # rolling-origin check: detect 2005-11, validate 2012-14
    det2, val2 = pts[pts.year.between(2005, 2011)].reset_index(drop=True), pts[pts.year.between(2012, 2014)].reset_index(drop=True)
    r2 = summarise_spots(det2, detect_peaks(det2, R, FINAL["min_accidents"]), 7, R)
    t2 = validate(r2, val2, 3, R, top_k=100)

    # control: persistence of ordinary dense sites (ranks 1000-1100) for context
    ctrl = validate(ranked.iloc[min(1000, len(ranked) - 100):min(1100, len(ranked))], val, ny_v, R)

    summary = {
        "final_config": {"method": "sliding-window peaks", "radius_m": R, "min_accidents": FINAL["min_accidents"],
                         "severity_weights": SEV_WEIGHT},
        "detection_years": list(DETECT), "validation_years": list(VALID),
        "n_black_spots": int(len(ranked)),
        "top100_persistence_rate": float(v100.persistent.mean()),
        "top100_median_ratio": float(v100.ratio.median()),
        "top100_mean_ksi_share": float(ranked.head(100).ksi_share.mean()),
        "overall_ksi_share": float((det.severity >= 1).mean()),
        "kde_overlap_top50": overlap, "dbscan_overlap_top50": dbscan_overlap,
        "rolling_2005_11_vs_2012_14_persistence_top100": float(t2.persistent.mean()),
        "rolling_median_ratio": float(t2.ratio.median()),
        "control_rank1000_persistence": float(ctrl.persistent.mean()),
        "top100_share_of_all_accidents": float(ranked.head(100).accidents.sum() / len(det)),
    }
    json.dump(summary, open(TAB_DIR / "blackspot_summary.json", "w"), indent=2)
    v100.to_csv(TAB_DIR / "blackspots_top100.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)

    # ---- data for the web map ----
    web = validate(ranked.head(500), val, ny_v, R)
    keep = ["cluster_id", "lat", "lon", "accidents", "score", "fatal", "serious", "ksi_share", "per_year", "radius_m",
            "night_share", "peak_hour", "speed_limit", "road_type", "junction", "road_class", "light", "urban", "la",
            "valid_per_year", "persistent"]
    web = web[keep].round(5)
    web.insert(0, "rank", np.arange(1, len(web) + 1))
    web.to_json(APP_DATA / "blackspots.json", orient="records")
    heat = det.assign(la_=det.lat.round(2), lo_=det.lon.round(2)).groupby(["la_", "lo_"]).sev_weight.sum().reset_index()
    heat["w"] = np.log1p(heat.sev_weight) / np.log1p(heat.sev_weight).max()
    json.dump(heat[["la_", "lo_", "w"]].round(3).values.tolist(), open(APP_DATA / "heat.json", "w"))
    print("saved web map data", len(web), len(heat))


if __name__ == "__main__":
    main()
