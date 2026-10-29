"""Step 1: clean the raw DfT file and engineer scene-level features.

Run:  python -m src.data_prep
Every cleaning decision logs its row count to reports/tables/cleaning_log.csv
so the report can quote the exact numbers.
"""
import json

import numpy as np
import pandas as pd

from .config import *

MISSING_TOKENS = ["Data missing or out of range", "Unknown", "unknown"]

CAT_COLS = [
    "1st_Road_Class", "2nd_Road_Class", "Junction_Detail", "Junction_Control",
    "Light_Conditions", "Road_Surface_Conditions", "Road_Type", "Weather_Conditions",
    "Urban_or_Rural_Area", "Carriageway_Hazards", "Special_Conditions_at_Site",
    "Pedestrian_Crossing-Human_Control", "Pedestrian_Crossing-Physical_Facilities",
]
DAY_NUM = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
           "Friday": 4, "Saturday": 5, "Sunday": 6}


def load_raw():
    usecols = [
        "Accident_Index", "1st_Road_Class", "1st_Road_Number", "2nd_Road_Class",
        "Accident_Severity", "Carriageway_Hazards", "Date", "Day_of_Week",
        "Junction_Control", "Junction_Detail", "Latitude", "Longitude",
        "Light_Conditions", "Local_Authority_(District)", "Number_of_Vehicles",
        "Number_of_Casualties", "Did_Police_Officer_Attend_Scene_of_Accident",
        "Pedestrian_Crossing-Human_Control", "Pedestrian_Crossing-Physical_Facilities",
        "Police_Force", "Road_Surface_Conditions", "Road_Type",
        "Special_Conditions_at_Site", "Speed_limit", "Time", "Urban_or_Rural_Area",
        "Weather_Conditions", "Year",
    ]
    return pd.read_csv(RAW_CSV, usecols=usecols, low_memory=False)


def clean(df):
    log = []

    def note(step, frame):
        log.append({"step": step, "rows": len(frame)})

    note("raw file", df)

    df = df.drop_duplicates(subset="Accident_Index")
    note("drop duplicate accident ids", df)

    df = df.dropna(subset=["Latitude", "Longitude"])
    note("drop rows with missing latitude/longitude", df)

    lat_ok = df.Latitude.between(*UK_BBOX["lat"])
    lon_ok = df.Longitude.between(*UK_BBOX["lon"])
    df = df[lat_ok & lon_ok]
    note("keep coordinates inside UK bounding box", df)

    df = df.dropna(subset=["Time"])
    note("drop rows with missing time of day", df)

    df = df[df.Speed_limit.isin([20, 30, 40, 50, 60, 70])]
    note("keep valid speed limits (20-70 mph)", df)

    df = df.copy()
    for c in CAT_COLS:
        df[c] = df[c].astype("object").replace(MISSING_TOKENS, np.nan)
    # the two pedestrian crossing columns are numeric codes in the raw file
    for c in ["Pedestrian_Crossing-Human_Control", "Pedestrian_Crossing-Physical_Facilities"]:
        df[c] = df[c].apply(lambda v: np.nan if pd.isna(v) else f"code_{int(float(v))}"
                            if not isinstance(v, str) else v)
    # absent hazard / special condition is a genuine "None" level, not missing data
    df["Carriageway_Hazards"] = df["Carriageway_Hazards"].fillna("None")
    df["Special_Conditions_at_Site"] = df["Special_Conditions_at_Site"].fillna("None")
    df["2nd_Road_Class"] = df["2nd_Road_Class"].fillna("None")
    df = df[df.Urban_or_Rural_Area.isin(["Urban", "Rural"])]
    note("drop 'Unallocated' urban/rural rows", df)
    return df, pd.DataFrame(log)


def engineer(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["Date"])
    t = df["Time"].str.split(":", expand=True).astype(int)
    df["hour"] = t[0]
    df["month"] = df.date.dt.month
    df["dow"] = df["Day_of_Week"].map(DAY_NUM)
    df["is_weekend"] = (df.dow >= 5).astype(int)
    df["is_rush_hour"] = df.hour.isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df["is_night"] = df.Light_Conditions.fillna("").str.startswith("Darkness").astype(int)
    df["season"] = df.month.map({12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring",
                                 5: "spring", 6: "summer", 7: "summer", 8: "summer",
                                 9: "autumn", 10: "autumn", 11: "autumn"})
    df["speed_limit"] = df.Speed_limit.astype(int)
    df["speed_band"] = pd.cut(df.speed_limit, [0, 30, 50, 100], labels=["low", "medium", "high"]).astype(str)
    df["n_vehicles"] = df.Number_of_Vehicles.clip(upper=6)
    df["is_urban"] = (df.Urban_or_Rural_Area == "Urban").astype(int)
    wet = df.Road_Surface_Conditions.isin(["Wet or damp", "Flood over 3cm. deep"])
    df["wet_surface"] = wet.astype(int)
    df["bad_weather"] = df.Weather_Conditions.isin(
        ["Raining no high winds", "Raining + high winds", "Snowing no high winds",
         "Snowing + high winds", "Fog or mist"]).astype(int)
    unlit = df.Light_Conditions.isin(["Darkness - no lighting", "Darkness - lights unlit"])
    df["dark_and_unlit"] = unlit.astype(int)
    at_junction = ~df.Junction_Detail.isin(["Not at junction or within 20 metres"]).fillna(False)
    df["junction_at_night"] = (at_junction & (df.is_night == 1)).astype(int)
    df["high_speed_rural"] = ((df.speed_limit >= 60) & (df.is_urban == 0)).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)

    df["severity"] = df.Accident_Severity.map(SEV_MAP).astype(int)
    df["ksi"] = (df.severity >= 1).astype(int)            # Killed or Seriously Injured
    df["sev_weight"] = df.Accident_Severity.map(SEV_WEIGHT)
    df["lat"] = df.Latitude.astype("float32")
    df["lon"] = df.Longitude.astype("float32")
    df["la_district"] = df["Local_Authority_(District)"]
    df["road_key"] = (df["1st_Road_Class"].fillna("NA") + "_" + df["1st_Road_Number"].astype(str)
                      + "_" + df.la_district)
    df["year"] = df.Year.astype(int)
    return df


def add_history_features(df):
    """Leak-free spatial history.

    For an accident in year Y, every statistic is built ONLY from accidents in years < Y
    (an expanding window). A training row therefore never sees its own outcome, and the
    validation/test rows only use history that would have been known at the time.
    Implemented with integer codes and running totals to keep memory low.
    """
    from sklearn.neighbors import BallTree

    df = df.sort_values("year", kind="stable").reset_index(drop=True)
    cell = (np.floor(df.lat / GRID_DEG).astype(int).astype(str) + "_"
            + np.floor(df.lon / GRID_DEG).astype(int).astype(str))
    df["cell"] = cell
    codes = {}
    for p, col in {"cell": "cell", "road": "road_key", "la": "la_district"}.items():
        codes[p], _ = pd.factorize(df[col], use_na_sentinel=False)
    ksi = df.ksi.values.astype("float64")
    wgt = df.sev_weight.values.astype("float64")
    lat = df.lat.values.astype("float64")
    lon = df.lon.values.astype("float64")
    yr = df.year.values
    n_rows = len(df)

    out = {f"{p}_{s}": np.full(n_rows, np.nan, dtype="float32")
           for p in codes for s in ("n", "ksi_rate", "score")}
    out["dist_hotspot_m"] = np.full(n_rows, np.nan, dtype="float32")
    out["in_hotspot"] = np.zeros(n_rows, dtype="int8")

    size = {p: c.max() + 1 for p, c in codes.items()}
    cum = {p: {"n": np.zeros(size[p]), "k": np.zeros(size[p]), "s": np.zeros(size[p])} for p in codes}
    cell_lat = np.zeros(size["cell"])
    cell_lon = np.zeros(size["cell"])

    def absorb(idx):
        for p, c in codes.items():
            cum[p]["n"] += np.bincount(c[idx], minlength=size[p])
            cum[p]["k"] += np.bincount(c[idx], weights=ksi[idx], minlength=size[p])
            cum[p]["s"] += np.bincount(c[idx], weights=wgt[idx], minlength=size[p])
        cell_lat[:] += np.bincount(codes["cell"][idx], weights=lat[idx], minlength=size["cell"])
        cell_lon[:] += np.bincount(codes["cell"][idx], weights=lon[idx], minlength=size["cell"])

    years = sorted(set(yr))
    absorb(np.where(yr == years[0])[0])
    for y in years[1:]:
        cur = np.where(yr == y)[0]
        total_n = cum["la"]["n"].sum()
        prior_ksi = cum["la"]["k"].sum() / total_n
        for p, c in codes.items():
            m = 50 if p == "la" else 20                      # Bayesian smoothing strength
            n, k, s = cum[p]["n"][c[cur]], cum[p]["k"][c[cur]], cum[p]["s"][c[cur]]
            out[f"{p}_n"][cur] = n
            out[f"{p}_ksi_rate"][cur] = (k + m * prior_ksi) / (n + m)
            out[f"{p}_score"][cur] = s
        # hotspot cells: top 2% weighted-score cells (with >= 5 prior accidents)
        cn, cs = cum["cell"]["n"], cum["cell"]["s"]
        ok = cn >= 5
        thr = np.quantile(cs[ok], 0.98)
        top = np.where(ok & (cs >= thr))[0]
        tree = BallTree(np.radians(np.c_[cell_lat[top] / cn[top], cell_lon[top] / cn[top]]),
                        metric="haversine")
        d, _ = tree.query(np.radians(np.c_[lat[cur], lon[cur]]), k=1)
        dist = d[:, 0] * 6371008.8
        out["dist_hotspot_m"][cur] = dist
        out["in_hotspot"][cur] = (dist < 300).astype("int8")
        absorb(cur)
        print(f"  history features built for {y}", flush=True)

    for k, v in out.items():
        df[f"h_{k}"] = v
    return df


def main():
    print("loading raw file ...")
    raw = load_raw()
    df, log = clean(raw)
    del raw
    df = engineer(df)
    print("adding historical spatial features ...")
    df = add_history_features(df)

    keep = (["Accident_Index", "year", "date", "severity", "ksi", "sev_weight", "lat", "lon",
             "Number_of_Casualties", "Did_Police_Officer_Attend_Scene_of_Accident"]
            + CAT_COLS + ["la_district", "Police_Force", "road_key", "cell", "season", "speed_band",
                          "hour", "month", "dow", "is_weekend", "is_rush_hour", "is_night",
                          "speed_limit", "n_vehicles", "is_urban", "wet_surface", "bad_weather",
                          "dark_and_unlit", "junction_at_night", "high_speed_rural",
                          "hour_sin", "hour_cos"]
            + [c for c in df.columns if c.startswith("h_")])
    df = df[keep].reset_index(drop=True)
    df.to_parquet(PROC_DIR / "accidents_clean.parquet", index=False)
    log.to_csv(TAB_DIR / "cleaning_log.csv", index=False)

    dd = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_pct": (df.isna().mean() * 100).round(3).values,
        "n_unique": df.nunique().values,
        "example": [str(df[c].dropna().iloc[0]) for c in df.columns],
    })
    dd.to_csv(TAB_DIR / "data_dictionary_processed.csv", index=False)
    with open(TAB_DIR / "dataset_summary.json", "w") as f:
        json.dump({"rows": int(len(df)), "cols": int(df.shape[1]),
                   "years": [int(df.year.min()), int(df.year.max())],
                   "class_share": df.severity.map(dict(enumerate(SEVERITY_ORDER)))
                   .value_counts(normalize=True).round(4).to_dict()}, f, indent=2)
    print(log.to_string(index=False))
    print(df.shape)


if __name__ == "__main__":
    main()
