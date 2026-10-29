"""Step 2: exploratory plots for the report. Run: python -m src.eda"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import *

COL = {"Slight": "#34d399", "Serious": "#fbbf24", "Fatal": "#f43f5e"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 10})


def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close()


def cramers_v(a, b):
    ct = pd.crosstab(a, b)
    chi2 = ((ct - ct.sum(1).values[:, None] * ct.sum(0).values / ct.values.sum()) ** 2
            / (ct.sum(1).values[:, None] * ct.sum(0).values / ct.values.sum())).values.sum()
    n = ct.values.sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1)))


def main():
    df = pd.read_parquet(PROC_DIR / "accidents_clean.parquet")
    df["sev"] = df.severity.map(dict(enumerate(SEVERITY_ORDER)))
    pct = lambda s: s.div(s.sum(1), axis=0) * 100

    # 1 class distribution
    vc = df.sev.value_counts().reindex(SEVERITY_ORDER)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(vc.index, vc.values, color=[COL[k] for k in vc.index])
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,}\n({v / vc.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, vc.max() * 1.18)
    ax.set_title("Severity class distribution")
    save("01_class_distribution")

    # 2 hour / weekday
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    h = pd.crosstab(df.hour, df.sev)[SEVERITY_ORDER]
    h[["Slight"]].plot.area(ax=axs[0], color=COL["Slight"], alpha=.5, legend=False)
    ax2 = axs[0].twinx()
    (pct(h)[["Serious", "Fatal"]]).plot(ax=ax2, color=[COL["Serious"], COL["Fatal"]], lw=2)
    axs[0].set_title("Accidents by hour (area) and Serious/Fatal share % (lines)")
    axs[0].set_xlabel("hour of day"); ax2.set_ylabel("% of accidents")
    d = pct(pd.crosstab(df.dow, df.sev)[SEVERITY_ORDER])
    d.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    d.plot.bar(stacked=True, ax=axs[1], color=[COL[k] for k in SEVERITY_ORDER], rot=0)
    axs[1].set_title("Severity share by weekday (%)"); axs[1].legend(ncol=3, frameon=False)
    save("02_hour_weekday")

    # 3 conditions
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, (col, ttl) in zip(axs.ravel(), [("speed_limit", "Speed limit (mph)"),
                                            ("Road_Type", "Road type"),
                                            ("Light_Conditions", "Light conditions"),
                                            ("Weather_Conditions", "Weather")]):
        t = pct(pd.crosstab(df[col].fillna("Unknown"), df.sev)[SEVERITY_ORDER])
        t = t.sort_values("Fatal")
        t[["Serious", "Fatal"]].plot.barh(ax=ax, stacked=True, color=[COL["Serious"], COL["Fatal"]])
        ax.set_title(f"Serious + Fatal share by {ttl.lower()} (%)"); ax.set_ylabel("")
        ax.legend(frameon=False)
    save("03_conditions")

    # 4 urban vs rural
    t = pct(pd.crosstab(df.Urban_or_Rural_Area, df.sev)[SEVERITY_ORDER])
    ax = t.plot.bar(stacked=True, color=[COL[k] for k in SEVERITY_ORDER], figsize=(5, 3.8), rot=0)
    ax.set_title("Urban vs rural severity share (%)"); ax.legend(frameon=False, ncol=3)
    save("04_urban_rural")

    # 5 yearly trend
    y = pd.crosstab(df.year, df.sev)[SEVERITY_ORDER]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    y.plot(ax=axs[0], color=[COL[k] for k in SEVERITY_ORDER], marker="o")
    axs[0].set_title("Accidents per year"); axs[0].legend(frameon=False)
    pct(y)[["Serious", "Fatal"]].plot(ax=axs[1], color=[COL["Serious"], COL["Fatal"]], marker="o")
    axs[1].set_title("Serious and Fatal share per year (%)  - note the 2016/17 jump")
    save("05_yearly_trend")

    # 6 missing values (raw categorical columns, before imputation)
    miss = pd.read_csv(RAW_CSV, nrows=300000, low_memory=False).replace(
        "Data missing or out of range", np.nan)
    m = (miss.isna().mean() * 100).sort_values(ascending=False)
    m = m[m > 0].head(14)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(m.index[::-1], m.values[::-1], color="#6366f1")
    ax.set_title("Missing / out-of-range values in the raw file (%, first 300k rows)")
    save("06_missing_values")
    m.round(2).to_csv(TAB_DIR / "missing_values_raw.csv", header=["missing_pct"])

    # 7 association heatmap (Cramer's V vs severity and between features)
    cols = ["sev", "speed_limit", "Road_Type", "Light_Conditions", "Weather_Conditions",
            "Road_Surface_Conditions", "Junction_Detail", "Urban_or_Rural_Area", "1st_Road_Class",
            "hour", "dow", "n_vehicles"]
    s = df.sample(150000, random_state=SEED)
    M = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            M.loc[a, b] = 1.0 if a == b else cramers_v(s[a].fillna("NA"), s[b].fillna("NA"))
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="mako_r", ax=ax, cbar_kws={"shrink": .7})
    ax.set_title("Cramer's V association (150k sample)")
    save("07_cramers_v")

    # 8 UK map
    s = df.sample(300000, random_state=SEED)
    fig, ax = plt.subplots(figsize=(5.5, 7.5))
    ax.hexbin(s.lon, s.lat, gridsize=170, bins="log", cmap="inferno", mincnt=1)
    ax.set_aspect(1.6); ax.set_title("Accident density across Great Britain")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    save("08_uk_density")

    # 9 vehicles / casualties (casualty column shown only to illustrate leakage)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    pct(pd.crosstab(df.n_vehicles, df.sev)[SEVERITY_ORDER])[["Serious", "Fatal"]].plot.bar(
        ax=axs[0], stacked=True, color=[COL["Serious"], COL["Fatal"]], rot=0)
    axs[0].set_title("Serious + Fatal share by number of vehicles (%)")
    c = df.Number_of_Casualties.clip(upper=6)
    pct(pd.crosstab(c, df.sev)[SEVERITY_ORDER])[["Serious", "Fatal"]].plot.bar(
        ax=axs[1], stacked=True, color=[COL["Serious"], COL["Fatal"]], rot=0)
    axs[1].set_title("Same by casualties  (post-accident, EXCLUDED as leakage)")
    save("09_vehicles_casualties_leakage")

    # leakage evidence table
    leak = pd.DataFrame({
        "feature": ["Number_of_Casualties", "Did_Police_Officer_Attend_Scene_of_Accident"],
        "Fatal_rate_when_value_high": [
            df[df.Number_of_Casualties >= 3].severity.eq(2).mean(),
            df[df.Did_Police_Officer_Attend_Scene_of_Accident == 1].severity.eq(2).mean()],
        "Fatal_rate_when_value_low": [
            df[df.Number_of_Casualties == 1].severity.eq(2).mean(),
            df[df.Did_Police_Officer_Attend_Scene_of_Accident == 2].severity.eq(2).mean()],
    })
    leak.to_csv(TAB_DIR / "leakage_evidence.csv", index=False)

    # summary numbers for the report
    summ = {
        "ksi_share_by_year": (pct(y)[["Serious", "Fatal"]].sum(1)).round(2).to_dict(),
        "night_ksi_share": float(df[df.is_night == 1].ksi.mean()),
        "day_ksi_share": float(df[df.is_night == 0].ksi.mean()),
        "rural_ksi_share": float(df[df.is_urban == 0].ksi.mean()),
        "urban_ksi_share": float(df[df.is_urban == 1].ksi.mean()),
    }
    pd.Series(summ).to_json(TAB_DIR / "eda_summary.json", indent=2)
    print(summ)


if __name__ == "__main__":
    main()
