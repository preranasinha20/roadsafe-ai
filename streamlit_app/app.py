"""Streamlit companion app: loads the saved pipeline and never retrains.

Run from the project root:  streamlit run streamlit_app/app.py
The main, fully static site lives in app/ (React); this version uses the full Python pipeline.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import GRID_DEG, SEVERITY_ORDER  # noqa: E402
from src.modelio import load_model  # noqa: E402

DATA = ROOT / "app" / "public" / "data"
COLORS = {"Slight": "#34d399", "Serious": "#fbbf24", "Fatal": "#fb4b6e"}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(page_title="RoadSafe AI", page_icon="🚦", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem} h1{letter-spacing:-.02em}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return load_model()


@st.cache_data
def load_json(name):
    return json.loads((DATA / f"{name}.json").read_text())


def history_for(meta, lat, lon, la_name):
    la = next(x for x in meta["la"] if x["name"] == la_name)
    key = f"{int(np.floor(lat / GRID_DEG))}_{int(np.floor(lon / GRID_DEG))}"
    cell = meta["cells"].get(key, [0, 0, 0])
    prior = meta["prior_ksi"]
    hot = np.array(meta["hot"])
    d = np.hypot((hot[:, 0] - lat) * 111_000, (hot[:, 1] - lon) * 111_000 * np.cos(np.radians(lat))).min()
    return {"h_cell_n": cell[0], "h_cell_ksi_rate": (cell[1] + 20 * prior) / (cell[0] + 20), "h_cell_score": cell[2],
            "h_la_n": la["n"], "h_la_ksi_rate": la["ksi_rate"], "h_la_score": la["score"],
            "h_dist_hotspot_m": d, "h_in_hotspot": int(d < 300), "Police_Force": la["force"]}


def build_row(s, meta):
    night = s["Light_Conditions"].startswith("Darkness")
    urban = s["Urban_or_Rural_Area"] == "Urban"
    junction = s["Junction_Detail"] != "Not at junction or within 20 metres"
    sl = s["speed_limit"]
    row = dict(s)
    row.update({
        "season": ["winter", "winter", "spring", "spring", "spring", "summer", "summer", "summer", "autumn", "autumn", "autumn", "winter"][s["month"] - 1],
        "speed_band": "low" if sl <= 30 else "medium" if sl <= 50 else "high",
        "is_weekend": int(s["dow"] >= 5), "is_rush_hour": int(s["hour"] in (7, 8, 9, 16, 17, 18, 19)),
        "is_night": int(night), "hour_sin": np.sin(2 * np.pi * s["hour"] / 24), "hour_cos": np.cos(2 * np.pi * s["hour"] / 24),
        "is_urban": int(urban), "wet_surface": int(s["Road_Surface_Conditions"] in ("Wet or damp", "Flood over 3cm. deep")),
        "bad_weather": int(s["Weather_Conditions"] in ("Raining no high winds", "Raining + high winds", "Snowing no high winds", "Snowing + high winds", "Fog or mist")),
        "dark_and_unlit": int(s["Light_Conditions"] in ("Darkness - no lighting", "Darkness - lights unlit")),
        "junction_at_night": int(junction and night), "high_speed_rural": int(sl >= 60 and not urban),
        "2nd_Road_Class": "None", "Carriageway_Hazards": "None", "Special_Conditions_at_Site": "None",
        "Pedestrian_Crossing-Human_Control": "code_0", "Pedestrian_Crossing-Physical_Facilities": "code_0",
    })
    row.update(history_for(meta, s["lat"], s["lon"], s["la_district"]))
    return row


bundle, meta = load_model(), load_json("meta")
final = load_json("final")
st.title("🚦 RoadSafe AI")
st.caption("Explainable accident-severity prediction and black-spot detection · UK STATS19 2005-2017")

tab1, tab2, tab3, tab4 = st.tabs(["Severity predictor", "Black-spot map", "Model performance", "About"])

with tab1:
    c1, c2 = st.columns([1, 1])
    opts = meta["options"]
    with c1:
        la = st.selectbox("Local authority", [x["name"] for x in meta["la"]], index=[x["name"] for x in meta["la"]].index("Westminster") if any(x["name"] == "Westminster" for x in meta["la"]) else 0)
        loc = next(x for x in meta["la"] if x["name"] == la)
        lat = st.number_input("Latitude", value=float(loc["lat"]), format="%.4f")
        lon = st.number_input("Longitude", value=float(loc["lon"]), format="%.4f")
        speed = st.select_slider("Speed limit (mph)", [20, 30, 40, 50, 60, 70], value=30)
        hour = st.slider("Hour of day", 0, 23, 17)
        dow = st.selectbox("Day", DAYS, index=2)
        month = st.slider("Month", 1, 12, 6)
        vehicles = st.slider("Vehicles involved", 1, 6, 2)
    with c2:
        sel = {k: st.selectbox(k.replace("_", " "), v) for k, v in opts.items()}
        urban = st.radio("Area", ["Urban", "Rural"], horizontal=True)
    s = {**sel, "lat": lat, "lon": lon, "la_district": la, "speed_limit": speed, "hour": hour, "dow": DAYS.index(dow),
         "month": month, "n_vehicles": vehicles, "Urban_or_Rural_Area": urban, "Junction_Control": sel.get("Junction_Control", "Give way or uncontrolled")}
    row = pd.DataFrame([build_row(s, meta)])[bundle["features"]]
    raw = bundle["pipeline"].predict_proba(row)
    probs = bundle["calibrator"].transform(raw)[0]
    label = SEVERITY_ORDER[int((probs * bundle["scales"]).argmax())]
    st.markdown(f"### Predicted severity: <span style='color:{COLORS[label]}'>{label}</span>", unsafe_allow_html=True)
    st.metric("Risk of death or serious injury", f"{(probs[1] + probs[2]) * 100:.1f}%", f"test-period average {meta['test_ksi_rate'] * 100:.1f}%", delta_color="off")
    st.bar_chart(pd.Series(probs, index=SEVERITY_ORDER))

with tab2:
    spots = pd.DataFrame(json.loads((DATA / "blackspots.json").read_text()))
    n = st.slider("Top black spots", 10, 300, 100, 10)
    st.map(spots.head(n).rename(columns={"lon": "longitude", "lat": "latitude"})[["latitude", "longitude"]])
    st.dataframe(spots.head(n)[["rank", "la", "accidents", "fatal", "serious", "ksi_share", "score", "radius_m", "valid_per_year", "persistent"]], use_container_width=True)

with tab3:
    st.metric("Test macro-F1 (2016-17)", f"{final['test']['tuned']['macro_f1']:.3f}")
    st.dataframe(pd.DataFrame(final["protocols"]), use_container_width=True)
    st.info(final["accuracy_note"])
    st.dataframe(pd.DataFrame(final["errors"]), use_container_width=True)

with tab4:
    st.write("See the React site in `app/` and the report in `reports/` for the full methodology, SHAP/LIME analysis and limitations.")
