"""Builds the 10-slide presentation (PDF) from the saved results.  Run: python -m src.build_slides"""
import base64
import json

from .config import *

res = json.load(open(TAB_DIR / "final_results.json"))
bs = json.load(open(TAB_DIR / "blackspot_summary.json"))
w = res["xgb"] if res["xgb"]["val_tuned"]["macro_f1"] >= res["lgbm"]["val_tuned"]["macro_f1"] else res["lgbm"]
t, a, rs, bk = w["test_tuned"], w["test_argmax"], res["random_split"], res["binary_ksi"]


def im(n, h=330):
    return f'<img style="height:{h}px;border-radius:10px" src="data:image/png;base64,{base64.b64encode((FIG_DIR / n).read_bytes()).decode()}"/>'


S = [
    ("Explainable ML for Road Accident Severity &amp; Black Spots",
     "<p class=big>Predict severity. Locate black spots. Explain every decision.</p><p>UK STATS19 · 2.05 M accidents · 2005-2017<br>AUCML (TE7483) · AIML · Symbiosis Institute of Technology, Pune</p>"),
    ("Problem and motivation",
     "<ul><li>Injury accidents: 85% Slight, 14% Serious, 1.3% Fatal</li><li>Authorities need <b>where</b> to act and <b>why</b>, with a limited budget</li><li>Gap: accuracy-only reporting, random splits, no link between prediction, hotspots and explanation</li><li>RQ1 drivers of severity · RQ2 black-spot drivers · RQ3 value of leak-free spatial history</li></ul>"),
    ("Data and leakage control",
     f"<div class=two><ul><li>2,046,815 clean rows (0.02% dropped, every step logged)</li><li>Casualty count and police attendance <b>excluded</b>: consequences of severity</li><li>Features: scene, time, location and <b>expanding-window spatial history</b> built from earlier years only</li></ul>{im('05_yearly_trend.png')}</div>"),
    ("Pipeline",
     "<p class=flow>Clean → Engineer → Leak-free history → <b>Time split</b> (train 2006-14 · val 2015 · test 2016-17) → Pipeline (impute + one-hot) → Imbalance strategy → Model → Calibrate + decision scales (on 2015) → Test once → SHAP / LIME / black spots → Web app</p><p>Everything fitted on training years only; resampling never touches validation or test.</p>"),
    ("Models and imbalance study",
     "<ul><li>Majority, Logistic Regression, Decision Tree, Random Forest, LightGBM, XGBoost</li><li>None · class weights · under-sampling · SMOTE · <b>tuned decision scales</b></li><li>Optuna tuning on validation macro-F1; isotonic calibration; 5-fold CV + paired t-tests</li><li><b>Finding:</b> after scale tuning all strategies land within ~0.01 macro-F1, so the cheap one wins</li></ul>"),
    ("Results on unseen 2016-17",
     f"<div class=kpis><div><b>{t['macro_f1']:.3f}</b>macro-F1<br>(baseline 0.301)</div><div><b>{t['recall_KSI']*100:.0f}%</b>KSI recall</div><div><b>{t['roc_auc_ovr']:.3f}</b>ROC-AUC</div><div><b>{w['ece_test_calibrated']:.3f}</b>ECE<br>(raw {w['ece_test_raw']:.3f})</div></div><p>Accuracy {t['accuracy']*100:.1f}% (tuned) · {a['accuracy']*100:.1f}% (argmax) vs 82.1% majority. Random split: {rs['unweighted_argmax']['accuracy']*100:.1f}% vs {rs['majority_accuracy']*100:.1f}%. Binary KSI AUC {bk['argmax']['roc_auc']:.3f}.</p><p class=note>Accuracy above 90% would require leaked features; we report honest, imbalance-aware metrics.</p>"),
    ("Ablations and honesty",
     "<ul><li>Removing spatial history, time features or coordinates each lowers macro-F1 and AUC</li><li>Scene-only features cap performance: vehicle and casualty tables would be the next gain</li><li>Distribution shift in 2016-17 (Serious+Fatal 15.5% → 18.6%) handled by calibration on 2015</li><li>Leakage columns shown in the ablation to quantify what they would add</li></ul>"),
    ("Black-spot detection",
     f"<div class=two><ul><li>150 m severity-weighted (EPDO 10/5/1) peak detection; DBSCAN/HDBSCAN rejected as primary (chaining)</li><li>{bs['n_black_spots']:,} sites from 2005-14; <b>{bs['top100_persistence_rate']*100:.0f}%</b> of top-100 persist in 2015-17</li><li>Control sites persist 85%, so we also report rank correlation; sensitivity over radius and weights</li></ul>{im('08_uk_density.png')}</div>"),
    ("Explainability",
     f"<div class=two><ul><li>SHAP: global, per-class, dependence, local waterfalls, inside vs outside black spots</li><li>LIME compared with SHAP, with seed-stability measured</li><li>Speed limit, local history and hour dominate; weather is minor</li></ul>{im('20_shap_importance_per_class.png')}</div>"),
    ("Application, limits and future work",
     "<ul><li>Static React app on GitHub Pages: map predictor, in-browser XGBoost + calibration + exact TreeSHAP, black-spot map, model lab (parity-tested vs Python)</li><li>Limits: UK-only, no traffic exposure, injury-only, scene-level features</li><li>Future: join vehicle/casualty tables, Indian MoRTH data, exposure-normalised risk, spatio-temporal forecasting</li></ul><p class=big>Thank you · Questions?</p>"),
]
css = """@page{size:1280px 720px;margin:0}body{margin:0;font-family:Inter,Helvetica,Arial,sans-serif}
.s{width:1280px;height:720px;box-sizing:border-box;padding:56px 72px;background:linear-gradient(135deg,#0b1020,#1b1f4a);color:#eef;page-break-after:always;position:relative}
h1{font-size:42px;margin:0 0 26px;background:linear-gradient(90deg,#38bdf8,#a78bfa);-webkit-background-clip:text;color:transparent}
li{font-size:25px;line-height:1.4;margin:10px 0}p{font-size:23px;line-height:1.45}.big{font-size:32px;font-weight:600}.flow{font-size:28px;line-height:1.7}
.two{display:flex;gap:30px;align-items:center}.kpis{display:flex;gap:20px;margin-bottom:12px}.kpis div{flex:1;background:#ffffff14;border-radius:14px;padding:18px;font-size:19px}
.kpis b{display:block;font-size:46px;color:#38bdf8}.note{color:#fbbf24;font-size:21px}.n{position:absolute;right:40px;bottom:24px;opacity:.5}"""
html = "<html><head><meta charset=utf-8><style>" + css + "</style></head><body>" + "".join(
    f"<div class=s><h1>{h}</h1>{b}<span class=n>{i+1}</span></div>" for i, (h, b) in enumerate(S)) + "</body></html>"
p = ROOT / "reports" / "slides.html"
p.write_text(html)
from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(); pg.goto(p.as_uri()); pg.wait_for_timeout(500)
    pg.pdf(path=str(p.with_suffix(".pdf")), width="1280px", height="720px", print_background=True); b.close()
print("slides written")
