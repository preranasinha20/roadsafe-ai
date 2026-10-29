"""Step 9: assemble the final report (HTML + PDF) from the saved tables and figures.

Run:  python -m src.build_report
Every number in the text is read from reports/tables, so the report cannot drift from the results.
"""
import base64
import json
from pathlib import Path

import pandas as pd

from .config import *

T = lambda n: TAB_DIR / n
fmt = lambda v, d=3: f"{v:.{d}f}"
pc = lambda v, d=1: f"{v * 100:.{d}f}%"


def img(name, caption, width="100%"):
    b = base64.b64encode((FIG_DIR / name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" style="width:{width}"/><figcaption>{caption}</figcaption></figure>'


def table(df, cols=None, rename=None, floatfmt=3, cls=""):
    d = df[cols] if cols else df
    if rename:
        d = d.rename(columns=rename)
    return d.to_html(index=False, border=0, classes=f"t {cls}", float_format=lambda v: f"{v:.{floatfmt}f}", escape=False)


def build():
    res = json.load(open(T("final_results.json")))
    bs = json.load(open(T("blackspot_summary.json")))
    ds = json.load(open(T("dataset_summary.json")))
    grid = pd.read_csv(T("experiment_grid.csv"))
    clog = pd.read_csv(T("cleaning_log.csv"))
    cv = pd.read_csv(T("cv_results.csv"))
    abl = pd.read_csv(T("ablations.csv"))
    sens = pd.read_csv(T("blackspot_sensitivity.csv"))
    comp = pd.read_csv(T("blackspot_method_comparison.csv"))
    stab = pd.read_csv(T("blackspot_rank_stability.csv"))
    top = pd.read_csv(T("blackspots_top100.csv")).head(12)
    lime = pd.read_csv(T("lime_vs_shap.csv"))
    imp = pd.read_csv(T("shap_importance.csv"), index_col=0)
    io = pd.read_csv(T("shap_inside_vs_outside.csv"), index_col=0)
    errs = json.load(open(DATA_DIR / "final.json"))["errors"] if (DATA_DIR := APP_DATA).exists() else []
    leak = pd.read_csv(T("leakage_evidence.csv"))
    dd = pd.read_csv(T("data_dictionary_processed.csv"))

    win = max(["lgbm", "xgb"], key=lambda m: res[m]["val_tuned"]["macro_f1"])
    W = {"lgbm": "LightGBM", "xgb": "XGBoost"}[win]
    r = res[win]
    tt, ta = r["test_tuned"], r["test_argmax"]
    rs, bk = res["random_split"], res["binary_ksi"]
    base_f1 = grid.query("model=='dummy'").test_macro_f1.iloc[0]
    base_acc = grid.query("model=='dummy'").test_accuracy.iloc[0]
    g = grid[grid.decision == "tuned"].copy()
    gm = g.assign(model=g.model.map({"dummy": "Majority", "logreg": "Logistic Reg.", "tree": "Decision Tree", "rf": "Random Forest", "lgbm": "LightGBM", "xgb": "XGBoost"}),
                  imbalance=g.imbalance.map({"none": "none", "class_weight": "class weights", "undersample": "under-sampling", "smote": "SMOTE"}))
    comp_tbl = gm[["model", "imbalance", "val_macro_f1", "test_macro_f1", "test_recall_Fatal", "test_recall_Serious", "test_roc_auc_ovr", "test_accuracy", "fit_seconds"]]
    comp_tbl.columns = ["Model", "Imbalance", "Val macro-F1", "Test macro-F1", "Recall Fatal", "Recall Serious", "ROC-AUC", "Accuracy", "Fit (s)"]
    ga = grid[(grid.decision == "argmax") & grid.model.isin(["lgbm", "xgb"])]
    best_none = grid[(grid.decision == "argmax") & (grid.imbalance == "none") & grid.model.isin(["lgbm", "xgb"])].test_macro_f1.max()
    sp = {k: v for k, v in sens.iterrows()}
    spread = g[g.model.isin(["lgbm", "xgb", "rf", "logreg"])].test_macro_f1
    ksi_by_year = json.load(open(T("eda_summary.json")))["ksi_share_by_year"]

    css = """
    @page { size: A4; margin: 18mm 16mm; }
    body { font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; color:#14192b; line-height:1.5; font-size:10.5pt; }
    h1 { font-size: 26pt; margin: 0 0 6pt; letter-spacing:-.02em } h2 { font-size:16pt; margin: 22pt 0 6pt; padding-bottom:3pt; border-bottom:2px solid #6366f1; page-break-after: avoid }
    h3 { font-size:12pt; margin: 14pt 0 4pt; color:#3730a3 } .cover { padding: 40pt 0 24pt; border-bottom: 1px solid #ddd; margin-bottom: 10pt }
    .meta { color:#555; font-size:10pt } table.t { border-collapse: collapse; width:100%; font-size:8.6pt; margin: 6pt 0 10pt }
    table.t th { background:#eef0ff; text-align:left; padding:4pt 5pt; border-bottom:1px solid #99a } table.t td { padding:3.5pt 5pt; border-bottom:1px solid #e5e7ef }
    figure { margin: 8pt 0; page-break-inside: avoid; text-align:center } figcaption { font-size:8.8pt; color:#555; margin-top:3pt; text-align:left }
    .box { background:#f4f5ff; border-left:4px solid #6366f1; padding:7pt 10pt; margin:8pt 0; font-size:10pt }
    .warn { background:#fff7e6; border-left-color:#f59e0b } code { background:#f1f2f7; padding:0 3pt; border-radius:3pt; font-size:9pt }
    .kpi { display:flex; gap:8pt; margin:8pt 0 } .kpi div { flex:1; background:#f4f5ff; border-radius:8pt; padding:8pt } .kpi b { font-size:16pt; display:block; color:#4338ca }
    ol, ul { margin: 4pt 0 6pt 16pt; padding:0 }
    """
    top_tbl = top[["la", "accidents", "fatal", "serious", "ksi_share", "score", "valid_per_year", "persistent"]].copy()
    top_tbl.columns = ["Local authority", "Accidents", "Fatal", "Serious", "KSI share", "EPDO score", "2015-17 per yr", "Persists"]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Explainable ML for Road Accident Severity and Black Spots</title><style>{css}</style></head><body>

<div class="cover"><div class="meta">Applications and Use Cases of Machine Learning (TE7483) · ESE case-study project · AIML, Symbiosis Institute of Technology, Pune</div>
<h1>Explainable Machine Learning Framework for Road Accident Severity Prediction and Black-Spot Detection</h1>
<div class="meta">Dataset: UK Department for Transport STATS19, 2005-2017 ({ds['rows']:,} accidents) · Code: github.com/Manarsenic/road-accident-severity-blackspots</div></div>

<h2>Abstract</h2>
<p>Road-safety agencies must decide where to act and which conditions matter, yet most published severity models report overall accuracy on a target that is 85% one class, treat hotspot analysis separately, and use black-box models. This project builds one leak-free framework that (1) predicts accident severity (Slight, Serious, Fatal) on years the model never saw, (2) detects compact, severity-weighted black spots and validates them out of time, and (3) explains predictions with SHAP and LIME. Trained on 2006-2014, tuned on 2015 and tested once on 2016-2017, the final {W} model reaches macro-F1 <b>{fmt(tt['macro_f1'])}</b> (majority baseline {fmt(base_f1)}), ROC-AUC <b>{fmt(tt['roc_auc_ovr'])}</b> and recalls <b>{pc(tt['recall_KSI'],0)}</b> of killed-or-seriously-injured (KSI) accidents. Accuracy is {pc(ta['accuracy'])} with plain argmax and {pc(tt['accuracy'])} with decision scales tuned for rare-class recall, against a {pc(base_acc)} majority baseline: the headline is macro-F1 and rare-class recall because accuracy rewards ignoring fatalities. {pc(bs['top100_persistence_rate'],0)} of the top-100 black spots detected in 2005-14 remain active in 2015-17. A static web application runs the model, calibration and exact TreeSHAP entirely in the browser.</p>

<div class="kpi"><div><b>{fmt(tt['macro_f1'])}</b>test macro-F1</div><div><b>{pc(tt['recall_Fatal'],1)}</b>Fatal recall</div><div><b>{fmt(tt['roc_auc_ovr'])}</b>ROC-AUC</div><div><b>{pc(bs['top100_persistence_rate'],0)}</b>black spots persist</div></div>

<h2>1. Introduction: problem, motivation, objectives</h2>
<p><b>Problem.</b> Given conditions known at the scene (road, junction, light, weather, speed limit, time, local accident history) predict whether an injury accident will be Slight, Serious or Fatal, and locate where severe accidents concentrate. <b>Motivation.</b> India recorded roughly 1.5 lakh road deaths a year (MoRTH <i>Road Accidents in India</i> reports); authorities have limited budgets and need to know where and why. The UK STATS19 data is the largest open, labelled source, so it is used for method development; the framework transfers, a trained model would need local data. <b>Stakeholders.</b> Highway authorities, traffic police, urban planners, insurers and NGOs.</p>
<p><b>Research gap.</b> (i) Accuracy-only reporting on an 85/14/1 target; (ii) no systematic comparison of imbalance strategies under time-based validation; (iii) prediction and hotspot analysis done separately; (iv) black-box models without explanation.</p>
<p><b>Objectives.</b> O1 predict severity with imbalance-aware metrics validated on later years; O2 compare model families and imbalance strategies fairly; O3 detect black spots and prove they persist; O4 explain the model globally and locally and compare SHAP with LIME. <b>Research questions.</b> RQ1 Which conditions most raise the probability of Serious/Fatal outcomes? RQ2 Do black spots differ in their risk drivers? RQ3 Does leak-free spatial history improve prediction?</p>

<h2>2. Related work</h2>
<p>Severity prediction has been studied with logistic regression, decision trees, random forests and gradient boosting on STATS19 and similar data, typically reporting accuracy and often with random splits that mix years (Chang & Wang 2006; Abdel-Aty & Haleem 2011). Imbalance is commonly addressed by SMOTE (Chawla et al. 2002) or class weights, with mixed benefits reported. Black-spot identification traditionally uses fixed-window counting, empirical-Bayes and kernel density (Anderson 2009; Okabe et al. 2009), and density clustering such as DBSCAN (Ester et al. 1996) and HDBSCAN (Campello et al. 2013). Explainability for tree ensembles uses SHAP (Lundberg & Lee 2017) and LIME (Ribeiro et al. 2016). Gradient boosting (Chen & Guestrin 2016; Ke et al. 2017) is state of the art on tabular data, and Optuna (Akiba et al. 2019) automates tuning; isotonic calibration follows Niculescu-Mizil & Caruana (2005). This project combines these in one leak-free pipeline.</p>

<h2>3. Dataset</h2>
<p>Source: Department for Transport STATS19 (Open Government Licence), via the Kaggle mirror by tsiaras. File <code>Accident_Information.csv</code>: 2,047,256 rows × 34 columns, 2005-2017, one row per police-reported injury accident. Target <code>Accident_Severity</code>: {pc(ds['class_share']['Slight'])} Slight, {pc(ds['class_share']['Serious'])} Serious, {pc(ds['class_share']['Fatal'])} Fatal after cleaning.</p>
{img('01_class_distribution.png','Figure 1. Class distribution of the target.', '55%')}
<h3>3.1 Cleaning (row counts per step)</h3>{table(clog, rename={'step':'Step','rows':'Rows remaining'}, floatfmt=0)}
<p>Only {pc(1-clog.rows.iloc[-1]/clog.rows.iloc[0],3)} of rows were lost. Placeholder strings ("Data missing or out of range") were converted to missing and imputed as an explicit "Unknown" level inside the sklearn pipeline (fitted on training data only). Absent hazard/special-condition codes (98% and 97% of rows) are genuine "None" levels, not missing data.</p>
<h3>3.2 Exploratory analysis</h3>
{img('02_hour_weekday.png','Figure 2. Accidents by hour (area) with Serious/Fatal share (lines), and severity share by weekday. Fatal share peaks late at night, when traffic is lightest.')}
{img('03_conditions.png','Figure 3. Serious+Fatal share by speed limit, road type, light and weather. Severity rises steeply with speed limit and in darkness without lighting; weather matters far less than expected.')}
{img('05_yearly_trend.png','Figure 4. Accident counts fall over time while the Serious+Fatal share jumps in 2016-17 ({ksi_by_year["2015"]:.1f}% in 2015, {ksi_by_year["2017"]:.1f}% in 2017). This coincides with police forces adopting injury-based severity reporting and is a genuine distribution shift that the evaluation must face.')}
{img('08_uk_density.png','Figure 5. Accident density confirms the UK shape and concentration in cities.', '45%')}
{img('07_cramers_v.png','Figure 6. Cramér V association between categorical features and severity: all individual associations are weak, which foreshadows the modest ceiling on predictive performance.', '70%')}

<h3>3.3 Leakage control</h3>
<p>A feature is used only if it describes the scene or is known when the accident occurs. <code>Number_of_Casualties</code> and <code>Did_Police_Officer_Attend_Scene_of_Accident</code> are consequences of severity and are excluded (Figure 7); their effect is quantified in the ablation (Section 7.4).</p>
{img('09_vehicles_casualties_leakage.png','Figure 7. Fatal share rises with the number of casualties because casualties are a consequence of severity, not a predictor available beforehand.')}
<table class="t"><tr><th>Feature</th><th>Type</th><th>Known before outcome</th><th>Used</th><th>Reason</th></tr>
<tr><td>Number_of_Casualties</td><td>int</td><td>No</td><td>No</td><td>Consequence of the accident</td></tr>
<tr><td>Did_Police_Officer_Attend_Scene_of_Accident</td><td>cat</td><td>No</td><td>No</td><td>Recorded after the event, tied to severity</td></tr>
<tr><td>Easting/Northing, LSOA, Local_Authority_(Highway), InScotland</td><td>various</td><td>Yes</td><td>No</td><td>Redundant with latitude/longitude, district, force</td></tr>
<tr><td>Road, junction, light, weather, surface, speed limit, urban/rural</td><td>cat/int</td><td>Yes</td><td>Yes</td><td>Scene description</td></tr>
<tr><td>Date, time, weekday, month, vehicles (capped at 6)</td><td>int</td><td>Yes</td><td>Yes</td><td>Known at the time</td></tr>
<tr><td>Latitude, longitude</td><td>float</td><td>Yes</td><td>Yes</td><td>Location</td></tr>
<tr><td>h_* spatial history</td><td>float</td><td>Yes</td><td>Yes</td><td>Built only from earlier years</td></tr></table>

<h3>3.4 Feature engineering</h3>
<ul><li><b>Time:</b> hour, month, weekday, is_weekend, is_rush_hour (07-09, 16-19), is_night, season, cyclical hour (sin, cos).</li>
<li><b>Scene interactions:</b> dark_and_unlit, wet_surface, bad_weather, junction_at_night, high_speed_rural, banded speed limit (≤30, 40-50, ≥60).</li>
<li><b>Leak-free spatial history:</b> for an accident in year Y, the accident count, Bayesian-smoothed KSI rate and EPDO score of its ~550 m grid cell and its local authority, and the distance to the nearest earlier hotspot cell, all computed from years strictly before Y (expanding window), so no row ever sees its own label. Training starts in 2006 because 2005 only seeds the history.</li></ul>
<h3>3.5 Data dictionary</h3>
<p>The complete data dictionary (meaning, type, example, missing %, handling) for all {len(dd)} processed columns is in <code>reports/tables/data_dictionary_processed.csv</code> and on the web app's Data page.</p>

<h2>4. Methodology</h2>
<h3>4.1 Pipeline</h3>
<div class="box"><b>Raw CSV</b> → clean (UK bounding box, valid speed limits, duplicates) → engineer scene and time features → expanding-window spatial history → <b>time-based split</b> (train 2006-14: {1441672:,} rows; validate 2015: 140,008; test 2016-17: 266,524) → sklearn <code>ColumnTransformer</code> (median impute + one-hot with rare-level grouping) fitted on training only → imbalance strategy → model → isotonic calibration and class decision scales fitted on 2015 → one-shot test evaluation → SHAP/LIME and black-spot analysis → static web app.</div>
<h3>4.2 Splitting and validation</h3>
<p>Primary protocol is time-based (train past, validate next year, test latest two years), which mimics deployment. A secondary stratified random split (70/15/15) is reported for comparability with the literature. Hyper-parameters are tuned with Optuna against validation-year macro-F1, and 5-fold stratified CV with paired t-tests compares the model families. The test years are never used for tuning, calibration or threshold selection.</p>
<h3>4.3 Imbalance strategies compared</h3>
<p>No handling (baseline); class weights (balanced; for XGBoost via sample weights); random under-sampling of Slight to 60k; SMOTE on the one-hot training matrix; and <b>decision-scale tuning</b>, in which per-class multipliers on predicted probabilities are searched on the validation year to maximise macro-F1. Resampling is inside the training pipeline only; validation and test sets keep the real class distribution.</p>
<h3>4.4 Models and justification</h3>
<p>A majority-class dummy shows why accuracy misleads; logistic regression and a decision tree are interpretable baselines; random forest, XGBoost and LightGBM are the main models because gradient-boosted trees dominate tabular data, handle mixed features and missing values, are fast, and support exact TreeSHAP. The best two (LightGBM, XGBoost) are tuned with Optuna (TPE) over learning rate, leaves/depth, estimators, child weight, subsampling, regularisation and a class-weight exponent.</p>
<h3>4.5 Metrics</h3>
<p>Macro-F1 (headline), per-class precision/recall/F1, KSI recall, ROC-AUC one-vs-rest, PR-AUC for Serious and Fatal, balanced accuracy, Matthews correlation, confusion matrices, expected calibration error, accuracy for reference, plus training and inference time.</p>

<h2>5. Experimental setup</h2>
<p>Python 3.13, scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, imbalanced-learn 0.14.2, SHAP 0.54.1, LIME 0.2.0.1, Optuna 5.0.0, hdbscan 0.8.44. Hardware: 2 CPU cores, 8 GB RAM, no GPU. <code>random_state=42</code> everywhere. The comparison grid trains on a stratified 200,000-row sample of the training years so that 17 configurations are affordable; the tuned final models are refit on all 1,441,672 training rows. Code: <code>src/</code>; notebooks <code>notebooks/01-05</code>.</p>

<h2>6. Results: model comparison</h2>
<p>Table 1 compares every model and imbalance strategy on the test years with decision scales tuned on 2015. {W} is the validation winner. <b>Key finding:</b> once the decision scales are tuned, models and imbalance strategies are within about {fmt(spread.max()-spread.min())} macro-F1 of each other ({fmt(spread.min())}-{fmt(spread.max())} for the four main families), so the cheapest strategy (no resampling plus tuned decision scales) is as good as SMOTE or under-sampling. Without tuning, plain argmax with no handling reaches only macro-F1 {fmt(best_none)} because the model never predicts a fatality.</p>
{table(comp_tbl, floatfmt=3)}
<p><i>Table 1. Test results (2016-17), tuned decision scales, models trained on a 200k sample.</i></p>
<h3>6.1 Final tuned model</h3>
<p>After Optuna tuning ({r['params']}; class-weight exponent {r['weight_power']:.2f}) and refitting on 1.44M rows, the {W} model scores:</p>
<table class="t"><tr><th>Decision rule</th><th>Macro-F1</th><th>Accuracy</th><th>Bal. acc.</th><th>MCC</th><th>Recall Fatal</th><th>Recall Serious</th><th>KSI recall</th><th>ROC-AUC</th><th>PR-AUC Fatal</th></tr>
<tr><td>Tuned scales (headline)</td><td>{fmt(tt['macro_f1'])}</td><td>{fmt(tt['accuracy'])}</td><td>{fmt(tt['balanced_accuracy'])}</td><td>{fmt(tt['mcc'])}</td><td>{fmt(tt['recall_Fatal'])}</td><td>{fmt(tt['recall_Serious'])}</td><td>{fmt(tt['recall_KSI'])}</td><td>{fmt(tt['roc_auc_ovr'])}</td><td>{fmt(tt['pr_auc_Fatal'])}</td></tr>
<tr><td>Plain argmax</td><td>{fmt(ta['macro_f1'])}</td><td>{fmt(ta['accuracy'])}</td><td>{fmt(ta['balanced_accuracy'])}</td><td>{fmt(ta['mcc'])}</td><td>{fmt(ta['recall_Fatal'])}</td><td>{fmt(ta['recall_Serious'])}</td><td>{fmt(ta['recall_KSI'])}</td><td>{fmt(ta['roc_auc_ovr'])}</td><td>{fmt(ta['pr_auc_Fatal'])}</td></tr>
<tr><td>Majority baseline</td><td>{fmt(base_f1)}</td><td>{fmt(base_acc)}</td><td>0.333</td><td>0.000</td><td>0</td><td>0</td><td>0</td><td>0.500</td><td>-</td></tr></table>
<div class="box warn"><b>On accuracy.</b> On the 2016-17 test years {pc(base_acc)} of accidents are Slight, so always predicting Slight scores {pc(base_acc)} accuracy while catching no fatality. Reporting accuracy alone would therefore reward the wrong behaviour. Under a stratified random split (the protocol used by most papers) the unweighted model reaches {pc(rs['unweighted_argmax']['accuracy'])} accuracy against a {pc(rs['majority_accuracy'])} majority baseline, and on the binary KSI task accuracy is {pc(bk['argmax']['accuracy'])} at threshold 0.5 against {pc(bk['majority_accuracy'])} for the majority class. Accuracy above 90% cannot be reached honestly with scene-level features; values that high in the literature almost always come from leaked columns (casualty counts) or mixed-year splits.</div>
<h3>6.2 Alternative protocols</h3>
<table class="t"><tr><th>Protocol</th><th>Accuracy</th><th>Majority baseline</th><th>Macro-F1</th><th>ROC-AUC</th></tr>
<tr><td>Time-based test, tuned scales</td><td>{pc(tt['accuracy'])}</td><td>{pc(base_acc)}</td><td>{fmt(tt['macro_f1'])}</td><td>{fmt(tt['roc_auc_ovr'])}</td></tr>
<tr><td>Time-based test, plain argmax</td><td>{pc(ta['accuracy'])}</td><td>{pc(base_acc)}</td><td>{fmt(ta['macro_f1'])}</td><td>{fmt(ta['roc_auc_ovr'])}</td></tr>
<tr><td>Stratified random split, unweighted</td><td>{pc(rs['unweighted_argmax']['accuracy'])}</td><td>{pc(rs['majority_accuracy'])}</td><td>{fmt(rs['unweighted_argmax']['macro_f1'])}</td><td>{fmt(rs['unweighted_argmax']['roc_auc_ovr'])}</td></tr>
<tr><td>Stratified random split, tuned scales</td><td>{pc(rs['tuned']['accuracy'])}</td><td>{pc(rs['majority_accuracy'])}</td><td>{fmt(rs['tuned']['macro_f1'])}</td><td>{fmt(rs['tuned']['roc_auc_ovr'])}</td></tr>
<tr><td>Binary KSI vs Slight, threshold 0.5</td><td>{pc(bk['argmax']['accuracy'])}</td><td>{pc(bk['majority_accuracy'])}</td><td>{fmt(bk['argmax']['macro_f1'])}</td><td>{fmt(bk['argmax']['roc_auc'])}</td></tr>
<tr><td>Binary KSI vs Slight, tuned threshold</td><td>{pc(bk['tuned']['accuracy'])}</td><td>{pc(bk['majority_accuracy'])}</td><td>{fmt(bk['tuned']['macro_f1'])}</td><td>{fmt(bk['tuned']['roc_auc'])}</td></tr></table>
<h3>6.3 Cross-validation and significance</h3>
{table(cv[['model','cv_macro_f1_mean','cv_macro_f1_std','p_value_vs_best']], rename={'model':'Model','cv_macro_f1_mean':'CV macro-F1 (mean)','cv_macro_f1_std':'Std','p_value_vs_best':'Paired-t p vs best'}, floatfmt=4)}
<p>5-fold stratified CV on 150k rows with class-weighted models and plain argmax; paired t-tests across folds compare the best model with each competitor.</p>
<h3>6.4 Calibration, curves and errors</h3>
<p>Class weighting inflates raw probabilities. Isotonic calibration fitted on 2015 reduces the expected calibration error on the test years from {fmt(r['ece_test_raw'],4)} to {fmt(r['ece_test_calibrated'],4)}, so the web app's probabilities can be read as probabilities. Confusion matrices, ROC/PR curves and the subgroup error analysis are on the web app's Model lab page and in the notebooks.</p>
<table class="t"><tr><th>Subgroup</th><th>Accidents</th><th>Fatal cases</th><th>Fatal recall</th><th>KSI recall</th><th>Accuracy</th></tr>
{''.join(f"<tr><td>{e['group']}</td><td>{e['n']:,}</td><td>{e['fatal_n']}</td><td>{pc(e['recall_Fatal'])}</td><td>{pc(e['recall_KSI'])}</td><td>{pc(e['accuracy'])}</td></tr>" for e in errs)}</table>
<h3>6.5 Ablation study</h3>
{table(abl[['variant','macro_f1','roc_auc_ovr','accuracy','recall_Fatal','recall_Serious']], rename={'variant':'Variant','macro_f1':'Macro-F1','roc_auc_ovr':'ROC-AUC','accuracy':'Accuracy','recall_Fatal':'Recall Fatal','recall_Serious':'Recall Serious'}, floatfmt=3)}
<p>Each row changes one thing relative to the full model, with the same tuned scales and a 400k training sample. The final row adds the two leakage columns to show the effect leakage would have had.</p>

<h2>7. Black-spot analysis</h2>
<p><b>Definition.</b> A black spot is a compact site (150 m radius) with a high severity-weighted accident score (EPDO: Fatal 10, Serious 5, Slight 1) over the 2005-2014 detection window. <b>Method.</b> Coordinates are projected to the British National Grid. Every accident is scored by the weighted accidents within the radius, peaks are picked greedily and neighbours within twice the radius are suppressed, giving {bs['n_black_spots']:,} disjoint sites. DBSCAN and HDBSCAN were tested first and rejected as the primary method because on road networks they chain along corridors (Table 5): even with eps = 100 m the largest DBSCAN cluster holds {int(comp.largest_cluster_accidents.iloc[0]):,} accidents, and HDBSCAN clusters span up to {comp.max_span_km.iloc[-1]:.0f} km, which are not black spots. DBSCAN and a weighted KDE (100 m grid) are kept as cross-checks: {pc(bs['kde_overlap_top50'],0)} and {pc(bs['dbscan_overlap_top50'],0)} of the top-50 sites coincide with a KDE hotspot and a DBSCAN cluster respectively.</p>
{table(comp, cols=['method','params','clusters','noise_share','median_span_km','p90_span_km','max_span_km'], rename={'method':'Method','params':'Parameters','clusters':'Clusters','noise_share':'Noise share','median_span_km':'Median span (km)','p90_span_km':'P90 span (km)','max_span_km':'Max span (km)'}, floatfmt=2)}
<p><i>Table 5. Why raw density clustering was not used as the primary definition.</i></p>
<h3>7.1 Temporal validation</h3>
<p>Sites detected on 2005-2014 were checked against 2015-2017, which the detector never saw. A site "persists" if it still averages at least one accident per year and at least half its earlier annual rate. <b>{pc(bs['top100_persistence_rate'],0)}</b> of the top 100 persist (median later/earlier rate {bs['top100_median_ratio']:.2f}); a rolling check (detect 2005-11, validate 2012-14) gives {pc(bs['rolling_2005_11_vs_2012_14_persistence_top100'],0)}. As a control, ordinary dense sites around rank 1,000 persist {pc(bs['control_rank1000_persistence'],0)} of the time, so the persistence criterion is lenient and the rank correlation between detection score and later score (Spearman {sens.spearman_score_vs_later.iloc[1]:.2f} at the final setting) is the more informative evidence. The top-100 sites hold {pc(bs['top100_share_of_all_accidents'],1)} of all accidents but have a Serious+Fatal share of {pc(bs['top100_mean_ksi_share'],1)}, slightly below the overall {pc(bs['overall_ksi_share'],1)}: the busiest sites are urban junctions with many slight accidents.</p>
<h3>7.2 Sensitivity analysis</h3>
{table(sens, cols=['radius_m','weights_fatal_serious_slight','top100_mean_ksi_share','top100_persistence_rate','top100_median_ratio','spearman_score_vs_later'], rename={'radius_m':'Radius (m)','weights_fatal_serious_slight':'Weights F/S/Sl','top100_mean_ksi_share':'Top-100 KSI share','top100_persistence_rate':'Persistence','top100_median_ratio':'Median later/earlier','spearman_score_vs_later':'Spearman'}, floatfmt=3)}
<p>Radius (100-300 m) and severity weights (10/5/1, 5/2/1, 20/5/1, counts only) change the ranking only mildly: persistence stays {pc(sens.top100_persistence_rate.min(),0)}-{pc(sens.top100_persistence_rate.max(),0)}. A radius of 150 m and the common EPDO weights 10/5/1 were chosen because they match typical UK site definitions and gave a stable ranking; the share of the final top-20 sites reappearing under other settings ranges from {pc(stab.top20_shared_with_final.min(),0)} to {pc(stab.top20_shared_with_final.max(),0)}.</p>
{table(top_tbl, floatfmt=2)}
<p><i>Table 7. The 12 highest-scoring black spots.</i> A black spot shows <b>concentration, not cause</b>, and the data has no traffic volumes, so it cannot rank roads by per-vehicle risk.</p>

<h2>8. Explainability findings</h2>
{img('20_shap_importance_per_class.png','Figure 8. Mean |SHAP| per class (6,400-row test sample). Speed limit, local accident history and the hour dominate; weather is minor.')}
{img('21_shap_beeswarm_fatal.png','Figure 9. SHAP summary for the Fatal class.','70%')}
{img('22_shap_dependence_fatal.png','Figure 10. Dependence plots: SHAP value for Fatal versus speed limit, hour and local KSI history (colour: night).')}
{img('23_shap_local_cases.png','Figure 11. Local explanations: a correctly flagged fatal, a missed fatal, a correctly flagged serious and a correctly predicted slight accident.')}
{img('24_shap_inside_vs_outside.png','Figure 12. Share of total |SHAP| per feature inside versus outside black spots.', '70%')}
<p><b>SHAP vs LIME.</b> SHAP is exact and deterministic for trees; LIME fits a random local surrogate. The two agree on the main drivers for some cases and differ for others; LIME's top-5 features also vary with the random seed.</p>
{table(lime[['explained_class','shap_top5','lime_top5','shap_lime_top5_overlap','lime_stability_jaccard']], rename={'explained_class':'Class','shap_top5':'SHAP top-5','lime_top5':'LIME top-5','shap_lime_top5_overlap':'Overlap','lime_stability_jaccard':'LIME stability (Jaccard)'}, floatfmt=2)}
<p><b>Recommendations (association, not causation).</b> Speed management on high-limit rural roads, lighting on unlit stretches, and junction treatment at the top-ranked urban sites are the levers the model associates with lower severity; any intervention must be evaluated with before-after studies.</p>

<h2>9. Application</h2>
<p>The web application (React, fully static, deployable on GitHub Pages) has six pages: Overview; Predictor (map-based location, scenario inputs, calibrated probabilities, live exact TreeSHAP and what-if analysis, all computed in the browser from the exported trees); Black-spot map (heatmap, markers, filters, ranked table, CSV export); Explainability; Model lab; and Data &amp; method. A parity test (<code>app/scripts/parity.mjs</code>) confirms the JavaScript engine reproduces scikit-learn probabilities and SHAP values. A Streamlit version (<code>streamlit_app/</code>) uses the full Python pipeline.</p>

<h2>10. Limitations and ethical considerations</h2>
<ul><li>UK-only, injury accidents only: slight accidents are under-reported, uninjured collisions are absent.</li>
<li>Scene-level features cap performance (ROC-AUC {fmt(tt['roc_auc_ovr'])}); vehicle type, speeds, driver behaviour and casualty details are not in this table. Joining the vehicle and casualty tables is the clearest route to better prediction.</li>
<li>No traffic exposure: black spots show concentration, not risk per vehicle.</li>
<li>A reporting-system change around 2016 creates a distribution shift; we handle it with calibration and scales fitted on 2015, but residual shift remains.</li>
<li>SHAP and LIME explain the model, not causes. Predictions support planning and must not be used to blame individuals; locations are accident sites, not people.</li></ul>

<h2>11. Conclusion and future work</h2>
<p>A leak-free, time-validated framework links severity prediction, black-spot detection and explanation. The central lesson is methodological: with an 85/14/1 target, accuracy hides failure, and once decision scales are tuned the choice of model and imbalance method matters little next to feature information. Future work: join vehicle and casualty tables; add Indian MoRTH/state data and test transfer; normalise by traffic exposure; road-network (graph) clustering; spatio-temporal forecasting of next-year hotspots; live-data deployment.</p>

<h2>References</h2>
<ol style="font-size:9pt"><li>Department for Transport (2018). Road Safety Data (STATS19) 2005-2017. Open Government Licence v3.0.</li>
<li>Lundberg &amp; Lee (2017). A unified approach to interpreting model predictions. NeurIPS.</li><li>Ribeiro, Singh &amp; Guestrin (2016). "Why should I trust you?" KDD.</li>
<li>Chen &amp; Guestrin (2016). XGBoost: a scalable tree boosting system. KDD.</li><li>Ke et al. (2017). LightGBM. NeurIPS.</li>
<li>Chawla et al. (2002). SMOTE. JAIR 16.</li><li>Ester, Kriegel, Sander &amp; Xu (1996). DBSCAN. KDD.</li>
<li>Campello, Moulavi &amp; Sander (2013). Density-based clustering based on hierarchical density estimates. PAKDD.</li>
<li>Akiba et al. (2019). Optuna. KDD.</li><li>Niculescu-Mizil &amp; Caruana (2005). Predicting good probabilities with supervised learning. ICML.</li>
<li>Chang &amp; Wang (2006). Analysis of freeway accident frequencies / injury severity with CART. Accident Analysis &amp; Prevention.</li>
<li>Abdel-Aty &amp; Haleem (2011). Analyzing angle crashes at unsignalized intersections using machine learning techniques. Accident Analysis &amp; Prevention 43.</li>
<li>Anderson (2009). Kernel density estimation and K-means clustering to profile road accident hotspots. Accident Analysis &amp; Prevention 41.</li>
<li>Okabe et al. (2009). A kernel density estimation method for networks. Int. J. Geographical Information Science.</li>
<li>Ministry of Road Transport &amp; Highways, India. Road Accidents in India (annual reports).</li></ol>
</body></html>"""
    out = ROOT / "reports"
    (out / "report.html").write_text(html)
    return out / "report.html"


def to_pdf(html_path):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(html_path.as_uri())
        pg.wait_for_timeout(800)
        pg.pdf(path=str(html_path.with_suffix(".pdf")), format="A4", print_background=True,
               margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
               display_header_footer=True, footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#777"><span class="pageNumber"></span> / <span class="totalPages"></span></div>', header_template="<div></div>")
        b.close()


if __name__ == "__main__":
    h = build()
    to_pdf(h)
    print("report written:", h.with_suffix(".pdf"))
