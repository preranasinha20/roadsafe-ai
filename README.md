# Explainable ML for Road Accident Severity Prediction and Black-Spot Detection

AUCML (TE7483) ESE case study · UK STATS19 2005-2017 (2.05 M accidents) · AIML, Symbiosis Institute of Technology, Pune

**Live app:** https://manarsenic.github.io/road-accident-severity-blackspots/ (GitHub Pages, fully static; the model, calibration and exact TreeSHAP run in the browser)

## What it does
1. **Severity prediction** (Slight / Serious / Fatal) validated on years the model never saw.
2. **Black-spot detection** (150 m severity-weighted peaks), validated out of time.
3. **Explainability** with SHAP and LIME, including inside versus outside black spots.

## Headline results (test years 2016-17, strict time split)
| Metric | XGBoost, tuned scales | Majority baseline |
|---|---|---|
| Macro-F1 | **0.412** | 0.301 |
| Recall Fatal  | 0.174 / 0.373 | 0 |
| ROC-AUC (OvR) | 0.698 | 0.500 |
| Accuracy | 0.723 (argmax 0.821) | 0.821 |

Random-split accuracy 0.846 (majority 0.846); binary KSI AUC 0.658.

**On accuracy:** 82.1% of test accidents are Slight, so "always Slight" already scores 82.1%. Accuracy of 85-90%+ is reachable only on a random split of the unweighted model (which mostly predicts Slight) or with leaked columns (casualty counts). We therefore headline macro-F1, per-class recall, ROC-AUC and PR-AUC, and report accuracy under every protocol with its baseline. Only scene-level features were available (the vehicle file was not part of the supplied data), which caps performance.

## Repository
```
src/         data_prep, eda, pipeline, metrics, run_experiments, train_final, run_ablations,
             blackspots, explain, export_web, make_notebooks, build_report, build_slides
notebooks/   01-05 executed walkthroughs
reports/     report.pdf (full report), slides.pdf (10 slides), VIVA.md, RUBRIC_MAP.md, tables/, figures/
app/         React + Vite web app (deployed with GitHub Actions)
streamlit_app/  companion Streamlit app using the Python pipeline
```

## Reproduce
```bash
pip install -r requirements.txt
# download Accident_Information.csv from https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles into data/raw/
python -m src.data_prep && python -m src.eda
python -m src.run_experiments && python -m src.train_final && python -m src.run_ablations
python -m src.blackspots && python -m src.explain && python -m src.export_web
python -m src.make_notebooks && python -m src.build_report && python -m src.build_slides
cd app && npm ci && node scripts/parity.mjs && npm run dev      # web app
streamlit run streamlit_app/app.py                              # optional Streamlit app
```
Large artefacts (raw/processed data, model binaries) are git-ignored; exported web data is committed so the site builds anywhere. Fixed seed 42 throughout.

## Method in one paragraph
Clean, engineer scene/time features and expanding-window spatial history (only earlier years), split by time (train 2006-14, validate 2015, test 2016-17), compare six model families and five imbalance strategies, tune with Optuna, calibrate isotonically and tune class decision scales on 2015, evaluate once on the test years and refine later
