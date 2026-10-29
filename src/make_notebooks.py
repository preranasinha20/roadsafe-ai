"""Builds the five walkthrough notebooks from the saved artefacts and executes them.

The heavy lifting lives in src/*.py (so it is testable and reproducible from the command line);
the notebooks narrate each stage and display its saved tables and figures.
Run:  python -m src.make_notebooks
"""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

from .config import ROOT

NB_DIR = ROOT / "notebooks"
HEAD = "import sys, json; sys.path.insert(0, '..')\nimport pandas as pd\nfrom IPython.display import Image, display, Markdown\npd.set_option('display.max_columns', 40, 'display.width', 160)\nT = '../reports/tables/'; F = '../reports/figures/'\n"


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
    return n


NOTEBOOKS = {
    "01_eda": [
        ("md", "# 01 · Data understanding and exploratory analysis\nUK STATS19 accidents 2005-2017. Produced by `python -m src.data_prep` and `python -m src.eda`."),
        ("code", HEAD + "print(json.load(open(T+'dataset_summary.json')))\npd.read_csv(T+'cleaning_log.csv')"),
        ("md", "Only a handful of rows are lost: every step logs its row count."),
        ("code", "pd.read_csv(T+'missing_values_raw.csv', index_col=0)"),
        ("code", "display(Image(F+'01_class_distribution.png')); display(Image(F+'05_yearly_trend.png'))"),
        ("md", "**Reading the plots.** 85% of accidents are Slight, and the Serious+Fatal share jumps in 2016-17, a distribution shift we handle with calibration and decision scales fitted on 2015."),
        ("code", "display(Image(F+'02_hour_weekday.png')); display(Image(F+'03_conditions.png'))"),
        ("code", "display(Image(F+'04_urban_rural.png')); display(Image(F+'07_cramers_v.png')); display(Image(F+'08_uk_density.png'))"),
        ("md", "## Leakage evidence\nCasualty counts and police attendance are consequences of severity, so they are excluded."),
        ("code", "display(Image(F+'09_vehicles_casualties_leakage.png')); pd.read_csv(T+'leakage_evidence.csv')"),
    ],
    "02_preprocessing": [
        ("md", "# 02 · Preprocessing and feature engineering\nImplemented in `src/data_prep.py` and `src/pipeline.py`."),
        ("code", HEAD + "pd.read_csv(T+'data_dictionary_processed.csv')"),
        ("md", "## Leak-free spatial history\nFor an accident in year *Y*, every `h_*` feature uses only accidents from years before *Y* (expanding window). Training rows never see their own label."),
        ("code", "df = pd.read_parquet('../data/processed/accidents_clean.parquet')\ndf[[c for c in df.columns if c.startswith('h_')] + ['year', 'severity']].groupby('year').mean().round(3)"),
        ("code", "from src.pipeline import feature_set, make_preprocessor\ncats, nums = feature_set('full')\nprint(len(cats), 'categorical +', len(nums), 'numeric features')\nmake_preprocessor(cats, nums)"),
    ],
    "03_modelling": [
        ("md", "# 03 · Modelling: model families × imbalance strategies\nRun with `python -m src.run_experiments`, `python -m src.train_final` and `python -m src.run_ablations`."),
        ("code", HEAD + "g = pd.read_csv(T+'experiment_grid.csv')\ng[g.decision=='tuned'][['model','imbalance','val_macro_f1','test_macro_f1','test_recall_Fatal','test_recall_Serious','test_roc_auc_ovr','test_accuracy','fit_seconds']].sort_values('test_macro_f1', ascending=False).round(3)"),
        ("md", "Plain argmax versus tuned decision scales (fitted on 2015 only):"),
        ("code", "g[g.decision=='argmax'][['model','imbalance','val_macro_f1','test_macro_f1','test_recall_Fatal','test_accuracy']].round(3)"),
        ("code", "r = json.load(open(T+'final_results.json')); {k: r[k]['test_tuned'] for k in ('lgbm','xgb')}"),
        ("md", "## Cross-validation and significance"),
        ("code", "pd.read_csv(T+'cv_results.csv').round(4)"),
        ("md", "## Ablations"),
        ("code", "pd.read_csv(T+'ablations.csv').round(4)"),
        ("md", "## Other protocols\nStratified random split and the binary Killed/Seriously-injured task."),
        ("code", "{k: {m: round(v, 4) for m, v in r[k]['unweighted_argmax' if k=='random_split' else 'tuned'].items() if m in ('accuracy','macro_f1','roc_auc_ovr','roc_auc','recall_KSI')} for k in ('random_split','binary_ksi')}"),
    ],
    "04_blackspots": [
        ("md", "# 04 · Black-spot detection and validation\nImplemented in `src/blackspots.py`."),
        ("code", HEAD + "print(json.dumps(json.load(open(T+'blackspot_summary.json')), indent=2))"),
        ("md", "## Sensitivity to the clustering parameters"),
        ("code", "pd.read_csv(T+'blackspot_sensitivity.csv').round(3)"),
        ("md", "## Top 20 black spots"),
        ("code", "pd.read_csv(T+'blackspots_top100.csv').head(20)[['cluster_id','la','accidents','fatal','serious','ksi_share','score','radius_m','valid_per_year','persistent','road_type','speed_limit','junction']].round(2)"),
    ],
    "05_explainability": [
        ("md", "# 05 · Explainability: SHAP and LIME\nImplemented in `src/explain.py`."),
        ("code", HEAD + "pd.read_csv(T+'shap_importance.csv', index_col=0).round(4).head(15)"),
        ("code", "display(Image(F+'20_shap_importance_per_class.png')); display(Image(F+'21_shap_beeswarm_fatal.png'))"),
        ("code", "display(Image(F+'22_shap_dependence_fatal.png')); display(Image(F+'23_shap_local_cases.png'))"),
        ("md", "## Inside versus outside black spots"),
        ("code", "display(Image(F+'24_shap_inside_vs_outside.png')); pd.read_csv(T+'shap_inside_vs_outside.csv', index_col=0).round(4).head(10)"),
        ("md", "## LIME versus SHAP\nLIME fits a random local surrogate, so we report how stable its top features are across seeds."),
        ("code", "pd.read_csv(T+'lime_vs_shap.csv')"),
    ],
}


def main():
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        n = nb(cells)
        ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(n, {"metadata": {"path": str(NB_DIR)}})
        nbf.write(n, NB_DIR / f"{name}.ipynb")
        print("executed", name)


if __name__ == "__main__":
    main()
