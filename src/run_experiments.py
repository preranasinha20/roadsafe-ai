"""Step 3: model x imbalance-method comparison (3-class severity).

Run:  python -m src.run_experiments
All models share the same features and the same time-based split. Experiments use a stratified
200k-row sample of the training years so the whole grid is affordable on a laptop; the final
model (src/train_final.py) is refit on the complete training period.
Selection is done on the 2015 validation year. The 2016-17 test years are only scored.
"""
import time
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline

from .config import *
from .metrics import evaluate, tune_class_scales
from .pipeline import (feature_set, load_splits, make_classifier, make_preprocessor,
                       sample_weights)

warnings.filterwarnings("ignore")
N_SAMPLE = 200_000

# (model, imbalance method)
GRID = [
    ("dummy", "none"),
    ("logreg", "none"), ("logreg", "class_weight"), ("logreg", "smote"),
    ("tree", "none"), ("tree", "class_weight"),
    ("rf", "none"), ("rf", "class_weight"), ("rf", "undersample"),
    ("lgbm", "none"), ("lgbm", "class_weight"), ("lgbm", "undersample"), ("lgbm", "smote"),
    ("xgb", "none"), ("xgb", "class_weight"), ("xgb", "undersample"), ("xgb", "smote"),
]


def build(model, method, cats, nums):
    prep = make_preprocessor(cats, nums, scale=(model == "logreg"))
    cw = "balanced" if method == "class_weight" and model in ("logreg", "tree", "rf", "lgbm") else None
    clf = make_classifier(model, class_weight=cw)
    steps = [("prep", prep)]
    if method == "undersample":
        steps.append(("under", RandomUnderSampler(sampling_strategy={0: 60_000}, random_state=SEED)))
    if method == "smote":
        steps.append(("smote", SMOTE(random_state=SEED, k_neighbors=5)))
    steps.append(("clf", clf))
    return (ImbPipeline if len(steps) > 2 else Pipeline)(steps)


def main():
    tr, va, te = load_splits(sample=N_SAMPLE)
    cats, nums = feature_set("full")
    cols = cats + nums
    Xtr, ytr = tr[cols], tr.severity.values
    Xva, yva = va[cols], va.severity.values
    Xte, yte = te[cols], te.severity.values
    print(f"train {len(tr):,}  val {len(va):,}  test {len(te):,}", flush=True)

    rows = []
    for model, method in GRID:
        t0 = time.time()
        pipe = build(model, method, cats, nums)
        fit_kw = {}
        if model == "xgb" and method == "class_weight":
            fit_kw["clf__sample_weight"] = sample_weights(ytr)
        pipe.fit(Xtr, ytr, **fit_kw)
        fit_s = time.time() - t0
        t1 = time.time()
        p_va = pipe.predict_proba(Xva)
        p_te = pipe.predict_proba(Xte)
        inf_ms = (time.time() - t1) / (len(Xva) + len(Xte)) * 1000
        w = tune_class_scales(p_va, yva)          # tuned on validation only
        for tag, wts in [("argmax", None), ("tuned", w)]:
            mv = evaluate(yva, p_va, wts)
            mt = evaluate(yte, p_te, wts)
            rows.append({"model": model, "imbalance": method, "decision": tag,
                         "fit_seconds": round(fit_s, 1), "infer_ms_per_row": round(inf_ms, 4),
                         **{f"val_{k}": v for k, v in mv.items()},
                         **{f"test_{k}": v for k, v in mt.items()}})
        print(f"{model:7s}{method:13s} fit {fit_s:5.0f}s | val macroF1 argmax "
              f"{rows[-2]['val_macro_f1']:.3f} tuned {rows[-1]['val_macro_f1']:.3f} | "
              f"test macroF1 tuned {rows[-1]['test_macro_f1']:.3f} acc {rows[-1]['test_accuracy']:.3f}",
              flush=True)
        pd.DataFrame(rows).to_csv(TAB_DIR / "experiment_grid.csv", index=False)


if __name__ == "__main__":
    main()
