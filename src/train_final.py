"""Step 5: tune, refit and evaluate the final models.

Run:  python -m src.train_final
 1. Optuna tuning on a stratified 300k sample, objective = validation-year macro-F1 after
    class-scale tuning (so the objective matches the headline metric).
 2. Refit on the complete training period (2006-2014).
 3. Isotonic calibration + decision scales fitted on 2015 only.
 4. One-shot evaluation on the 2016-17 test years.
 5. Secondary protocols: stratified random split, and the binary Killed-or-Seriously-Injured task.
"""
import json
import time
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import *
from .metrics import confusion, evaluate, expected_calibration_error, tune_class_scales
from .modelio import Calibrator
from .pipeline import feature_set, load_splits, make_classifier, make_preprocessor, sample_weights

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def fit_pipe(model, params, weight_power, cats, nums, X, y, n_classes=3):
    pipe = Pipeline([("prep", make_preprocessor(cats, nums)),
                     ("clf", make_classifier(model, n_classes=n_classes, **params))])
    w = None
    if weight_power > 0:
        counts = np.bincount(y)
        w = ((len(y) / (len(counts) * counts)) ** weight_power)[y]
    pipe.fit(X, y, clf__sample_weight=w)
    return pipe


def tune(model, cats, nums, tr, va, n_trials, seed=SEED):
    cols = cats + nums
    Xtr, ytr = tr[cols], tr.severity.values
    Xva, yva = va[cols], va.severity.values
    prep = make_preprocessor(cats, nums).fit(Xtr)       # fit once, reused for every trial
    Atr, Ava = prep.transform(Xtr), prep.transform(Xva)

    def objective(trial):
        wp = trial.suggest_float("weight_power", 0.0, 1.0)
        if model == "lgbm":
            p = dict(num_leaves=trial.suggest_int("num_leaves", 15, 63),
                     learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
                     n_estimators=trial.suggest_int("n_estimators", 120, 450),
                     min_child_samples=trial.suggest_int("min_child_samples", 40, 300),
                     subsample=trial.suggest_float("subsample", 0.6, 1.0),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                     reg_lambda=trial.suggest_float("reg_lambda", 0.5, 20, log=True))
        else:
            p = dict(max_depth=trial.suggest_int("max_depth", 4, 9),
                     learning_rate=trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
                     n_estimators=trial.suggest_int("n_estimators", 120, 400),
                     min_child_weight=trial.suggest_int("min_child_weight", 2, 40),
                     subsample=trial.suggest_float("subsample", 0.6, 1.0),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                     reg_lambda=trial.suggest_float("reg_lambda", 0.5, 20, log=True))
        clf = make_classifier(model, **p)
        counts = np.bincount(ytr)
        w = ((len(ytr) / (3 * counts)) ** wp)[ytr]
        clf.fit(Atr, ytr, sample_weight=w)
        proba = clf.predict_proba(Ava)
        scales = tune_class_scales(proba, yva)
        return evaluate(yva, proba, scales)["macro_f1"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params)
    wp = best.pop("weight_power")
    return best, wp, study


def load_best(path):
    t = pd.read_csv(path)
    row = t.loc[t.value.idxmax()]
    ints = {"num_leaves", "n_estimators", "min_child_samples", "max_depth", "min_child_weight"}
    p = {c[7:]: (int(row[c]) if c[7:] in ints else float(row[c])) for c in t.columns
         if c.startswith("params_") and pd.notna(row[c])}
    wp = p.pop("weight_power")
    return p, wp


def main():
    t_all = time.time()
    cats, nums = feature_set("full")
    cols = cats + nums
    tr_small, va, te = load_splits(sample=300_000)
    tr_full, _, _ = load_splits(sample=None)
    print(f"tune sample {len(tr_small):,} | full train {len(tr_full):,} | val {len(va):,} | test {len(te):,}")

    results, best_models = {}, {}
    for model, trials in [("lgbm", 24), ("xgb", 14)]:
        t0 = time.time()
        saved = TAB_DIR / f"optuna_trials_{model}.csv"
        if saved.exists():                      # resume: reuse the finished study instead of retuning
            best, wp = load_best(saved)
            print(f"[{model}] reusing saved Optuna study: {best} weight_power {wp:.2f}", flush=True)
        else:
            best, wp, study = tune(model, cats, nums, tr_small, va, trials)
            print(f"[{model}] best val macro-F1 {study.best_value:.4f}  params {best} weight_power {wp:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            study.trials_dataframe().to_csv(saved, index=False)
        t0 = time.time()
        pipe = fit_pipe(model, best, wp, cats, nums, tr_full[cols], tr_full.severity.values)
        fit_s = time.time() - t0
        p_va = pipe.predict_proba(va[cols])
        cal = Calibrator().fit(p_va, va.severity.values)
        c_va = cal.transform(p_va)
        scales = tune_class_scales(c_va, va.severity.values)
        t1 = time.time()
        p_te = cal.transform(pipe.predict_proba(te[cols]))
        inf_ms = (time.time() - t1) / len(te) * 1000
        yte = te.severity.values
        res = {"params": best, "weight_power": wp, "scales": scales.tolist(),
               "fit_seconds": fit_s, "infer_ms_per_row": inf_ms,
               "val_tuned": evaluate(va.severity.values, c_va, scales),
               "test_argmax": evaluate(yte, p_te),
               "test_tuned": evaluate(yte, p_te, scales),
               "ece_test_raw": expected_calibration_error(pipe.predict_proba(te[cols]), yte),
               "ece_test_calibrated": expected_calibration_error(p_te, yte)}
        results[model] = res
        best_models[model] = (pipe, cal, scales, p_te)
        print(f"[{model}] TEST tuned: macroF1 {res['test_tuned']['macro_f1']:.4f} "
              f"acc {res['test_tuned']['accuracy']:.4f} recall Fatal {res['test_tuned']['recall_Fatal']:.3f} "
              f"Serious {res['test_tuned']['recall_Serious']:.3f} AUC {res['test_tuned']['roc_auc_ovr']:.4f}",
              flush=True)

    winner = max(results, key=lambda m: results[m]["val_tuned"]["macro_f1"])
    pipe, cal, scales, p_te = best_models[winner]
    print("winner (by validation macro-F1):", winner)
    joblib.dump({"pipeline": pipe, "calibrator": cal, "scales": scales, "features": cols,
                 "cats": cats, "nums": nums, "classes": SEVERITY_ORDER, "model": winner},
                MODELS_DIR / "best_model.joblib", compress=3)
    np.save(MODELS_DIR / "test_proba.npy", p_te)
    te[["Accident_Index", "year", "severity", "is_night", "is_urban", "speed_limit", "hour",
        "Road_Type", "Junction_Detail", "Light_Conditions"]].to_parquet(
        MODELS_DIR / "test_meta.parquet", index=False)

    # ---------- secondary protocol 1: stratified random split (comparable to the literature) ----------
    import gc
    del best_models, pipe, tr_small
    gc.collect()
    keep = cols + ["severity", "ksi"]
    df = pd.concat([tr_full[keep], va[keep], te[keep]], ignore_index=True)
    a, rest = train_test_split(df, test_size=0.30, stratify=df.severity, random_state=SEED)
    v, t = train_test_split(rest, test_size=0.5, stratify=rest.severity, random_state=SEED)
    rp = results[winner]
    rpipe = fit_pipe(winner, rp["params"], rp["weight_power"], cats, nums, a[cols], a.severity.values)
    pv, pt = rpipe.predict_proba(v[cols]), rpipe.predict_proba(t[cols])
    sc = tune_class_scales(pv, v.severity.values)
    results["random_split"] = {"argmax": evaluate(t.severity.values, pt),
                               "tuned": evaluate(t.severity.values, pt, sc),
                               "majority_accuracy": float((t.severity == 0).mean())}
    # unweighted version: same model trained without class weights, plain argmax -> best raw accuracy
    upipe = fit_pipe(winner, rp["params"], 0.0, cats, nums, a[cols], a.severity.values)
    results["random_split"]["unweighted_argmax"] = evaluate(t.severity.values, upipe.predict_proba(t[cols]))
    print("random split:", {k: round(v_, 4) for k, v_ in results["random_split"]["unweighted_argmax"].items()
                            if k in ("accuracy", "macro_f1", "roc_auc_ovr")})

    # ---------- secondary protocol 2: binary KSI task on the time-based split ----------
    del df, a, rest, v, t, rpipe, upipe
    gc.collect()
    ytr_b, yva_b, yte_b = tr_full.ksi.values, va.ksi.values, te.ksi.values
    bparams = {k: v_ for k, v_ in rp["params"].items()}
    bpipe = Pipeline([("prep", make_preprocessor(cats, nums)),
                      ("clf", make_classifier(winner, **bparams))])
    bpipe.fit(tr_full[cols], ytr_b)
    pbv, pbt = bpipe.predict_proba(va[cols]), bpipe.predict_proba(te[cols])
    wb = tune_class_scales(pbv, yva_b)
    results["binary_ksi"] = {"argmax": evaluate(yte_b, pbt), "tuned": evaluate(yte_b, pbt, wb),
                             "majority_accuracy": float((yte_b == 0).mean()), "scale": wb.tolist()}
    print("binary KSI test:", {k: round(v_, 4) for k, v_ in results["binary_ksi"]["tuned"].items()
                               if k in ("accuracy", "macro_f1", "roc_auc", "recall_KSI")})
    joblib.dump({"pipeline": bpipe, "scale": wb}, MODELS_DIR / "binary_model.joblib", compress=3)

    json.dump(results, open(TAB_DIR / "final_results.json", "w"), indent=2, default=float)
    print(f"done in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
