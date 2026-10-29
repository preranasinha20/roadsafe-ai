"""Evaluation helpers: imbalance-aware metrics and class-prior tuning."""
import itertools

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

from .config import SEVERITY_ORDER


def tune_class_scales(proba, y, grid=None):
    """Find per-class multipliers on the predicted probabilities that maximise macro-F1.

    This is the 'threshold tuning' method of the imbalance study. It is fitted on the
    validation year only and then applied unchanged to the test years.
    """
    k = proba.shape[1]
    grid = grid if grid is not None else np.exp(np.linspace(0, np.log(25), 13))
    best, best_w = -1, np.ones(k)
    if k == 2:
        for w1 in np.exp(np.linspace(0, np.log(8), 40)):
            w = np.array([1, w1]); f = f1_score(y, (proba * w).argmax(1), average="macro")
            if f > best: best, best_w = f, w
        return best_w
    for w1, w2 in itertools.product(grid, grid):
        w = np.array([1.0, w1, w2 * w1 / 2])
        f = f1_score(y, (proba * w).argmax(1), average="macro")
        if f > best:
            best, best_w = f, w
    return best_w


def expected_calibration_error(proba, y, n_bins=15):
    conf = proba.max(1)
    pred = proba.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return ece


def evaluate(y, proba, weights=None, labels=SEVERITY_ORDER):
    """Return a flat dict of metrics for a 3-class (or binary) problem."""
    y = np.asarray(y)
    k = proba.shape[1]
    w = np.ones(k) if weights is None else np.asarray(weights)
    pred = (proba * w).argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(k), zero_division=0)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "mcc": matthews_corrcoef(y, pred),
    }
    for i, name in enumerate(labels[:k] if k == 3 else ["Non-KSI", "KSI"]):
        out[f"precision_{name}"], out[f"recall_{name}"], out[f"f1_{name}"] = p[i], r[i], f[i]
    if k == 3:
        out["recall_KSI"] = float(((pred >= 1) & (y >= 1)).sum() / max((y >= 1).sum(), 1))
        out["roc_auc_ovr"] = roc_auc_score(y, proba, multi_class="ovr", average="macro")
        out["pr_auc_Serious"] = average_precision_score(y == 1, proba[:, 1])
        out["pr_auc_Fatal"] = average_precision_score(y == 2, proba[:, 2])
    else:
        out["roc_auc"] = roc_auc_score(y, proba[:, 1])
        out["pr_auc_KSI"] = average_precision_score(y, proba[:, 1])
    return out


def confusion(y, proba, weights=None):
    w = np.ones(proba.shape[1]) if weights is None else weights
    return confusion_matrix(y, (proba * w).argmax(1))
