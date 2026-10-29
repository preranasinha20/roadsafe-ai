"""Shared model artefact helpers: the calibrator class and a loader that can unpickle it."""
import sys

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .config import MODELS_DIR


class Calibrator:
    """One-vs-rest isotonic calibration, re-normalised so the rows sum to one."""

    def fit(self, proba, y):
        self.iso = [IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5)
                    .fit(proba[:, k], (y == k).astype(int)) for k in range(proba.shape[1])]
        return self

    def transform(self, proba):
        out = np.column_stack([m.predict(proba[:, k]) for k, m in enumerate(self.iso)])
        return out / out.sum(1, keepdims=True)


def load_model(name="best_model.joblib"):
    # artefacts trained by `python -m src.train_final` reference the class under __main__
    sys.modules["__main__"].Calibrator = Calibrator
    return joblib.load(MODELS_DIR / name)
