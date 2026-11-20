"""Ranking model: gradient-boosted trees + isotonic calibration.

The model never decides who is listed - the rules do. It only orders the
students that a rule has already flagged (R2 section 4.1).
Time-ordered split: fit on weeks 5-18, calibrate on weeks 19-22, test on
weeks 29-34 (labels look 6 weeks ahead, so no test label is visible in
training).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C
from .features import FEATURES

try:  # LightGBM is the design choice; scikit-learn is the fallback
    import lightgbm as lgb

    BACKEND = "lightgbm"
except Exception:  # pragma: no cover - only when the wheel is unavailable
    lgb = None
    BACKEND = "sklearn"

from sklearn.isotonic import IsotonicRegression


def _rows(feats: np.ndarray, labels: np.ndarray, weeks: tuple[int, int]):
    ts = range(weeks[0] - 1, weeks[1])
    X = np.concatenate([feats[t] for t in ts])
    y = np.concatenate([labels[t] for t in ts])
    keep = ~np.isnan(y)
    return X[keep], y[keep].astype(int)


@dataclass
class RankModel:
    model: object
    iso: IsotonicRegression
    backend: str

    def raw(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.iso.predict(self.raw(X))

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Per-feature attribution (TreeSHAP values in log-odds), shape (n, F)."""
        if self.backend == "lightgbm":
            return self.model.booster_.predict(X, pred_contrib=True)[:, :-1]
        # fallback: permutation-free proxy = feature importance x standardised value
        imp = self.model.feature_importances_
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
        return np.nan_to_num((X - mu) / sd) * imp


def train(feats: np.ndarray, labels: np.ndarray, seed: int = C.SEED) -> tuple[RankModel, dict]:
    fit_weeks = (C.TRAIN_WEEKS[0], C.CALIB_WEEKS[0] - 1)
    Xf, yf = _rows(feats, labels, fit_weeks)
    Xc, yc = _rows(feats, labels, C.CALIB_WEEKS)
    if BACKEND == "lightgbm":
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, verbose=-1,
        )
    else:  # pragma: no cover
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, random_state=seed)
    model.fit(Xf, yf)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(model.predict_proba(Xc)[:, 1], yc)
    info = {
        "backend": BACKEND, "features": len(FEATURES),
        "fit_rows": len(yf), "fit_positive_rate": round(float(yf.mean()), 4),
        "calibration_rows": len(yc),
        "fit_weeks": list(fit_weeks), "calibration_weeks": list(C.CALIB_WEEKS), "test_weeks": list(C.TEST_WEEKS),
    }
    return RankModel(model=model, iso=iso, backend=BACKEND), info
