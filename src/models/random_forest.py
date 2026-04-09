"""
Random Forest Baseline
======================
Pipeline:  raw window (N, W)  →  FeatureExtractor  →  RandomForestClassifier

Why RF?
-------
- Interpretable via feature importances
- No normalisation required
- Handles class imbalance via ``class_weight='balanced'``
- Fast to train; strong baseline for bearing fault classification

The feature extractor adds 8 time-domain + 67 frequency-domain = 75 features
per window.  RF uses 300 trees with no depth limit to capture nonlinear
interactions between fault indicators (kurtosis × dominant_freq, etc.).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..features.extractor import FeatureExtractor
from ..utils.logger import get_logger
from .base import BaseModel

log = get_logger("models.rf")


class RandomForestModel(BaseModel):
    """Engineered-feature Random Forest classifier."""

    def __init__(
        self,
        cfg:           Dict[str, Any],
        sampling_rate: int  = 12000,
        seed:          int  = 42,
    ) -> None:
        super().__init__(cfg, seed)
        m = cfg["models"]["baseline"] if "models" in cfg else cfg.get("baseline", cfg)
        self.extractor = FeatureExtractor.from_config(cfg, sampling_rate=sampling_rate)
        self.clf = RandomForestClassifier(
            n_estimators = m.get("n_estimators",     300),
            max_depth    = m.get("max_depth",        None),
            min_samples_split = m.get("min_samples_split", 4),
            class_weight = m.get("class_weight",     "balanced"),
            n_jobs       = -1,
            random_state = seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **_) -> "RandomForestModel":
        log.info(f"Extracting features from {len(X)} windows…")
        F = self.extractor.transform(X)
        log.info(f"Feature matrix: {F.shape}  — training RF…")
        self.clf.fit(F, y)
        self._fitted = True
        log.info("RF training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.clf.predict(self.extractor.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.clf.predict_proba(self.extractor.transform(X))

    def feature_importances(self) -> np.ndarray:
        self._check_fitted()
        return self.clf.feature_importances_

    def save(self, path: Path | str) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"clf": self.clf, "extractor": self.extractor}, f)
        log.info(f"RF saved -> {p}")

    def load(self, path: Path | str) -> "RandomForestModel":
        with open(Path(path), "rb") as f:
            d = pickle.load(f)
        self.clf       = d["clf"]
        self.extractor = d["extractor"]
        self._fitted   = True
        return self
