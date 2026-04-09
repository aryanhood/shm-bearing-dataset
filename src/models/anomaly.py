"""
Anomaly Detector  ·  Isolation Forest
======================================
Provides an unsupervised anomaly score that complements the supervised
classifier.  Trained on normal (class-0) windows only.

Anomaly score s ∈ (−1, 0):
  s < threshold (e.g. −0.25) → flag as anomalous

Mathematical note
-----------------
The Isolation Forest score is related to the average path length needed
to isolate a point across random binary trees.  Anomalous points have
shorter isolation paths:

  s(x) = 2^{ −E[h(x)] / c(n) }

where h(x) is the tree depth and c(n) is the expected depth for n samples.
We use sklearn's raw score (score_samples) which maps this to (−1, 0).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import IsolationForest

from ..features.extractor import FeatureExtractor
from ..utils.logger import get_logger

log = get_logger("models.anomaly")


class AnomalyDetector:
    """Isolation Forest trained on engineered features of normal data."""

    def __init__(
        self,
        cfg:           Dict[str, Any],
        sampling_rate: int = 12000,
    ) -> None:
        if "models" in cfg:
            m = cfg["models"]["anomaly"]
        else:
            m = cfg.get("anomaly", cfg)
        self.extractor = FeatureExtractor.from_config(cfg, sampling_rate=sampling_rate)
        self.model = IsolationForest(
            contamination = m.get("contamination", 0.05),
            n_estimators  = m.get("n_estimators", 200),
            random_state  = m.get("random_state", 42),
            n_jobs        = -1,
        )
        self._fitted = False

    def fit(self, X_normal: np.ndarray) -> "AnomalyDetector":
        """Train on normal windows only."""
        log.info(f"Fitting IsolationForest on {len(X_normal)} normal samples…")
        F = self.extractor.transform(X_normal)
        self.model.fit(F)
        self._fitted = True
        log.info("AnomalyDetector ready.")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return raw anomaly scores (N,);  more negative ⟹ more anomalous."""
        if not self._fitted:
            raise RuntimeError("Anomaly detector not fitted. Call .fit() or .load() first.")
        return self.model.score_samples(self.extractor.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return +1 (normal) or −1 (anomaly) per window."""
        if not self._fitted:
            raise RuntimeError("Anomaly detector not fitted. Call .fit() or .load() first.")
        return self.model.predict(self.extractor.transform(X))

    def save(self, path: Path | str) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"model": self.model, "extractor": self.extractor}, f)
        log.info(f"AnomalyDetector saved -> {p}")

    def load(self, path: Path | str) -> "AnomalyDetector":
        with open(Path(path), "rb") as f:
            d = pickle.load(f)
        self.model     = d["model"]
        self.extractor = d["extractor"]
        self._fitted   = True
        return self
