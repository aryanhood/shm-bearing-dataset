"""
Preprocessor
============
Handles stratified train / val / test splitting and
channel-wise z-score normalisation.

The scaler is fitted on the training set only and applied to all splits,
preventing data leakage.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..utils.logger import get_logger

log = get_logger("data.preprocessor")


class Preprocessor:
    """
    Stratified split + z-score normalisation for 1-D signal windows.

    Parameters
    ----------
    train_frac : float
    val_frac   : float
    test_frac  : float  (inferred as 1 - train - val)
    seed       : int
    """

    def __init__(
        self,
        train_frac: float = 0.70,
        val_frac:   float = 0.15,
        test_frac:  float = 0.15,
        seed:       int   = 42,
    ) -> None:
        if abs(train_frac + val_frac + test_frac - 1.0) >= 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
        self.train_frac = train_frac
        self.val_frac   = val_frac
        self.test_frac  = test_frac
        self.seed       = seed
        self.scaler     = StandardScaler()
        self._fitted    = False

    # ── public ────────────────────────────────────────────────────────────────

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split and normalise the dataset.

        Returns
        -------
        dict with keys 'train', 'val', 'test', each (X_split, y_split).
        X arrays have the same shape as input: (N, window_size).
        """
        self._validate_inputs(X, y)
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y,
            test_size=self.val_frac + self.test_frac,
            stratify=y,
            random_state=self.seed,
        )
        rel_val = self.val_frac / (self.val_frac + self.test_frac)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp,
            test_size=1.0 - rel_val,
            stratify=y_tmp,
            random_state=self.seed,
        )

        # Fit scaler on train only
        X_tr  = self.scaler.fit_transform(X_tr)
        X_val = self.scaler.transform(X_val)
        X_te  = self.scaler.transform(X_te)
        self._fitted = True

        log.info(
            f"Split — train:{len(y_tr)}  val:{len(y_val)}  test:{len(y_te)}"
        )
        return {
            "train": (X_tr.astype(np.float32),  y_tr),
            "val":   (X_val.astype(np.float32), y_val),
            "test":  (X_te.astype(np.float32),  y_te),
        }

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        if X.ndim != 2:
            raise ValueError("Expected X to have shape (n_samples, window_size)")
        expected = int(getattr(self.scaler, "n_features_in_", X.shape[1]))
        if X.shape[1] != expected:
            raise ValueError(f"Expected window_size={expected}, received {X.shape[1]}")
        return self.scaler.transform(X).astype(np.float32)

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return raw stratified splits without fitting the scaler."""
        self._validate_inputs(X, y)
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y,
            test_size=self.val_frac + self.test_frac,
            stratify=y,
            random_state=self.seed,
        )
        rel_val = self.val_frac / (self.val_frac + self.test_frac)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp,
            test_size=1.0 - rel_val,
            stratify=y_tmp,
            random_state=self.seed,
        )
        return {
            "train": (X_tr.astype(np.float32), y_tr),
            "val": (X_val.astype(np.float32), y_val),
            "test": (X_te.astype(np.float32), y_te),
        }

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(
                {
                    "train_frac": self.train_frac,
                    "val_frac": self.val_frac,
                    "test_frac": self.test_frac,
                    "seed": self.seed,
                    "scaler": self.scaler,
                    "_fitted": self._fitted,
                    "n_features_in": int(getattr(self.scaler, "n_features_in_", 0)),
                },
                f,
            )

    def load(self, path: Path | str) -> "Preprocessor":
        with open(Path(path), "rb") as f:
            state = pickle.load(f)
        self.train_frac = state["train_frac"]
        self.val_frac = state["val_frac"]
        self.test_frac = state["test_frac"]
        self.seed = state["seed"]
        self.scaler = state["scaler"]
        self._fitted = state["_fitted"]
        return self

    @staticmethod
    def class_weights(y: np.ndarray) -> Dict[int, float]:
        """Inverse-frequency class weights for imbalanced data."""
        classes, counts = np.unique(y, return_counts=True)
        total = len(y)
        return {int(c): float(total / (len(classes) * cnt))
                for c, cnt in zip(classes, counts)}

    @staticmethod
    def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError("Expected X to have shape (n_samples, window_size)")
        if y.ndim != 1:
            raise ValueError("Expected y to have shape (n_samples,)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError("At least two classes are required for stratified splitting.")
        too_small = {int(c): int(cnt) for c, cnt in zip(classes, counts) if cnt < 3}
        if too_small:
            raise ValueError(
                "Each class needs at least 3 samples for train/val/test stratification. "
                f"Too-small classes: {too_small}"
            )
