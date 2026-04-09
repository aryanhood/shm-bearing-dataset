"""
Abstract base model
===================
Defines the interface every SHM model must implement.
Keeps the rest of the system decoupled from concrete implementations.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict

import numpy as np


class BaseModel(abc.ABC):

    def __init__(self, cfg: Dict[str, Any], seed: int = 42) -> None:
        self.cfg     = cfg
        self.seed    = seed
        self._fitted = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kw) -> "BaseModel":
        """Train the model in-place; return self."""

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer class labels (N,)."""

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability matrix (N, C)."""

    @abc.abstractmethod
    def save(self, path: Path | str) -> None: ...

    @abc.abstractmethod
    def load(self, path: Path | str) -> "BaseModel": ...

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() or .load() first.")

    def predict_with_conf(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: returns (predictions, max_probability)."""
        proba = self.predict_proba(X)
        return proba.argmax(axis=1), proba.max(axis=1)
