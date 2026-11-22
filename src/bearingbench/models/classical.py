"""Leak-free scikit-learn tabular pipelines (StandardScaler inside folds)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_random_forest(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    *,
    seed: int = 42,
    n_estimators: int = 200,
) -> Pipeline:
    """Fit StandardScaler -> RandomForestClassifier strictly on training fold."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    return pipe


def train_mlp(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    *,
    seed: int = 42,
    hidden_layer_sizes: tuple[int, ...] = (32,),
) -> Pipeline:
    """Fit StandardScaler -> MLPClassifier (BearingScope baseline)."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    alpha=1e-3,
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    return pipe


def predict_proba_sklearn(
    pipe: Pipeline, x: NDArray[np.float64], *, n_classes: int
) -> NDArray[np.float64]:
    """Return full probability matrix aligned to class indices [0..n_classes-1]."""
    raw = np.asarray(pipe.predict_proba(x), dtype=np.float64)
    if raw.shape[1] == n_classes:
        return raw
    clf = pipe[-1]
    classes = np.asarray(clf.classes_, dtype=np.int64)
    out = np.zeros((len(x), n_classes), dtype=np.float64)
    out[:, classes] = raw
    return out
