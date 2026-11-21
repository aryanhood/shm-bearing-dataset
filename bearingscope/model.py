"""A small classifier that is easy to ship: standardize -> MLP -> softmax.

The trained model is exported as plain JSON and executed in the browser by
web/src/lib/model.ts, so the live demo needs no server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES
from .physics import CLASSES

HIDDEN = (32,)
SEED = 7


@dataclass(frozen=True)
class Model:
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    weights: list[NDArray[np.float64]]
    biases: list[NDArray[np.float64]]

    def predict_proba(self, feats: NDArray[np.float64]) -> NDArray[np.float64]:
        h = (np.atleast_2d(feats) - self.mean) / self.scale
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0.0)
        h = h - h.max(axis=1, keepdims=True)
        e = np.exp(h)
        return np.asarray(e / e.sum(axis=1, keepdims=True))

    def to_json(self) -> dict[str, Any]:
        return {
            "classes": list(CLASSES),
            "features": list(FEATURE_NAMES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "layers": [
                {"weights": w.tolist(), "bias": b.tolist()}
                for w, b in zip(self.weights, self.biases, strict=True)
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), separators=(",", ":")))

    @classmethod
    def load(cls, path: Path) -> Model:
        d = json.loads(path.read_text())
        return cls(
            mean=np.array(d["mean"]),
            scale=np.array(d["scale"]),
            weights=[np.array(layer["weights"]) for layer in d["layers"]],
            biases=[np.array(layer["bias"]) for layer in d["layers"]],
        )


def train(feats: NDArray[np.float64], labels: NDArray[np.int64]) -> Model:
    scaler = StandardScaler().fit(feats)
    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN, alpha=1e-3, max_iter=2000, random_state=SEED
    ).fit(scaler.transform(feats), labels)
    if list(clf.classes_) != list(range(len(CLASSES))):
        raise ValueError("training data must contain every class")
    return Model(
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        weights=[np.asarray(w, dtype=np.float64) for w in clf.coefs_],
        biases=[np.asarray(b, dtype=np.float64) for b in clf.intercepts_],
    )
