"""Honest evaluation: every test fold is a set of recordings the model never saw.

Two protocols, both grouped by recording so no window leaks across the split:

* unseen_load        train on 3 motor loads, test on the 4th (new operating point)
* unseen_fault_size  train on 2 fault diameters, test on the 3rd (new defect severity)

The second is the harder, more realistic one: in the field you never get to
train on the exact defect you will meet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data import Windows
from .model import train
from .physics import CLASS_TITLES, CLASSES

PROTOCOLS = {
    "unseen_load": "train on 3 motor loads, test on the 4th",
    "unseen_fault_size": "train on 2 fault diameters, test on the 3rd",
}


@dataclass(frozen=True)
class Report:
    protocol: str
    folds: list[dict[str, float]]
    accuracy: float
    macro_f1: float
    confusion: NDArray[np.int64]
    per_class_recall: dict[str, float]
    n_windows: int

    def to_json(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "description": PROTOCOLS[self.protocol],
            "n_windows": self.n_windows,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_class_recall": {k: round(v, 4) for k, v in self.per_class_recall.items()},
            "folds": self.folds,
            "classes": list(CLASSES),
            "confusion": self.confusion.tolist(),
        }


def fold_groups(data: Windows, protocol: str) -> NDArray[np.int64]:
    """Fold id per window; -1 means the window always stays in the training set."""
    if protocol == "unseen_load":
        return data.load.copy()
    if protocol == "unseen_fault_size":
        size_fold = {7: 0, 14: 1, 21: 2}
        groups = np.array([size_fold.get(int(d), -1) for d in data.diameter], dtype=np.int64)
        # Normal has no fault size: hold out one load per fold, keep 3 HP for training.
        normal = data.y == CLASSES.index("normal")
        groups[normal] = np.where(data.load[normal] < 3, data.load[normal], -1)
        return groups
    raise ValueError(f"unknown protocol {protocol!r}")


def evaluate(data: Windows, feats: NDArray[np.float64], protocol: str) -> Report:
    groups = fold_groups(data, protocol)
    y_true: list[int] = []
    y_pred: list[int] = []
    folds: list[dict[str, float]] = []
    for g in sorted(set(groups.tolist()) - {-1}):
        test = groups == g
        model = train(feats[~test], data.y[~test])
        pred = model.predict_proba(feats[test]).argmax(axis=1)
        folds.append(
            {
                "fold": int(g),
                "n_test": int(test.sum()),
                "accuracy": round(float(accuracy_score(data.y[test], pred)), 4),
            }
        )
        y_true += data.y[test].tolist()
        y_pred += pred.tolist()
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    recall = cm.diagonal() / cm.sum(axis=1)
    return Report(
        protocol=protocol,
        folds=folds,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        confusion=cm,
        per_class_recall={c: float(r) for c, r in zip(CLASSES, recall, strict=True)},
        n_windows=len(y_true),
    )


def plot_confusion(reports: list[Report], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(reports), figsize=(5.4 * len(reports), 4.9), dpi=150)
    for ax, report in zip(np.atleast_1d(axes), reports, strict=True):
        cm = report.confusion
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                color = "white" if norm[i, j] > 0.6 else "#0f172a"
                ax.text(
                    j,
                    i,
                    f"{norm[i, j]:.0%}\n{cm[i, j]}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )
        ticks = range(len(CLASSES))
        ax.set_xticks(ticks, CLASS_TITLES, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(ticks, CLASS_TITLES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(
            f"{PROTOCOLS[report.protocol]}\naccuracy {report.accuracy:.1%} · "
            f"macro F1 {report.macro_f1:.3f}",
            fontsize=9.5,
        )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save(reports: list[Report], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_json() for r in reports], indent=2) + "\n")
