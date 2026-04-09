"""Centralized evaluation helpers."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float | np.ndarray | str]:
    """Compute the evaluation suite used across experiments."""
    results: Dict[str, float | np.ndarray | str] = {}

    results["accuracy"] = float(accuracy_score(y_true, y_pred))
    results["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    results["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    results["precision_weighted"] = float(
        precision_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    results["recall_weighted"] = float(
        recall_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    if y_proba is not None:
        n_classes = y_proba.shape[1]
        if n_classes == 2:
            results["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            try:
                results["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
            except ValueError:
                results["roc_auc"] = float("nan")

    normal_mask = y_true == 0
    if normal_mask.sum() > 0:
        results["false_alarm_rate"] = float((y_pred[normal_mask] != 0).sum() / normal_mask.sum())
    else:
        results["false_alarm_rate"] = float("nan")

    results["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    results["report"] = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    return results


def print_report(metrics: Dict[str, float | np.ndarray | str], title: str = "") -> None:
    """Pretty-print a metrics dictionary using ASCII-safe output."""
    if title:
        print(f"\n{'-' * 60}")
        print(f"  {title}")
        print(f"{'-' * 60}")
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print(f"  confusion_matrix:\n{value}")
        elif key == "report":
            print(f"\n{value}")
        elif isinstance(value, float):
            print(f"  {key:<28} {value:.4f}")
        else:
            print(f"  {key}: {value}")
