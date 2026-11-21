"""Turn class probabilities into a maintenance decision. Mirrored in web/src/lib/decision.ts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .physics import CLASS_TITLES, CLASSES

SAFE_MIN_NORMAL = 0.80  # confident the bearing is healthy
CRITICAL_MIN_FAULT = 0.80  # confident a specific defect is present


@dataclass(frozen=True)
class Decision:
    status: str  # SAFE | WATCH | CRITICAL
    label: str
    title: str
    confidence: float
    action: str


def decide(proba: NDArray[np.float64]) -> Decision:
    p = np.asarray(proba, dtype=np.float64).ravel()
    k = int(np.argmax(p))
    label, title, conf = CLASSES[k], CLASS_TITLES[k], float(p[k])
    if label == "normal" and conf >= SAFE_MIN_NORMAL:
        return Decision("SAFE", label, title, conf, "No action. Keep monitoring.")
    if label != "normal" and conf >= CRITICAL_MIN_FAULT:
        return Decision(
            "CRITICAL", label, title, conf, f"{title} detected. Schedule bearing replacement."
        )
    return Decision(
        "WATCH", label, title, conf, "Signal is ambiguous. Re-check at shorter interval."
    )
