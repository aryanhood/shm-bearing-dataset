"""Predictive calibration, temperature scaling, ensembles, and selective risk-coverage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

EPS = 1e-12


def softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable row-wise softmax."""
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - np.max(arr, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / np.sum(exp, axis=-1, keepdims=True), dtype=np.float64)


def negative_log_likelihood(probs: NDArray[np.float64], y: NDArray[np.int64]) -> float:
    """Mean multiclass negative log-likelihood (cross-entropy)."""
    p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0)
    idx = np.arange(len(y))
    return float(-np.mean(np.log(p[idx, y])))


def brier_score(probs: NDArray[np.float64], y: NDArray[np.int64]) -> float:
    """Multiclass Brier score sum_k (p_k - y_k)^2 averaged over samples."""
    p = np.asarray(probs, dtype=np.float64)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def reliability_bins(
    probs: NDArray[np.float64], y: NDArray[np.int64], *, n_bins: int = 15
) -> dict[str, list[float]]:
    """Compute per-bin confidence, accuracy, and counts for reliability diagrams."""
    p = np.asarray(probs, dtype=np.float64)
    conf = np.max(p, axis=1)
    pred = np.argmax(p, axis=1)
    acc = (pred == y).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf: list[float] = []
    bin_acc: list[float] = []
    bin_count: list[int] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        cnt = int(np.sum(mask))
        bin_count.append(cnt)
        if cnt > 0:
            bin_conf.append(round(float(np.mean(conf[mask])), 5))
            bin_acc.append(round(float(np.mean(acc[mask])), 5))
        else:
            bin_conf.append(round(0.5 * (lo + hi), 5))
            bin_acc.append(0.0)
    return {
        "edges": [round(float(e), 4) for e in edges],
        "bin_conf": bin_conf,
        "bin_acc": bin_acc,
        "bin_count": [float(c) for c in bin_count],
    }


def expected_calibration_error(
    probs: NDArray[np.float64], y: NDArray[np.int64], *, n_bins: int = 15
) -> float:
    """Expected Calibration Error (ECE) over equal-width confidence bins (Guo et al., 2017)."""
    p = np.asarray(probs, dtype=np.float64)
    conf = np.max(p, axis=1)
    pred = np.argmax(p, axis=1)
    correct = (pred == y).astype(np.float64)
    n = float(len(y))
    if n == 0.0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (float(cnt) / n) * abs(
                float(np.mean(correct[mask])) - float(np.mean(conf[mask]))
            )
    return float(ece)


@dataclass(frozen=True)
class TemperatureScaler:
    """Post-hoc temperature scaling fitted strictly on held-out calibration logits."""

    temperature: float = 1.0

    @classmethod
    def fit(cls, cal_logits: NDArray[np.float64], cal_y: NDArray[np.int64]) -> TemperatureScaler:
        """Fit T > 0 by minimizing NLL on calibration logits via L-BFGS-B."""
        logits = np.asarray(cal_logits, dtype=np.float64)

        def loss_fn(log_t: NDArray[np.float64]) -> float:
            t = float(np.exp(np.clip(log_t[0], -3.0, 3.0)))
            return negative_log_likelihood(softmax(logits / t), cal_y)

        res = minimize(loss_fn, x0=np.array([0.0]), method="L-BFGS-B", bounds=[(-3.0, 3.0)])
        t_opt = float(np.exp(res.x[0]))
        return cls(temperature=t_opt)

    def predict_proba(self, logits: NDArray[np.float64]) -> NDArray[np.float64]:
        return softmax(np.asarray(logits, dtype=np.float64) / self.temperature)


@dataclass(frozen=True)
class RiskCoverageSummary:
    """Selective classification metrics (Geifman & El-Yaniv, 2017)."""

    aurc: float
    acc_at_80_cov: float
    acc_at_90_cov: float
    full_acc: float
    coverages: list[float]
    risks: list[float]


def selective_risk_coverage(
    probs: NDArray[np.float64], y: NDArray[np.int64], *, n_grid: int = 21
) -> RiskCoverageSummary:
    """Compute Risk-Coverage curve, AURC, and selective accuracy at 80%/90% coverage."""
    p = np.asarray(probs, dtype=np.float64)
    conf = np.max(p, axis=1)
    correct = (np.argmax(p, axis=1) == y).astype(np.float64)

    order = np.argsort(-conf, kind="mergesort")
    sorted_correct = correct[order]
    n = len(sorted_correct)
    cum_correct = np.cumsum(sorted_correct)
    ks = np.arange(1, n + 1, dtype=np.float64)
    cov_all = ks / float(n)
    risk_all = 1.0 - (cum_correct / ks)
    aurc = float(np.trapezoid(risk_all, cov_all))

    def _acc_at(target_cov: float) -> float:
        idx = max(0, min(n - 1, round(target_cov * n) - 1))
        return float(1.0 - risk_all[idx])

    grid_idx = np.unique(
        np.clip(np.round(np.linspace(0.1, 1.0, n_grid) * n).astype(int) - 1, 0, n - 1)
    )
    return RiskCoverageSummary(
        aurc=round(aurc, 5),
        acc_at_80_cov=round(_acc_at(0.80), 4),
        acc_at_90_cov=round(_acc_at(0.90), 4),
        full_acc=round(float(1.0 - risk_all[-1]), 4),
        coverages=[round(float(cov_all[i]), 4) for i in grid_idx],
        risks=[round(float(risk_all[i]), 5) for i in grid_idx],
    )
