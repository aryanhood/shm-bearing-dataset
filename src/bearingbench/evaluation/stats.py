"""Multi-seed summary statistics, paired bootstrap CIs, and exact McNemar hypothesis tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest


@dataclass(frozen=True)
class SeedStats:
    mean: float
    std: float


def mean_std(values: list[float] | NDArray[np.float64]) -> SeedStats:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return SeedStats(0.0, 0.0)
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return SeedStats(round(float(np.mean(arr)), 4), round(sd, 4))


def paired_bootstrap_acc_diff(
    y_true: NDArray[np.int64],
    pred_a: NDArray[np.int64],
    pred_b: NDArray[np.int64],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap 95% CI for accuracy(A) - accuracy(B) over test samples."""
    corr_a = (pred_a == y_true).astype(np.float64)
    corr_b = (pred_b == y_true).astype(np.float64)
    diff = corr_a - corr_b
    n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "diff_mean": round(float(diff.mean()), 4),
        "ci95_low": round(float(lo), 4),
        "ci95_high": round(float(hi), 4),
    }


def mcnemar_exact(
    y_true: NDArray[np.int64],
    pred_a: NDArray[np.int64],
    pred_b: NDArray[np.int64],
) -> dict[str, float | int]:
    """Exact two-sided binomial McNemar test on discordant test predictions."""
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    a_wins = int(np.sum(a_ok & (~b_ok)))
    b_wins = int(np.sum((~a_ok) & b_ok))
    n_disc = a_wins + b_wins
    p_val = (
        1.0
        if n_disc == 0
        else float(binomtest(a_wins, n_disc, 0.5, alternative="two-sided").pvalue)
    )
    return {
        "a_only_correct": a_wins,
        "b_only_correct": b_wins,
        "n_discordant": n_disc,
        "p_value": round(p_val, 6) if not math.isnan(p_val) else 1.0,
    }
