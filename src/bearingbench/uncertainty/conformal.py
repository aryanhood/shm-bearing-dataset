"""Split Conformal Prediction (APS) and Online Adaptive Conformal Inference (ACI)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _aps_scores_all(
    probs: NDArray[np.float64],
    *,
    u_vec: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute APS cumulative score S(x, c) for every sample and candidate class c."""
    n = probs.shape[0]
    u = np.full(n, 0.5, dtype=np.float64) if u_vec is None else u_vec
    order = np.argsort(-probs, axis=1, kind="mergesort")
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum_incl = np.cumsum(sorted_p, axis=1)
    cum_before = cum_incl - sorted_p
    sorted_scores = cum_before + u[:, None] * sorted_p

    scores = np.empty_like(sorted_scores)
    row_idx = np.arange(n)[:, None]
    scores[row_idx, order] = sorted_scores
    return scores


def aps_prediction_sets(
    probs: NDArray[np.float64],
    q_hat: float,
    *,
    seed: int = 42,
    allow_empty: bool = False,
) -> list[tuple[int, ...]]:
    """Construct APS prediction sets {c : S(x, c) <= q_hat}."""
    rng = np.random.default_rng(seed)
    u_vec = rng.uniform(0.0, 1.0, size=len(probs))
    scores = _aps_scores_all(probs, u_vec=u_vec)
    sets: list[tuple[int, ...]] = []
    for i in range(len(probs)):
        included = [
            int(c) for c in np.argsort(-probs[i], kind="mergesort") if scores[i, c] <= q_hat
        ]
        if not included and not allow_empty:
            included = [int(np.argmax(probs[i]))]
        sets.append(tuple(included))
    return sets


@dataclass(frozen=True)
class ConformalMetrics:
    """Evaluation metrics for conformal prediction sets."""

    coverage: float
    avg_set_size: float
    singleton_rate: float
    singleton_accuracy: float
    target_coverage: float


def evaluate_prediction_sets(
    sets: list[tuple[int, ...]], y_true: NDArray[np.int64], *, target_coverage: float
) -> ConformalMetrics:
    """Compute empirical coverage, mean set size, singleton rate & selective singleton accuracy."""
    n = len(y_true)
    if n == 0:
        return ConformalMetrics(0.0, 0.0, 0.0, 0.0, target_coverage)
    covered = [1 if int(y_true[i]) in sets[i] else 0 for i in range(n)]
    sizes = [len(s) for s in sets]
    singletons = [i for i, s in enumerate(sets) if len(s) == 1]
    sing_acc = (
        sum(1 for i in singletons if sets[i][0] == int(y_true[i])) / float(len(singletons))
        if singletons
        else 0.0
    )
    return ConformalMetrics(
        coverage=round(float(sum(covered) / n), 4),
        avg_set_size=round(float(sum(sizes) / n), 4),
        singleton_rate=round(float(len(singletons) / n), 4),
        singleton_accuracy=round(float(sing_acc), 4),
        target_coverage=round(target_coverage, 4),
    )


@dataclass(frozen=True)
class SplitAPSConformal:
    """Finite-sample Split Conformal Adaptive Prediction Sets (Romano et al., 2020)."""

    alpha: float
    q_hat: float
    cal_scores: NDArray[np.float64]

    @classmethod
    def fit(
        cls,
        cal_probs: NDArray[np.float64],
        cal_y: NDArray[np.int64],
        *,
        alpha: float = 0.10,
        seed: int = 42,
    ) -> SplitAPSConformal:
        rng = np.random.default_rng(seed)
        u_vec = rng.uniform(0.0, 1.0, size=len(cal_y))
        all_scores = _aps_scores_all(cal_probs, u_vec=u_vec)
        cal_scores = all_scores[np.arange(len(cal_y)), cal_y]
        n = len(cal_scores)
        level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / float(n))
        q_hat = float(np.quantile(cal_scores, level, method="higher")) if level < 1.0 else 1.01
        return cls(alpha=alpha, q_hat=q_hat, cal_scores=cal_scores)

    def predict_sets(self, probs: NDArray[np.float64], *, seed: int = 123) -> list[tuple[int, ...]]:
        return aps_prediction_sets(probs, self.q_hat, seed=seed, allow_empty=False)

    def evaluate(
        self, probs: NDArray[np.float64], y_true: NDArray[np.int64], *, seed: int = 123
    ) -> ConformalMetrics:
        sets = self.predict_sets(probs, seed=seed)
        return evaluate_prediction_sets(sets, y_true, target_coverage=1.0 - self.alpha)


@dataclass(frozen=True)
class ACIResult:
    """Online Adaptive Conformal Inference output across a non-stationary test stream."""

    metrics: ConformalMetrics
    sets: list[tuple[int, ...]]
    alpha_trajectory: list[float]
    covered_stream: list[int]
    rolling_coverage: list[float]


def run_adaptive_conformal_inference(
    cal_scores: NDArray[np.float64],
    test_probs: NDArray[np.float64],
    test_y: NDArray[np.int64],
    *,
    alpha_target: float = 0.10,
    gamma: float = 0.05,
    rolling_window: int = 40,
    seed: int = 123,
) -> ACIResult:
    """Adaptive Conformal Inference (Gibbs & Candes, NeurIPS 2021) for non-stationary streams."""
    rng = np.random.default_rng(seed)
    n_test = len(test_y)
    u_vec = rng.uniform(0.0, 1.0, size=n_test)
    test_scores = _aps_scores_all(test_probs, u_vec=u_vec)

    n_cal = len(cal_scores)
    alpha_t = alpha_target
    sets: list[tuple[int, ...]] = []
    alphas: list[float] = []
    covered_stream: list[int] = []

    for t in range(n_test):
        alpha_eff = float(np.clip(alpha_t, 0.0, 1.0))
        if alpha_eff <= 0.0:
            q_t = 1.01
        elif alpha_eff >= 1.0:
            q_t = -1.0
        else:
            lvl = min(1.0, math.ceil((n_cal + 1) * (1.0 - alpha_eff)) / float(n_cal))
            q_t = float(np.quantile(cal_scores, lvl, method="higher")) if lvl < 1.0 else 1.01

        row_scores = test_scores[t]
        included = [
            int(c) for c in np.argsort(-test_probs[t], kind="mergesort") if row_scores[c] <= q_t
        ]
        if not included:
            included = [int(np.argmax(test_probs[t]))]
        c_set = tuple(included)
        sets.append(c_set)
        alphas.append(round(alpha_eff, 4))

        err_t = 0.0 if int(test_y[t]) in c_set else 1.0
        covered_stream.append(int(1.0 - err_t))
        alpha_t = float(np.clip(alpha_t + gamma * (alpha_target - err_t), 0.0, 1.0))

    rolling: list[float] = []
    for t in range(n_test):
        lo = max(0, t - rolling_window + 1)
        rolling.append(round(float(sum(covered_stream[lo : t + 1]) / (t - lo + 1)), 4))

    metrics = evaluate_prediction_sets(sets, test_y, target_coverage=1.0 - alpha_target)
    return ACIResult(
        metrics=metrics,
        sets=sets,
        alpha_trajectory=alphas,
        covered_stream=covered_stream,
        rolling_coverage=rolling,
    )
