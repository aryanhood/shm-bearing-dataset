"""Calibration, conformal prediction sets, and online ACI under shift."""

from __future__ import annotations

from .calibration import (
    RiskCoverageSummary,
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    selective_risk_coverage,
    softmax,
)
from .conformal import (
    ACIResult,
    ConformalMetrics,
    SplitAPSConformal,
    aps_prediction_sets,
    evaluate_prediction_sets,
    run_adaptive_conformal_inference,
)

__all__ = [
    "ACIResult",
    "ConformalMetrics",
    "RiskCoverageSummary",
    "SplitAPSConformal",
    "TemperatureScaler",
    "aps_prediction_sets",
    "brier_score",
    "evaluate_prediction_sets",
    "expected_calibration_error",
    "negative_log_likelihood",
    "reliability_bins",
    "run_adaptive_conformal_inference",
    "selective_risk_coverage",
    "softmax",
]
