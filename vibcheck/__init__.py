"""vibcheck: plain-English vibration and bearing health diagnostics."""

from __future__ import annotations

from .baseline import BaselineDrift, BaselineProfile, learn_baseline, score_against_baseline
from .bearings import BEARINGS, BearingSpec, get_bearing
from .compare import ComparisonResult, compare_signals
from .diagnostics import DiagnosisResult, analyze
from .io import SignalData, load_signal
from .report import format_terminal_report, write_comparison_html, write_html_report

__version__ = "0.1.0"

__all__ = [
    "BEARINGS",
    "BaselineDrift",
    "BaselineProfile",
    "BearingSpec",
    "ComparisonResult",
    "DiagnosisResult",
    "SignalData",
    "__version__",
    "analyze",
    "compare_signals",
    "format_terminal_report",
    "get_bearing",
    "learn_baseline",
    "load_signal",
    "score_against_baseline",
    "write_comparison_html",
    "write_html_report",
]
