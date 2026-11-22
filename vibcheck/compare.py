"""Before-vs-after repair comparison for rotating machinery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .bearings import BearingSpec
from .diagnostics import DiagnosisResult, IsoClass, analyze
from .io import SignalData

Verdict = Literal["IMPROVED", "WORSE", "UNCHANGED"]


@dataclass(frozen=True)
class ComparisonResult:
    before: DiagnosisResult
    after: DiagnosisResult
    rms_change_pct: float
    velocity_change_pct: float | None
    kurtosis_delta: float
    verdict: Verdict
    summary: str


def compare_signals(
    before_sig: SignalData,
    after_sig: SignalData,
    *,
    rpm: float | None = None,
    bearing: str | BearingSpec | None = None,
    iso_class: IsoClass = "group2_rigid",
) -> ComparisonResult:
    """Compare two recordings (e.g. before and after maintenance) in plain English."""
    b_res = analyze(before_sig, rpm=rpm, bearing=bearing, iso_class=iso_class)
    a_res = analyze(after_sig, rpm=rpm, bearing=bearing, iso_class=iso_class)

    rms_pct = ((a_res.rms - b_res.rms) / max(b_res.rms, 1e-12)) * 100.0
    vel_pct: float | None = None
    if b_res.iso is not None and a_res.iso is not None and b_res.iso.velocity_rms_mm_s > 0.0:
        vel_pct = (
            (a_res.iso.velocity_rms_mm_s - b_res.iso.velocity_rms_mm_s)
            / b_res.iso.velocity_rms_mm_s
        ) * 100.0
    kurt_delta = a_res.kurtosis - b_res.kurtosis

    if rms_pct <= -15.0 or (kurt_delta <= -2.0 and rms_pct <= 5.0):
        verdict: Verdict = "IMPROVED"
        summary = (
            f"Vibration improved: RMS changed {rms_pct:+.1f}% "
            f"({b_res.rms:.4f} -> {a_res.rms:.4f} {a_res.units}), "
            f"kurtosis {b_res.kurtosis:.1f} -> {a_res.kurtosis:.1f}."
        )
    elif rms_pct >= 15.0 or kurt_delta >= 2.0:
        verdict = "WORSE"
        summary = (
            f"Vibration got worse: RMS changed {rms_pct:+.1f}% "
            f"({b_res.rms:.4f} -> {a_res.rms:.4f} {a_res.units}), "
            f"kurtosis {b_res.kurtosis:.1f} -> {a_res.kurtosis:.1f}."
        )
    else:
        verdict = "UNCHANGED"
        summary = (
            f"No major change: RMS changed {rms_pct:+.1f}% "
            f"({b_res.rms:.4f} -> {a_res.rms:.4f} {a_res.units}), "
            f"kurtosis {b_res.kurtosis:.1f} -> {a_res.kurtosis:.1f}."
        )

    return ComparisonResult(
        before=b_res,
        after=a_res,
        rms_change_pct=round(rms_pct, 1),
        velocity_change_pct=round(vel_pct, 1) if vel_pct is not None else None,
        kurtosis_delta=round(kurt_delta, 2),
        verdict=verdict,
        summary=summary,
    )
