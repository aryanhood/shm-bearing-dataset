"""Tests for zero-label baseline learning, drift scoring, and before/after comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vibcheck.baseline import BaselineProfile, learn_baseline, score_against_baseline
from vibcheck.compare import compare_signals
from vibcheck.io import SignalData


def _synth_machine(fs_hz: float, *, faulty: bool, seed: int) -> SignalData:
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs_hz * 2.0), dtype=np.float64) / fs_hz
    base = 0.04 * np.sin(2.0 * np.pi * 30.0 * t) + rng.normal(0.0, 0.03, size=len(t))
    if faulty:
        impulses = np.exp(-((t * 158.0) % 1.0) * 35.0) * np.sin(2.0 * np.pi * 3000.0 * t)
        base = base + 0.45 * impulses
    return SignalData(base, fs_hz, "g", "faulty.csv" if faulty else "healthy.csv", "accel_g")


def test_baseline_roundtrip_and_drift_separation(tmp_path: Path) -> None:
    healthy_train = _synth_machine(12_000.0, faulty=False, seed=1)
    healthy_test = _synth_machine(12_000.0, faulty=False, seed=2)
    faulty_test = _synth_machine(12_000.0, faulty=True, seed=3)

    prof = learn_baseline(healthy_train, machine_name="Workshop Fan")
    p_path = tmp_path / "fan.json"
    prof.save(p_path)
    loaded = BaselineProfile.load(p_path)
    assert loaded.machine_name == "Workshop Fan"

    d_ok = score_against_baseline(healthy_test, loaded)
    d_bad = score_against_baseline(faulty_test, loaded)
    assert d_ok.drift_score < 25.0
    assert d_bad.drift_score > 65.0
    assert d_bad.rms_ratio > 1.2
    assert d_bad.kurtosis_delta > 10.0


def test_compare_signals_detects_improvement() -> None:
    before = _synth_machine(12_000.0, faulty=True, seed=10)
    after = _synth_machine(12_000.0, faulty=False, seed=11)
    comp = compare_signals(before, after, rpm=1800.0, bearing="6205")
    assert comp.verdict == "IMPROVED"
    assert comp.rms_change_pct < -20.0
