"""Tests for ISO 20816-3 integration, signal quality audit, and bearing fault detection."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from vibcheck.diagnostics import analyze, audit_signal, compute_iso_velocity
from vibcheck.io import SignalData, load_signal


def test_iso_velocity_matches_exact_analytical_sine_wave() -> None:
    fs_hz = 4000.0
    t = np.arange(int(fs_hz * 2.0), dtype=np.float64) / fs_hz
    freq_hz = 50.0
    amp_ms2 = 2.0
    sig = SignalData(
        values=amp_ms2 * np.sin(2.0 * np.pi * freq_hz * t),
        fs_hz=fs_hz,
        units="m/s2",
        source_name="sine.csv",
        channel="z",
    )
    iso = compute_iso_velocity(sig, machine_class="group2_rigid")
    assert iso is not None
    expected_mm_s = 1000.0 * amp_ms2 / (2.0 * math.pi * freq_hz * math.sqrt(2.0))
    assert iso.velocity_rms_mm_s == pytest.approx(expected_mm_s, rel=0.01)
    assert iso.zone == "D"

    audio_sig = SignalData(sig.values, fs_hz, "audio", "mic.wav", "audio_mono")
    assert compute_iso_velocity(audio_sig) is None


def test_audit_catches_flatline_and_clipping() -> None:
    dead = SignalData(np.zeros(2000), 1000.0, "g", "dead.csv", "a")
    assert not audit_signal(dead).ok

    t = np.arange(4000, dtype=np.float64) / 2000.0
    clipped = SignalData(
        np.clip(5.0 * np.sin(2.0 * np.pi * 30.0 * t), -1.0, 1.0), 2000.0, "g", "c.csv", "a"
    )
    q = audit_signal(clipped)
    assert not q.ok
    assert any("clipping" in msg.lower() for msg in q.issues)


def test_bundled_example_detects_6205_inner_race_fault() -> None:
    ex_path = Path("examples/motor_inner_fault.csv")
    if not ex_path.exists():
        pytest.skip("bundled example missing")
    sig = load_signal(ex_path)
    assert sig.fs_hz == pytest.approx(12_000.0, rel=1e-3)
    res = analyze(sig, rpm=1750.0, bearing="6205")
    assert res.status in {"INSPECT SOON", "CRITICAL"}
    bpfi_hits = [m for m in res.fault_matches if "BPFI" in m.name and m.detected]
    assert len(bpfi_hits) == 1
    assert bpfi_hits[0].snr_ratio > 10.0
