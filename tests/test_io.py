"""Tests for CSV (Phyphox/DAQ), WAV audio, and MAT signal loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat, wavfile

from vibcheck.io import load_signal


def test_load_phyphox_style_csv_infers_fs_and_units(tmp_path: Path) -> None:
    csv_path = tmp_path / "phyphox.csv"
    t = np.arange(500, dtype=np.float64) / 200.0
    x = 0.05 * np.sin(2.0 * np.pi * 25.0 * t)
    z = 0.80 * np.sin(2.0 * np.pi * 30.0 * t)
    lines = ["Time (s),Linear Acceleration x (m/s^2),Linear Acceleration z (m/s^2)"]
    lines.extend(f"{ti:.6f},{xi:.5f},{zi:.5f}" for ti, xi, zi in zip(t, x, z, strict=True))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sig = load_signal(csv_path)
    assert sig.fs_hz == pytest.approx(200.0, rel=1e-3)
    assert sig.units == "m/s2"
    # Auto-picks highest-variance axis (z)
    assert "z" in sig.channel.lower()


def test_load_single_col_csv_wav_and_mat(tmp_path: Path) -> None:
    raw_csv = tmp_path / "single.csv"
    vals = np.sin(2.0 * np.pi * 50.0 * np.arange(400) / 1000.0)
    raw_csv.write_text("\n".join(f"{v:.5f}" for v in vals), encoding="utf-8")
    with pytest.raises(ValueError):
        load_signal(raw_csv)
    s_csv = load_signal(raw_csv, fs_hz=1000.0, units="g")
    assert s_csv.fs_hz == 1000.0
    assert s_csv.units == "g"

    wav_path = tmp_path / "fan.wav"
    pcm = (vals * 20000.0).astype(np.int16)
    wavfile.write(wav_path, 8000, pcm)
    s_wav = load_signal(wav_path)
    assert s_wav.fs_hz == 8000.0
    assert s_wav.units == "audio"
    assert float(np.max(np.abs(s_wav.values))) <= 1.0

    mat_path = tmp_path / "sample.mat"
    savemat(mat_path, {"X105_DE_time": vals[:, None]})
    s_mat = load_signal(mat_path)
    assert s_mat.fs_hz == 12_000.0
    assert s_mat.units == "g"
