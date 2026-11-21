"""Synthetic variable-RPM encoder decoding and sensor audit tests."""

from __future__ import annotations

import numpy as np

from orderguard.encoder import audit_sensors, decode_encoder


def _synth_ttl_encoder(
    fs_hz: float,
    duration_s: float,
    ppr: int,
    *,
    f0_hz: float = 12.0,
    f1_hz: float = 28.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(int(fs_hz * duration_s), dtype=np.float64) / fs_hz
    # Linear chirp shaft frequency f_r(t) = f0 + (f1 - f0) * t / T
    revs_true = f0_hz * t + 0.5 * (f1_hz - f0_hz) * (t**2) / duration_s
    rpm_true = 60.0 * (f0_hz + (f1_hz - f0_hz) * t / duration_s)
    # 0-4.3V TTL square wave with slight analog noise
    phase = (revs_true * ppr) % 1.0
    rng = np.random.default_rng(7)
    ttl = np.where(phase < 0.5, 4.3, 0.04) + rng.normal(0.0, 0.02, size=len(t))
    return t, ttl, rpm_true


def test_decode_encoder_tracks_variable_rpm_chirp() -> None:
    fs_hz = 100_000.0
    ppr = 256
    t, ttl, rpm_true = _synth_ttl_encoder(fs_hz, duration_s=2.0, ppr=ppr)
    track = decode_encoder(ttl, fs_hz, ppr=ppr)

    # Interior check (skip small startup edge margin)
    mask = (t >= 0.05) & (t <= 1.95)
    rpm_est = track.rpm_at(t[mask])
    rel_err = np.abs(rpm_est - rpm_true[mask]) / rpm_true[mask]
    assert float(np.median(rel_err)) < 0.003
    assert float(np.max(rel_err)) < 0.01


def test_audit_sensors_flags_dead_clipped_and_lost_encoder() -> None:
    fs_hz = 50_000.0
    _, ttl, _ = _synth_ttl_encoder(fs_hz, duration_s=1.0, ppr=128)
    track = decode_encoder(ttl, fs_hz, ppr=128)
    rng = np.random.default_rng(11)
    good_vib = rng.normal(0.0, 0.05, size=len(ttl))

    ok_audit = audit_sensors(good_vib, track)
    assert ok_audit.ok
    assert ok_audit.issues == ()

    flat_vib = np.zeros_like(good_vib)
    assert not audit_sensors(flat_vib, track).ok

    clipped_vib = np.clip(good_vib * 100.0, -5.0, 5.0)
    clip_audit = audit_sensors(clipped_vib, track, rail_limit=5.0)
    assert not clip_audit.ok
    assert "vibration_rail_clipping" in clip_audit.issues

    dead_track = decode_encoder(np.zeros_like(ttl), fs_hz, ppr=128)
    lost_audit = audit_sensors(good_vib, dead_track)
    assert not lost_audit.ok
    assert "encoder_signal_lost" in lost_audit.issues
