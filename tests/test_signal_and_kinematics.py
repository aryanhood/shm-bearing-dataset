"""Tests for bearing kinematics, encoder decoding, degradation, and COT signal ops."""

from __future__ import annotations

import math

import numpy as np

from bearingbench.data.encoder import audit_sensors, decode_encoder, degrade_encoder_track
from bearingbench.data.kinematics import ER16K, SKF_6205, get_bearing, sideband_orders
from bearingbench.data.signal_ops import (
    N_ORDER_BINS,
    cepstrum_prewhiten,
    extract_cwru_gain_invariant_physics,
    extract_cwru_gain_sensitive_16,
    extract_cwru_multiband_cnn_input,
    extract_multiband_cnn_input,
    extract_multiband_order_physics,
    extract_order_cnn_channel,
    extract_time_cnn_input,
    multiband_order_envelopes,
    order_envelope_spectrum,
    order_first_cot_envelopes,
    slice_order_windows,
)


def _synth_variable_speed(
    *,
    fs_hz: float = 50_000.0,
    duration_s: float = 2.0,
    rpm_start: float = 900.0,
    rpm_end: float = 1800.0,
    ppr: int = 256,
    fault_order: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    t = np.arange(int(fs_hz * duration_s), dtype=np.float64) / fs_hz
    f0 = rpm_start / 60.0
    f1 = rpm_end / 60.0
    k = (f1 - f0) / duration_s
    theta = f0 * t + 0.5 * k * (t**2)
    encoder = np.where(np.sin(2.0 * np.pi * ppr * theta) >= 0.0, 5.0, 0.0)
    rng = np.random.default_rng(7)
    vib = 0.05 * rng.normal(size=len(t))
    if fault_order is not None:
        impulses = np.maximum(0.0, np.cos(2.0 * np.pi * fault_order * theta)) ** 8
        vib += impulses * np.sin(2.0 * np.pi * 6500.0 * t)
    return vib, encoder, fs_hz


def test_bearing_kinematics_orders() -> None:
    o = ER16K.orders()
    assert math.isclose(o.bpfo + o.bpfi, float(ER16K.n_balls), rel_tol=1e-9)
    assert o.bpfi > o.bpfo
    assert o.bsf_2x == 2.0 * o.bsf
    sb = sideband_orders(o)
    assert sb["bpfi_sidebands"][0] < o.bpfi < sb["bpfi_sidebands"][1]
    assert get_bearing("6205").code == SKF_6205.code


def test_encoder_decoding_degradation_and_cot_locking() -> None:
    bpfo = ER16K.orders().bpfo
    vib, enc, fs = _synth_variable_speed(fault_order=bpfo, ppr=256)
    track = decode_encoder(enc, fs, ppr=256)
    audit = audit_sensors(vib, track)
    assert audit.ok
    assert 800.0 < track.rpm_min < track.rpm_max < 1900.0

    deg = degrade_encoder_track(track, target_ppr=64, jitter_fraction=0.01, seed=42)
    assert deg.ppr == 64
    assert len(deg.edge_times_s) < len(track.edge_times_s)

    time_envs, osigs = multiband_order_envelopes(
        vib, fs, track, bands_hz=((1000.0, 4000.0), (4500.0, 9500.0))
    )
    wins, _, _ = slice_order_windows(osigs[1], window_revs=16)
    orders, spec = order_envelope_spectrum(wins[0])
    peak_order = float(orders[int(np.argmax(spec[orders >= 0.5])) + int(np.sum(orders < 0.5))])
    assert abs(peak_order - bpfo) <= 0.25

    t_cnn = extract_time_cnn_input(time_envs, fs, len(wins))
    assert t_cnn.shape == (len(wins), 2, N_ORDER_BINS)

    of_sigs = order_first_cot_envelopes(
        vib, fs, track, bands_hz=((1000.0, 4000.0),), raw_angle_spr=2048
    )
    assert len(of_sigs) == 1


def test_cepstrum_and_feature_shapes() -> None:
    rng = np.random.default_rng(11)
    sig = rng.normal(size=2048)
    cpw = cepstrum_prewhiten(sig)
    assert cpw.shape == sig.shape
    assert abs(float(cpw.std()) - 1.0) < 1e-6

    w_theta = np.sin(np.linspace(0.0, 40.0, 16 * 128))
    ch = extract_order_cnn_channel(w_theta)
    assert ch.shape == (N_ORDER_BINS,)
    assert extract_multiband_cnn_input([w_theta] * 4).shape == (4, N_ORDER_BINS)
    assert extract_multiband_order_physics([w_theta] * 4).shape == (132,)

    assert extract_cwru_gain_sensitive_16(sig, 1750.0).shape == (16,)
    assert extract_cwru_gain_invariant_physics(sig, 1750.0).shape == (27,)
    assert extract_cwru_multiband_cnn_input(sig).shape == (3, 256)
