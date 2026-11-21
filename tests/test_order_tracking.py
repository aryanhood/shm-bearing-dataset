"""Verify envelope-first Computed Order Tracking recovers exact fault orders on chirps."""

from __future__ import annotations

import numpy as np

from orderguard.encoder import decode_encoder
from orderguard.features import (
    extract_multiband_cnn_input,
    extract_multiband_order_physics,
    extract_order_physics_features,
)
from orderguard.kinematics import ER16K
from orderguard.order_tracking import (
    CARRIER_BANDS_HZ,
    N_ORDER_BINS,
    multiband_order_envelopes,
    order_envelope_spectrum,
    resample_to_angle,
    slice_order_windows,
    squared_envelope_time,
    time_envelope_fft,
)


def _synth_variable_speed_fault(
    *,
    fs_hz: float = 150_000.0,
    duration_s: float = 2.0,
    ppr: int = 256,
    fault_order: float = ER16K.orders().bpfi,
) -> tuple[np.ndarray, np.ndarray, float]:
    t = np.arange(int(fs_hz * duration_s), dtype=np.float64) / fs_hz
    f0, f1 = 12.0, 26.0
    revs = f0 * t + 0.5 * (f1 - f0) * (t**2) / duration_s

    ttl = np.where(((revs * ppr) % 1.0) < 0.5, 4.3, 0.04)

    impact_phase = (revs * fault_order) % 1.0
    pulse = np.exp(-impact_phase * 25.0)
    carrier = np.sin(2.0 * np.pi * 3000.0 * t) + 0.5 * np.sin(2.0 * np.pi * 10000.0 * t)
    shaft_gear_tone = 0.6 * np.sin(2.0 * np.pi * revs) + 0.4 * np.sin(2.0 * np.pi * 18.0 * revs)
    rng = np.random.default_rng(123)
    vib = pulse * carrier + shaft_gear_tone + rng.normal(0.0, 0.15, size=len(t))
    return vib, ttl, fs_hz


def test_cot_sharpens_smeared_variable_speed_fault_to_exact_bpfi() -> None:
    vib, ttl, fs_hz = _synth_variable_speed_fault()
    track = decode_encoder(ttl, fs_hz, ppr=256)
    env_t = squared_envelope_time(vib, fs_hz, bandpass_hz=(500.0, 5000.0), lowpass_env_hz=600.0)

    freqs, time_spec = time_envelope_fft(env_t, fs_hz, max_hz=300.0)
    t_mask = freqs >= 10.0
    t_vals = time_spec[t_mask]
    time_peak_ratio = float(np.max(t_vals) / (float(np.median(t_vals)) + 1e-12))

    order_sig = resample_to_angle(env_t, fs_hz, track)
    windows, _, rpms = slice_order_windows(order_sig, window_revs=16)
    assert len(windows) >= 1
    assert rpms[0] > 700.0

    orders, spec = order_envelope_spectrum(windows[0])
    o_mask = (orders >= 0.5) & (orders <= 20.0)
    o_vals = spec[o_mask]
    peak_order = float(orders[o_mask][int(np.argmax(o_vals))])
    order_peak_ratio = float(np.max(o_vals) / (float(np.median(o_vals)) + 1e-12))

    assert abs(peak_order - ER16K.orders().bpfi) <= 0.15
    assert order_peak_ratio > 2.0 * time_peak_ratio


def test_multiband_features_shape_and_amplitude_invariance() -> None:
    vib, ttl, fs_hz = _synth_variable_speed_fault()
    track = decode_encoder(ttl, fs_hz, ppr=256)
    _, order_sigs = multiband_order_envelopes(vib, fs_hz, track)
    band_wins = [slice_order_windows(osig, window_revs=16)[0][0] for osig in order_sigs]

    f1 = extract_order_physics_features(band_wins[0])
    assert f1.shape == (33,)

    mf1 = extract_multiband_order_physics(band_wins)
    mf2 = extract_multiband_order_physics([w * 17.5 for w in band_wins])
    assert mf1.shape == (33 * len(CARRIER_BANDS_HZ),)
    assert np.allclose(mf1, mf2, atol=1e-6)

    c1 = extract_multiband_cnn_input(band_wins)
    c2 = extract_multiband_cnn_input([w * 17.5 for w in band_wins])
    assert c1.shape == (len(CARRIER_BANDS_HZ), N_ORDER_BINS)
    assert np.allclose(c1, c2, atol=1e-5)
