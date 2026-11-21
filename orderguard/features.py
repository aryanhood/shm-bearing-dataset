"""Feature extraction for fair Time-Env FFT vs Order-SES benchmarking.

All three pipelines consume the exact same 4 demodulated structural carrier bands
(0.5-5, 5-15, 15-35, 35-65 kHz) and use strictly amplitude-invariant peak-to-floor
SNR representations so classifiers cannot take raw-gain or mount-FFT shortcuts:
  1. Time-domain envelope baseline (no encoder): 24 peak-SNR frequency bins per
     carrier band across 5..600 Hz (96 features).
  2. Order-domain physics features (encoder COT): ER16K harmonic + load-zone
     sideband SNRs and order-envelope shape per carrier band (4 x 33 = 132).
  3. Multi-band Order-SES tensor (4, 400) for OrderCNN / ONNX edge inference.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .kinematics import ER16K, BearingGeometry, sideband_orders
from .order_tracking import (
    MAX_ORDER,
    N_ORDER_BINS,
    SAMPLES_PER_REV,
    order_envelope_spectrum,
    time_envelope_fft,
)

EPS = 1e-12
TIME_ENV_BANDS = 24
ORDER_BANDS = 8
HARMONICS = (1, 2, 3)


def _peak_ring_snr(
    axis: NDArray[np.float64],
    spec: NDArray[np.float64],
    target_order: float,
    *,
    tol_orders: float = 0.14,
    ring_orders: float = 0.65,
) -> float:
    """Amplitude-invariant local SNR: log10(peak / local median ring) around target_order."""
    core_mask = (axis >= target_order - tol_orders) & (axis <= target_order + tol_orders)
    if not np.any(core_mask):
        return 0.0
    core = float(np.max(spec[core_mask]))
    ring_mask = (
        (axis >= max(0.25, target_order - tol_orders - ring_orders))
        & (axis < target_order - tol_orders)
    ) | ((axis > target_order + tol_orders) & (axis <= target_order + tol_orders + ring_orders))
    if not np.any(ring_mask):
        return 0.0
    bg = float(np.median(spec[ring_mask]))
    return float(np.log10((core + EPS) / (bg + EPS)))


def _harmonic_snr(
    axis: NDArray[np.float64],
    spec: NDArray[np.float64],
    base_order: float,
) -> list[float]:
    return [_peak_ring_snr(axis, spec, float(h) * base_order) for h in HARMONICS]


def extract_order_physics_features(
    window_theta: NDArray[np.float64],
    *,
    samples_per_rev: int = SAMPLES_PER_REV,
    max_order: float = MAX_ORDER,
    geometry: BearingGeometry = ER16K,
) -> NDArray[np.float64]:
    """Extract 33 amplitude-invariant kinematic & spectral features for one band."""
    orders, spec = order_envelope_spectrum(
        window_theta, samples_per_rev=samples_per_rev, max_order=max_order
    )
    spec_scale = float(np.mean(spec[orders >= 0.25]))
    if spec_scale > 0.0:
        spec = spec / spec_scale
    faults = geometry.orders()
    sb = sideband_orders(faults)

    feats: list[float] = []
    for base in (faults.bpfi, faults.bpfo, faults.bsf, faults.bsf_2x):
        feats.extend(_harmonic_snr(orders, spec, base))
    feats.append(_peak_ring_snr(orders, spec, faults.ftf))
    feats.append(_peak_ring_snr(orders, spec, 2.0 * faults.ftf))

    for target in sb["bpfi_sidebands"] + sb["bsf_2x_sidebands"]:
        feats.append(_peak_ring_snr(orders, spec, target))

    feats.append(_peak_ring_snr(orders, spec, 1.0))
    feats.append(_peak_ring_snr(orders, spec, 2.0))

    w = window_theta - window_theta.mean()
    rms = float(np.sqrt(np.mean(w**2))) + EPS
    kurt = float(np.mean(w**4)) / (rms**4)
    crest = float(np.max(np.abs(w))) / rms
    skew = float(np.mean(w**3)) / (rms**3)
    feats.extend([float(np.log10(max(kurt, EPS))), crest, skew])

    valid = orders >= 0.25
    p = spec[valid] ** 2 + EPS
    p_norm = p / float(p.sum())
    geom_mean = float(np.exp(np.mean(np.log(p))))
    arith_mean = float(np.mean(p))
    flatness = float(np.log10((geom_mean + EPS) / (arith_mean + EPS)))
    entropy = float(-np.sum(p_norm * np.log(p_norm)) / np.log(float(len(p_norm))))
    feats.extend([flatness, entropy])

    edges = np.linspace(0.25, max_order, ORDER_BANDS + 1)
    total_p = float(p.sum()) + EPS
    for i in range(ORDER_BANDS):
        m = (orders >= edges[i]) & (orders < edges[i + 1])
        band_share = float((spec[m] ** 2).sum() / total_p) if np.any(m) else 0.0
        feats.append(float(np.log10(band_share + EPS)))

    return np.asarray(feats, dtype=np.float64)


def extract_multiband_order_physics(
    windows_per_band: Sequence[NDArray[np.float64]],
    *,
    samples_per_rev: int = SAMPLES_PER_REV,
    max_order: float = MAX_ORDER,
    geometry: BearingGeometry = ER16K,
) -> NDArray[np.float64]:
    """Concatenate kinematic & Order-SES features across all carrier bands."""
    return np.concatenate(
        [
            extract_order_physics_features(
                w, samples_per_rev=samples_per_rev, max_order=max_order, geometry=geometry
            )
            for w in windows_per_band
        ]
    )


def extract_order_cnn_channel(
    window_theta: NDArray[np.float64],
    *,
    samples_per_rev: int = SAMPLES_PER_REV,
    max_order: float = MAX_ORDER,
) -> NDArray[np.float32]:
    """Amplitude-invariant log1p(Order-SES / median_floor) vector (length 400).

    Unlike per-window z-scoring, normalizing strictly by the median noise floor
    preserves peak prominence above baseline while remaining gain-invariant.
    """
    orders, spec = order_envelope_spectrum(
        window_theta, samples_per_rev=samples_per_rev, max_order=max_order
    )
    spec_masked = np.where(orders >= 0.25, spec, 0.0)
    bg = float(np.median(spec_masked[orders >= 0.25])) + EPS
    out = np.log1p(spec_masked / bg).astype(np.float32)
    if len(out) != N_ORDER_BINS:
        raise ValueError(f"expected {N_ORDER_BINS} order bins, got {len(out)}")
    return out


def extract_multiband_cnn_input(
    windows_per_band: Sequence[NDArray[np.float64]],
    *,
    samples_per_rev: int = SAMPLES_PER_REV,
    max_order: float = MAX_ORDER,
) -> NDArray[np.float32]:
    """Stack multi-band Order-SES channels into shape (C, 400) for OrderCNN."""
    return np.stack(
        [
            extract_order_cnn_channel(w, samples_per_rev=samples_per_rev, max_order=max_order)
            for w in windows_per_band
        ],
        axis=0,
    )


def extract_multiband_time_env_baseline(
    time_envelopes: Sequence[NDArray[np.float64]],
    fs_hz: float,
    n_windows: int,
) -> NDArray[np.float64]:
    """Extract time-domain envelope FFT SNR features WITHOUT encoder tracking."""
    n_samples = len(time_envelopes[0])
    seg_len = n_samples // n_windows
    edges = np.linspace(5.0, 600.0, TIME_ENV_BANDS + 1)
    rows: list[list[float]] = []

    for w_idx in range(n_windows):
        s = w_idx * seg_len
        e = s + seg_len
        feats: list[float] = []
        for env in time_envelopes:
            ef, esp = time_envelope_fft(env[s:e], fs_hz, max_hz=600.0)
            mask = ef >= 5.0
            bg = float(np.median(esp[mask])) + EPS
            snr = np.log1p(esp / bg)
            for b in range(TIME_ENV_BANDS):
                m = (ef >= edges[b]) & (ef < edges[b + 1])
                feats.append(float(np.max(snr[m])) if np.any(m) else 0.0)
        rows.append(feats)
    return np.asarray(rows, dtype=np.float64)
