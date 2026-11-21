"""Envelope-first Computed Order Tracking (COT) and Cepstrum Pre-whitening.

Implements the variable-speed bearing pipeline from Borghesani et al. (2013),
Randall & Antoni (2011), and Fyfe & Munck (1997):
  1. Bandpass + Hilbert squared-envelope demodulation runs in the TIME domain,
     across multiple structural carrier bands where housing resonances live.
  2. Zero-phase anti-aliasing low-pass filtering + cubic angular resampling maps
     each non-stationary envelope e(t) onto uniform shaft-angle increments e(theta).
  3. Order-domain Squared Envelope Spectrum (Order-SES) aligns all bearing fault
     impacts onto stationary shaft orders regardless of RPM ramps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, next_fast_len, rfft
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import butter, hilbert, sosfiltfilt

from .encoder import EncoderTrack

SAMPLES_PER_REV = 128
WINDOW_REVS = 16
MAX_ORDER = 25.0
N_ORDER_BINS = round(MAX_ORDER * WINDOW_REVS)  # 400 bins, delta_O = 0.0625 orders
DEFAULT_RESONANCE_BAND_HZ = (500.0, 5000.0)
CARRIER_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (500.0, 5000.0),
    (5000.0, 15_000.0),
    (15_000.0, 35_000.0),
    (35_000.0, 65_000.0),
)
DEFAULT_ENV_LOWPASS_HZ = 750.0


@dataclass(frozen=True)
class OrderSignal:
    """Angular-domain squared envelope resampled onto uniform shaft increments."""

    envelope_theta: NDArray[np.float64]
    theta_revs: NDArray[np.float64]
    rpm_theta: NDArray[np.float64]
    time_theta: NDArray[np.float64]
    samples_per_rev: int


def cepstrum_prewhiten(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Parameter-free Cepstrum Pre-whitening (Borghesani et al., 2013)."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    x = x - x.mean()
    nfft = next_fast_len(len(x))
    spec = rfft(x, n=nfft)
    cpw = irfft(spec / (np.abs(spec) + 1e-12), n=nfft)[: len(x)]
    std = float(cpw.std())
    return np.asarray(cpw / std if std > 1e-12 else cpw, dtype=np.float64)


def squared_envelope_time(
    signal: NDArray[np.float64],
    fs_hz: float,
    *,
    bandpass_hz: tuple[float, float] | None = DEFAULT_RESONANCE_BAND_HZ,
    lowpass_env_hz: float | None = DEFAULT_ENV_LOWPASS_HZ,
    use_cpw: bool = False,
) -> NDArray[np.float64]:
    """Compute band-limited squared envelope in the time domain."""
    x = (
        cepstrum_prewhiten(signal)
        if use_cpw
        else (np.asarray(signal, dtype=np.float64) - np.mean(signal))
    )
    x = np.asarray(x, dtype=np.float64).ravel()
    nyq = 0.5 * fs_hz
    if bandpass_hz is not None:
        hi = min(0.90 * nyq, bandpass_hz[1])
        lo = min(max(10.0, bandpass_hz[0]), 0.85 * hi)
        if lo < hi:
            sos_bp = butter(4, [lo, hi], btype="bandpass", fs=fs_hz, output="sos")
            x = np.asarray(sosfiltfilt(sos_bp, x), dtype=np.float64)

    nfft = next_fast_len(len(x))
    analytic = hilbert(x, N=nfft)[: len(x)]
    env_sq = np.abs(analytic) ** 2
    env_sq = env_sq - env_sq.mean()

    if lowpass_env_hz is not None and lowpass_env_hz < 0.9 * nyq:
        sos_lp = butter(4, lowpass_env_hz, btype="lowpass", fs=fs_hz, output="sos")
        env_sq = sosfiltfilt(sos_lp, env_sq)
    return np.asarray(env_sq, dtype=np.float64)


def _angle_time_grid(
    track: EncoderTrack,
    n_samples: int,
    fs_hz: float,
    samples_per_rev: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if len(track.edge_times_s) < 4:
        raise ValueError("encoder track needs at least 4 edges for angular resampling")

    theta_start = float(track.edge_revs[0]) + 0.05
    theta_end = float(track.edge_revs[-1]) - 0.05
    if theta_end <= theta_start:
        raise ValueError("encoder track spans fewer than 0.1 revolutions")

    theta_grid = np.arange(theta_start, theta_end, 1.0 / float(samples_per_rev), dtype=np.float64)
    time_of_rev = PchipInterpolator(track.edge_revs, track.edge_times_s)
    t_theta = np.asarray(time_of_rev(theta_grid), dtype=np.float64)

    t_max_sig = float(n_samples - 1) / fs_hz
    valid = (t_theta >= 0.0) & (t_theta <= t_max_sig)
    theta_grid = theta_grid[valid]
    t_theta = t_theta[valid]
    rpm_theta = track.rpm_at(t_theta)
    return theta_grid, t_theta, rpm_theta


def resample_to_angle(
    signal_t: NDArray[np.float64],
    fs_hz: float,
    track: EncoderTrack,
    *,
    samples_per_rev: int = SAMPLES_PER_REV,
) -> OrderSignal:
    """Resample a time-domain signal onto uniform shaft-angle steps via COT."""
    theta_grid, t_theta, rpm_theta = _angle_time_grid(track, len(signal_t), fs_hz, samples_per_rev)
    t_axis = np.arange(len(signal_t), dtype=np.float64) / fs_hz
    spline = CubicSpline(t_axis, signal_t, extrapolate=False)
    env_theta = np.asarray(spline(t_theta), dtype=np.float64)

    return OrderSignal(
        envelope_theta=env_theta,
        theta_revs=theta_grid,
        rpm_theta=rpm_theta,
        time_theta=t_theta,
        samples_per_rev=samples_per_rev,
    )


def multiband_order_envelopes(
    vibration: NDArray[np.float64],
    fs_hz: float,
    track: EncoderTrack,
    *,
    bands_hz: Sequence[tuple[float, float]] = CARRIER_BANDS_HZ,
    samples_per_rev: int = SAMPLES_PER_REV,
    lowpass_env_hz: float = DEFAULT_ENV_LOWPASS_HZ,
) -> tuple[list[NDArray[np.float64]], list[OrderSignal]]:
    """Demodulate multiple structural carrier bands and resample all onto angle."""
    vib = np.asarray(vibration, dtype=np.float64).ravel()
    theta_grid, t_theta, rpm_theta = _angle_time_grid(track, len(vib), fs_hz, samples_per_rev)
    t_axis = np.arange(len(vib), dtype=np.float64) / fs_hz

    time_envs: list[NDArray[np.float64]] = []
    order_sigs: list[OrderSignal] = []
    for band in bands_hz:
        env_t = squared_envelope_time(
            vib,
            fs_hz,
            bandpass_hz=band,
            lowpass_env_hz=lowpass_env_hz,
            use_cpw=False,
        )
        time_envs.append(env_t)
        env_theta = np.asarray(
            CubicSpline(t_axis, env_t, extrapolate=False)(t_theta), dtype=np.float64
        )
        order_sigs.append(
            OrderSignal(
                envelope_theta=env_theta,
                theta_revs=theta_grid,
                rpm_theta=rpm_theta,
                time_theta=t_theta,
                samples_per_rev=samples_per_rev,
            )
        )
    return time_envs, order_sigs


def slice_order_windows(
    order_sig: OrderSignal,
    *,
    window_revs: int = WINDOW_REVS,
    hop_revs: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Slice an angular-domain signal into fixed-revolution windows.

    Returns (windows, center_time_s, mean_rpm) per window.
    """
    win_len = window_revs * order_sig.samples_per_rev
    hop_len = (hop_revs if hop_revs is not None else window_revs) * order_sig.samples_per_rev
    n_total = len(order_sig.envelope_theta)
    if n_total < win_len:
        raise ValueError(f"recording has {n_total} angular samples, need at least {win_len}")

    starts = range(0, n_total - win_len + 1, hop_len)
    windows = np.stack([order_sig.envelope_theta[s : s + win_len] for s in starts])
    times = np.array(
        [0.5 * (order_sig.time_theta[s] + order_sig.time_theta[s + win_len - 1]) for s in starts],
        dtype=np.float64,
    )
    rpms = np.array(
        [float(order_sig.rpm_theta[s : s + win_len].mean()) for s in starts],
        dtype=np.float64,
    )
    return windows, times, rpms


def order_envelope_spectrum(
    window_theta: NDArray[np.float64],
    *,
    samples_per_rev: int = SAMPLES_PER_REV,
    max_order: float = MAX_ORDER,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the Order-Domain Squared Envelope Spectrum (Order-SES)."""
    w = np.asarray(window_theta, dtype=np.float64).ravel()
    n = len(w)
    w = w - w.mean()
    hann = np.hanning(n)
    spec = np.abs(rfft(w * hann)) / (float(hann.sum()) + 1e-12)
    window_revs = float(n) / float(samples_per_rev)
    orders = np.arange(len(spec), dtype=np.float64) / window_revs
    n_bins = round(max_order * window_revs)
    return orders[:n_bins], spec[:n_bins]


def time_envelope_fft(
    window_t: NDArray[np.float64],
    fs_hz: float,
    *,
    max_hz: float = 600.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute a fixed-time Squared Envelope Spectrum (without order tracking)."""
    w = np.asarray(window_t, dtype=np.float64).ravel()
    n = len(w)
    w = w - w.mean()
    hann = np.hanning(n)
    spec = np.abs(rfft(w * hann)) / (float(hann.sum()) + 1e-12)
    freqs = np.arange(len(spec), dtype=np.float64) * (fs_hz / float(n))
    mask = freqs <= max_hz
    return freqs[mask], spec[mask]
