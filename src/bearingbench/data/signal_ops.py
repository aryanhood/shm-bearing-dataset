"""Signal processing operations for Envelope-First COT, Order-First ablation, and CWRU/uOttawa features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, next_fast_len, rfft
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import butter, hilbert, sosfiltfilt

from .encoder import EncoderTrack
from .kinematics import ER16K, SKF_6205, BearingSpec, sideband_orders

EPS = 1e-12

# uOttawa (200 kHz) parameters
SAMPLES_PER_REV = 128
WINDOW_REVS = 16
MAX_ORDER = 25.0
N_ORDER_BINS = round(MAX_ORDER * WINDOW_REVS)  # 400 bins, delta_O = 0.0625
CARRIER_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (500.0, 5000.0),
    (5000.0, 15_000.0),
    (15_000.0, 35_000.0),
    (35_000.0, 65_000.0),
)
DEFAULT_ENV_LOWPASS_HZ = 750.0
RAW_ANGLE_SPR = 8192  # Oversampled angular rate for Order-First COT ablation
TIME_ENV_BANDS = 24
ORDER_BANDS = 8
HARMONICS = (1, 2, 3)

# CWRU (12 kHz) parameters
CWRU_FS_HZ = 12_000.0
CWRU_CARRIER_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (500.0, 1500.0),
    (1500.0, 3500.0),
    (3500.0, 5500.0),
)
CWRU_CNN_BINS = 256
CWRU_CNN_MAX_HZ = 375.0


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
    cpw = irfft(spec / (np.abs(spec) + EPS), n=nfft)[: len(x)]
    std = float(cpw.std())
    return np.asarray(cpw / std if std > EPS else cpw, dtype=np.float64)


def squared_envelope_time(
    signal: NDArray[np.float64],
    fs_hz: float,
    *,
    bandpass_hz: tuple[float, float] | None = CARRIER_BANDS_HZ[0],
    lowpass_env_hz: float | None = DEFAULT_ENV_LOWPASS_HZ,
    use_cpw: bool = False,
) -> NDArray[np.float64]:
    """Compute band-limited squared envelope in the time domain."""
    x = cepstrum_prewhiten(signal) if use_cpw else (np.asarray(signal, dtype=np.float64) - np.mean(signal))
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


def angle_time_grid(
    track: EncoderTrack,
    n_samples: int,
    fs_hz: float,
    samples_per_rev: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build monotonic uniform shaft-angle grid and interpolated timestamps."""
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
    theta_grid, t_theta, rpm_theta = angle_time_grid(track, len(signal_t), fs_hz, samples_per_rev)
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
    use_cpw: bool = False,
) -> tuple[list[NDArray[np.float64]], list[OrderSignal]]:
    """Envelope-First COT: demodulate structural carrier bands in time, then resample."""
    vib = np.asarray(vibration, dtype=np.float64).ravel()
    base_sig = cepstrum_prewhiten(vib) if use_cpw else (vib - float(np.mean(vib)))
    theta_grid, t_theta, rpm_theta = angle_time_grid(track, len(vib), fs_hz, samples_per_rev)
    t_axis = np.arange(len(vib), dtype=np.float64) / fs_hz

    time_envs: list[NDArray[np.float64]] = []
    order_sigs: list[OrderSignal] = []
    for band in bands_hz:
        # Bandpass + Hilbert on pre-whitened/mean-centered time waveform
        env_t = squared_envelope_time(
            base_sig,
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


def order_first_cot_envelopes(
    vibration: NDArray[np.float64],
    fs_hz: float,
    track: EncoderTrack,
    *,
    bands_hz: Sequence[tuple[float, float]] = CARRIER_BANDS_HZ,
    samples_per_rev: int = SAMPLES_PER_REV,
    raw_angle_spr: int = RAW_ANGLE_SPR,
    lowpass_env_hz: float = DEFAULT_ENV_LOWPASS_HZ,
) -> list[OrderSignal]:
    """Order-First COT ablation: angularly resample raw signal BEFORE bandpass & Hilbert.

    Resampling raw vibration onto angle prior to demodulation smears fixed-Hz
    structural housing resonances proportional to instantaneous shaft speed f_s(t),
    degrading envelope modulation SNR under rapid non-stationary RPM ramps.
    """
    vib = np.asarray(vibration, dtype=np.float64).ravel() - float(np.mean(vibration))
    theta_fine, t_fine, _ = angle_time_grid(track, len(vib), fs_hz, raw_angle_spr)
    t_axis = np.arange(len(vib), dtype=np.float64) / fs_hz
    vib_theta = np.asarray(CubicSpline(t_axis, vib, extrapolate=False)(t_fine), dtype=np.float64)

    mean_shaft_hz = max(track.rpm_mean / 60.0, 1.0)
    fs_order = float(raw_angle_spr)  # samples per shaft revolution
    nyq_order = 0.5 * fs_order

    theta_out, t_out, rpm_out = angle_time_grid(track, len(vib), fs_hz, samples_per_rev)
    order_sigs: list[OrderSignal] = []

    for lo_hz, hi_hz in bands_hz:
        lo_ord = max(1.0, lo_hz / mean_shaft_hz)
        hi_ord = min(0.90 * nyq_order, hi_hz / mean_shaft_hz)
        x_band = vib_theta
        if lo_ord < hi_ord:
            sos_bp = butter(4, [lo_ord, hi_ord], btype="bandpass", fs=fs_order, output="sos")
            x_band = np.asarray(sosfiltfilt(sos_bp, vib_theta), dtype=np.float64)
        nfft = next_fast_len(len(x_band))
        analytic = hilbert(x_band, N=nfft)[: len(x_band)]
        env_sq = np.abs(analytic) ** 2
        env_sq = env_sq - env_sq.mean()
        lp_ord = min(0.85 * nyq_order, lowpass_env_hz / mean_shaft_hz)
        if lp_ord > 1.0:
            sos_lp = butter(4, lp_ord, btype="lowpass", fs=fs_order, output="sos")
            env_sq = np.asarray(sosfiltfilt(sos_lp, env_sq), dtype=np.float64)
        env_out = np.asarray(
            np.interp(theta_out, theta_fine, env_sq),
            dtype=np.float64,
        )
        order_sigs.append(
            OrderSignal(
                envelope_theta=env_out,
                theta_revs=theta_out,
                rpm_theta=rpm_out,
                time_theta=t_out,
                samples_per_rev=samples_per_rev,
            )
        )
    return order_sigs


def slice_order_windows(
    order_sig: OrderSignal,
    *,
    window_revs: int = WINDOW_REVS,
    hop_revs: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Slice an angular-domain signal into fixed-revolution windows."""
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
    spec = np.abs(rfft(w * hann)) / (float(hann.sum()) + EPS)
    window_revs = float(n) / float(samples_per_rev)
    orders = np.arange(len(spec), dtype=np.float64) / window_revs
    n_bins = round(max_order * window_revs)
    return orders[:n_bins], spec[:n_bins]


def time_envelope_fft(
    window_t: NDArray[np.float64],
    fs_hz: float,
    *,
    max_hz: float = 600.0,
    nfft: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute a fixed-time Squared Envelope Spectrum (without order tracking)."""
    w = np.asarray(window_t, dtype=np.float64).ravel()
    w = w - w.mean()
    hann = np.hanning(len(w))
    fft_n = len(w) if nfft is None else nfft
    spec = np.abs(rfft(w * hann, n=fft_n)) / (float(hann.sum()) + EPS)
    freqs = np.arange(len(spec), dtype=np.float64) * (fs_hz / float(fft_n))
    mask = freqs <= max_hz
    return freqs[mask], spec[mask]


def _peak_ring_snr(
    axis: NDArray[np.float64],
    spec: NDArray[np.float64],
    target: float,
    *,
    tol: float = 0.14,
    ring: float = 0.65,
    min_axis: float = 0.25,
) -> float:
    """Amplitude-invariant local SNR: log10(peak / local median ring)."""
    core_mask = (axis >= target - tol) & (axis <= target + tol)
    if not np.any(core_mask):
        return 0.0
    core = float(np.max(spec[core_mask]))
    ring_mask = ((axis >= max(min_axis, target - tol - ring)) & (axis < target - tol)) | (
        (axis > target + tol) & (axis <= target + tol + ring)
    )
    if not np.any(ring_mask):
        return 0.0
    bg = float(np.median(spec[ring_mask]))
    return float(np.log10((core + EPS) / (bg + EPS)))


def extract_order_physics_features(
    window_theta: NDArray[np.float64],
    *,
    samples_per_rev: int = SAMPLES_PER_REV,
    max_order: float = MAX_ORDER,
    geometry: BearingSpec = ER16K,
) -> NDArray[np.float64]:
    """Extract 33 amplitude-invariant kinematic & Order-SES features for one band."""
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
        feats.extend(_peak_ring_snr(orders, spec, float(h) * base) for h in HARMONICS)
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
    geometry: BearingSpec = ER16K,
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
    """Amplitude-invariant log1p(Order-SES / median_floor) vector (length 400)."""
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
    """Stack multi-band Order-SES channels into shape (C, 400)."""
    return np.stack(
        [
            extract_order_cnn_channel(w, samples_per_rev=samples_per_rev, max_order=max_order)
            for w in windows_per_band
        ],
        axis=0,
    )


def extract_time_cnn_input(
    time_envelopes: Sequence[NDArray[np.float64]],
    fs_hz: float,
    n_windows: int,
    *,
    n_bins: int = N_ORDER_BINS,
    max_hz: float = 600.0,
) -> NDArray[np.float32]:
    """Extract matched (N, C, 400) time-domain envelope spectra WITHOUT order tracking.

    Uses the exact same 4 carrier bands and log1p(S / median_floor) normalization as
    Order-SES, enabling an unconfounded representation ablation (Time-Env CNN vs Order-CNN).
    """
    n_samples = len(time_envelopes[0])
    seg_len = n_samples // n_windows
    grid_hz = np.linspace(0.0, max_hz, n_bins, dtype=np.float64)
    out = np.zeros((n_windows, len(time_envelopes), n_bins), dtype=np.float32)

    for w_idx in range(n_windows):
        s = w_idx * seg_len
        e = s + seg_len
        for b_idx, env in enumerate(time_envelopes):
            ef, esp = time_envelope_fft(env[s:e], fs_hz, max_hz=max_hz)
            interp_spec = np.interp(grid_hz, ef, esp)
            masked = np.where(grid_hz >= 5.0, interp_spec, 0.0)
            bg = float(np.median(masked[grid_hz >= 5.0])) + EPS
            out[w_idx, b_idx] = np.log1p(masked / bg).astype(np.float32)
    return out


def extract_multiband_time_env_baseline(
    time_envelopes: Sequence[NDArray[np.float64]],
    fs_hz: float,
    n_windows: int,
) -> NDArray[np.float64]:
    """Extract time-domain envelope FFT SNR features (96-d) WITHOUT encoder tracking."""
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


# ---- CWRU 12 kHz extractors (comparing gain-sensitive vs gain-invariant physics) ----


def extract_cwru_gain_sensitive_16(window: NDArray[np.float64], rpm: float) -> NDArray[np.float64]:
    """Reproduce BearingScope 16-d features (includes gain-sensitive log_rms & band PSDs)."""
    w = np.asarray(window, dtype=np.float64).ravel() - float(np.mean(window))
    rms = float(np.sqrt(np.mean(w**2))) + EPS
    kurt = float(np.mean(w**4)) / (rms**4)
    crest = float(np.max(np.abs(w))) / rms
    skew = float(np.mean(w**3)) / (rms**3)
    shape = rms / (float(np.mean(np.abs(w))) + EPS)

    hann = np.hanning(len(w))
    psd = np.abs(rfft(w * hann)) ** 2
    edges = np.linspace(0, len(psd), 9, dtype=int)
    bands = [float(np.log10(psd[edges[i] : edges[i + 1]].mean() + EPS)) for i in range(8)]

    sos = butter(4, [1500.0, 5500.0], btype="bandpass", fs=CWRU_FS_HZ, output="sos")
    bp = sosfiltfilt(sos, w)
    nfft = next_fast_len(len(bp))
    env = np.abs(hilbert(bp, N=nfft)[: len(bp)])
    env = env - env.mean()
    ef = np.arange(len(env) // 2 + 1, dtype=np.float64) * (CWRU_FS_HZ / float(len(env)))
    es = np.abs(rfft(env * hann))

    fhz = SKF_6205.frequencies_hz(rpm)
    faults = [
        _peak_ring_snr(ef, es, fhz["bpfi"], tol=5.0, ring=20.0, min_axis=5.0),
        _peak_ring_snr(ef, es, fhz["bpfo"], tol=5.0, ring=20.0, min_axis=5.0),
        _peak_ring_snr(ef, es, fhz["bsf"], tol=5.0, ring=20.0, min_axis=5.0),
    ]
    return np.asarray([float(np.log10(rms)), kurt, crest, skew, shape, *bands, *faults], dtype=np.float64)


def extract_cwru_gain_invariant_physics(
    window: NDArray[np.float64], rpm: float
) -> NDArray[np.float64]:
    """27-d strictly gain-invariant multi-band envelope harmonic SNRs & spectral shape."""
    w = np.asarray(window, dtype=np.float64).ravel() - float(np.mean(window))
    rms = float(np.sqrt(np.mean(w**2))) + EPS
    kurt = float(np.mean(w**4)) / (rms**4)
    crest = float(np.max(np.abs(w))) / rms
    shape = rms / (float(np.mean(np.abs(w))) + EPS)

    fhz = SKF_6205.frequencies_hz(rpm)
    feats: list[float] = [float(np.log10(max(kurt, EPS))), crest, shape]
    for band in CWRU_CARRIER_BANDS_HZ:
        env = squared_envelope_time(w, CWRU_FS_HZ, bandpass_hz=band, lowpass_env_hz=600.0)
        ef, es = time_envelope_fft(env, CWRU_FS_HZ, max_hz=CWRU_CNN_MAX_HZ, nfft=8192)
        scale = float(np.median(es[ef >= 5.0])) + EPS
        es_norm = es / scale
        for key in ("bpfi", "bpfo", "bsf_2x"):
            feats.append(_peak_ring_snr(ef, es_norm, fhz[key], tol=4.5, ring=18.0, min_axis=4.0))
            feats.append(_peak_ring_snr(ef, es_norm, 2.0 * fhz[key], tol=5.5, ring=20.0, min_axis=4.0))
        feats.append(_peak_ring_snr(ef, es_norm, fhz["shaft_1x"], tol=3.5, ring=12.0, min_axis=4.0))
        # Normalized band spectral entropy
        p = es_norm[ef >= 5.0] ** 2 + EPS
        p_norm = p / float(p.sum())
        ent = float(-np.sum(p_norm * np.log(p_norm)) / np.log(float(len(p_norm))))
        feats.append(ent)
    return np.asarray(feats, dtype=np.float64)


def extract_cwru_multiband_cnn_input(window: NDArray[np.float64]) -> NDArray[np.float32]:
    """Extract (3, 256) gain-invariant 3-band squared-envelope spectra for CWRU."""
    w = np.asarray(window, dtype=np.float64).ravel() - float(np.mean(window))
    grid_hz = np.linspace(0.0, CWRU_CNN_MAX_HZ, CWRU_CNN_BINS, dtype=np.float64)
    out = np.zeros((len(CWRU_CARRIER_BANDS_HZ), CWRU_CNN_BINS), dtype=np.float32)

    for b_idx, band in enumerate(CWRU_CARRIER_BANDS_HZ):
        env = squared_envelope_time(w, CWRU_FS_HZ, bandpass_hz=band, lowpass_env_hz=600.0)
        ef, es = time_envelope_fft(env, CWRU_FS_HZ, max_hz=CWRU_CNN_MAX_HZ, nfft=8192)
        interp_spec = np.interp(grid_hz, ef, es)
        masked = np.where(grid_hz >= 4.0, interp_spec, 0.0)
        bg = float(np.median(masked[grid_hz >= 4.0])) + EPS
        out[b_idx] = np.log1p(masked / bg).astype(np.float32)
    return out
