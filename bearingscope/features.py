"""Signal features. Mirrored line-for-line in web/src/lib/features.ts.

Three groups, 16 numbers per window:
  * 5 time-domain statistics (impulsiveness, shape)
  * 8 spectral band energies (where the vibration energy lives)
  * 3 envelope-spectrum scores, one per defect frequency (the physics)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .physics import SAMPLE_RATE_HZ, WINDOW, fault_frequencies

FloatArray = NDArray[np.float64]

N_BANDS = 8
ENVELOPE_BAND_HZ = (1500.0, 5500.0)  # structural resonance band excited by impacts
HARMONICS = (1, 2, 3)
PEAK_TOLERANCE_BINS = 1
RING_BINS = 8
EPS = 1e-12

FEATURE_NAMES: tuple[str, ...] = (
    "log_rms",
    "kurtosis",
    "crest_factor",
    "skewness",
    "shape_factor",
    *(f"band_{i}" for i in range(N_BANDS)),
    "env_bpfi",
    "env_bpfo",
    "env_bsf",
)


@dataclass(frozen=True)
class Envelope:
    freqs: FloatArray  # Hz, length WINDOW // 2
    spectrum: FloatArray  # envelope spectrum magnitude


def _check(x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (WINDOW,):
        raise ValueError(f"expected a window of {WINDOW} samples, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("window contains NaN or inf")
    return x - x.mean()


def envelope_spectrum(window: FloatArray) -> Envelope:
    """Band-pass around the resonance, demodulate, and take the spectrum of the envelope."""
    x = _check(window)
    n = WINDOW
    spec = np.fft.fft(x)
    lo = round(ENVELOPE_BAND_HZ[0] / SAMPLE_RATE_HZ * n)
    hi = round(ENVELOPE_BAND_HZ[1] / SAMPLE_RATE_HZ * n)
    analytic = np.zeros(n, dtype=np.complex128)
    analytic[lo : hi + 1] = 2.0 * spec[lo : hi + 1]
    env = np.abs(np.fft.ifft(analytic))
    env = env - env.mean()
    mag = np.abs(np.fft.fft(env))[: n // 2]
    freqs = np.arange(n // 2, dtype=np.float64) * (SAMPLE_RATE_HZ / n)
    return Envelope(freqs=freqs, spectrum=mag)


def _peak_score(env: FloatArray, freq_hz: float) -> float:
    """How far the envelope peaks at a defect frequency (and harmonics) rise above
    their immediate neighbourhood. Sum of log10 ratios over 3 harmonics; ~0 = no peak."""
    df = SAMPLE_RATE_HZ / WINDOW
    total = 0.0
    for h in HARMONICS:
        k = round(h * freq_hz / df)
        core = float(np.max(env[k - PEAK_TOLERANCE_BINS : k + PEAK_TOLERANCE_BINS + 1]))
        ring = np.concatenate(
            (
                env[k - PEAK_TOLERANCE_BINS - RING_BINS : k - PEAK_TOLERANCE_BINS],
                env[k + PEAK_TOLERANCE_BINS + 1 : k + PEAK_TOLERANCE_BINS + 1 + RING_BINS],
            )
        )
        total += float(np.log10(core / (float(np.median(ring)) + EPS) + EPS))
    return total


def extract(window: FloatArray, rpm: float) -> FloatArray:
    x = _check(window)
    n = WINDOW
    rms = float(np.sqrt(np.mean(x**2))) + EPS
    time_feats = [
        np.log10(rms),
        float(np.mean(x**4)) / rms**4,
        float(np.max(np.abs(x))) / rms,
        float(np.mean(x**3)) / rms**3,
        rms / (float(np.mean(np.abs(x))) + EPS),
    ]

    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)
    power = np.abs(np.fft.fft(x * hann))[: n // 2] ** 2
    total = float(power.sum()) + EPS
    width = (n // 2) // N_BANDS
    bands = [
        float(np.log10(power[i * width : (i + 1) * width].sum() / total + EPS))
        for i in range(N_BANDS)
    ]

    env = envelope_spectrum(x).spectrum
    ff = fault_frequencies(rpm)
    env_feats = [_peak_score(env, f) for f in (ff.bpfi, ff.bpfo, ff.bsf)]

    return np.array(time_feats + bands + env_feats, dtype=np.float64)


def extract_many(windows: FloatArray, rpms: FloatArray) -> FloatArray:
    return np.stack([extract(w, r) for w, r in zip(windows, rpms, strict=True)])
