"""
Feature Extractor
=================
Converts raw 1-D bearing-vibration windows into engineered feature vectors
used by the Random Forest baseline.

Time-domain statistics
----------------------
  rms          = √( (1/N) Σ xᵢ² )
  kurtosis     = E[(x−μ)⁴] / σ⁴          — sensitive to impulsive faults
  skewness     = E[(x−μ)³] / σ³
  crest factor = |x|_max / rms            — peaks relative to energy
  peak-to-peak = max(x) − min(x)
  shape factor = rms / mean(|x|)
  impulse fac  = |x|_max / mean(|x|)
  ZCR          = zero-crossing rate

Frequency-domain
----------------
  First n_fft_bins of the one-sided amplitude spectrum (|rfft(x)|),
  normalised by the DC component, plus:
    spectral centroid   = Σ f·|X(f)| / Σ |X(f)|
    spectral entropy    = −Σ p·log(p)   (p = normalised power)
    dominant frequency  = argmax |X(f)|

Implementation note: we keep n_fft_bins=64 so the total feature vector
stays compact (≈ 75 features), which prevents RF from over-fitting on
the bin activations while still encoding spectral shape.
"""
from __future__ import annotations

from typing import List, Mapping, Optional, Tuple

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew


class FeatureExtractor:
    """
    Stateless feature extractor for 1-D vibration windows.

    Parameters
    ----------
    sampling_rate : int   Hz
    n_fft_bins    : int   number of spectrum bins to keep
    use_fft       : bool
    """

    def __init__(
        self,
        sampling_rate: int = 12000,
        n_fft_bins:    int = 64,
        use_fft:       bool = True,
        include_time:  bool = True,
        include_freq:  Optional[bool] = None,
    ) -> None:
        self.sr         = sampling_rate
        self.n_fft_bins = n_fft_bins
        self.include_time = include_time
        self.use_fft    = use_fft if include_freq is None else include_freq
        self._names: Optional[List[str]] = None

    @classmethod
    def from_config(
        cls,
        cfg: Mapping,
        *,
        sampling_rate: int | None = None,
    ) -> "FeatureExtractor":
        feature_cfg = cfg.get("features", {})
        freq_cfg = feature_cfg.get("freq_domain", {})
        return cls(
            sampling_rate=sampling_rate or int(cfg.get("data", {}).get("sampling_rate", 12000)),
            n_fft_bins=int(freq_cfg.get("n_fft_bins", 64)),
            include_time=bool(feature_cfg.get("time_domain", [True])),
            include_freq=bool(freq_cfg.get("enabled", True)),
        )

    # ── public ────────────────────────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (N, window_size)  float32

        Returns
        -------
        F : (N, n_features)   float32
        """
        rows = [self._extract_one(x) for x in X]
        return np.stack(rows).astype(np.float32)

    @property
    def feature_names(self) -> List[str]:
        if self._names is None:
            dummy = np.zeros(256)
            self._extract_one(dummy)  # populates _names
        return self._names  # type: ignore[return-value]

    def n_features(self, window_size: int = 1024) -> int:
        dummy = np.zeros(window_size)
        return len(self._extract_one(dummy))

    # ── internals ─────────────────────────────────────────────────────────────

    def _extract_one(self, x: np.ndarray) -> np.ndarray:
        parts: List[np.ndarray] = []
        names: List[str] = []

        if self.include_time:
            td, td_names = self._time_domain(x)
            parts.append(td)
            names += td_names

        if self.use_fft:
            fd, fd_names = self._freq_domain(x)
            parts.append(fd)
            names += fd_names

        if not parts:
            raise ValueError("FeatureExtractor requires at least one enabled feature group.")

        if self._names is None:
            self._names = names

        return np.concatenate(parts)

    # ── time domain ───────────────────────────────────────────────────────────

    @staticmethod
    def _time_domain(x: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        rms_val   = float(np.sqrt(np.mean(x ** 2)))
        abs_mean  = float(np.mean(np.abs(x))) + 1e-12
        max_abs   = float(np.max(np.abs(x)))

        feats = np.array([
            rms_val,                                        # rms
            float(kurtosis(x, fisher=True)),                # kurtosis
            float(skew(x)),                                 # skewness
            max_abs / (rms_val + 1e-12),                    # crest_factor
            float(np.ptp(x)),                               # peak_to_peak
            rms_val / abs_mean,                             # shape_factor
            max_abs / abs_mean,                             # impulse_factor
            float(np.mean(np.abs(np.diff(np.sign(x))))/2), # zcr
        ], dtype=np.float64)

        names = [
            "rms", "kurtosis", "skewness", "crest_factor",
            "peak_to_peak", "shape_factor", "impulse_factor", "zcr",
        ]
        return feats.astype(np.float32), names

    # ── frequency domain ──────────────────────────────────────────────────────

    def _freq_domain(self, x: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        N    = len(x)
        mags = np.abs(rfft(x))          # one-sided amplitude spectrum
        # normalise by DC to make it amplitude-independent
        mags = mags / (mags[0] + 1e-12)

        freqs = rfftfreq(N, d=1.0 / self.sr)
        n_bins = min(self.n_fft_bins, len(mags))
        bins   = mags[:n_bins]
        if len(bins) < self.n_fft_bins:
            bins = np.pad(bins, (0, self.n_fft_bins - len(bins)))

        # aggregate descriptors
        power  = mags[:n_bins] ** 2 + 1e-12
        power /= power.sum()
        sp_centroid = float(np.dot(freqs[:n_bins], power))
        sp_entropy  = float(-np.sum(power * np.log(power)))
        dom_freq    = float(freqs[np.argmax(mags[:n_bins])])

        extras = np.array([sp_centroid, sp_entropy, dom_freq], dtype=np.float32)
        feats  = np.concatenate([bins.astype(np.float32), extras])

        bin_names   = [f"fft_{i}" for i in range(self.n_fft_bins)]
        extra_names = ["spectral_centroid", "spectral_entropy", "dominant_freq"]
        return feats, bin_names + extra_names
