"""Learn a machine's healthy baseline from one recording and measure drift over time."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .io import SignalData

EPS = 1e-12
N_BANDS = 8
FEATURE_NAMES: tuple[str, ...] = (
    "log_rms",
    "kurtosis",
    "crest_factor",
    "shape_factor",
    *(f"band_{i}" for i in range(N_BANDS)),
)


@dataclass(frozen=True)
class BaselineProfile:
    """Portable healthy-machine fingerprint (~2 KB JSON)."""

    machine_name: str
    fs_hz: float
    units: str
    n_windows: int
    healthy_rms: float
    healthy_kurtosis: float
    feature_median: tuple[float, ...]
    feature_scale: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "machine_name": self.machine_name,
            "fs_hz": self.fs_hz,
            "units": self.units,
            "n_windows": self.n_windows,
            "healthy_rms": round(self.healthy_rms, 6),
            "healthy_kurtosis": round(self.healthy_kurtosis, 3),
            "feature_names": list(FEATURE_NAMES),
            "feature_median": [round(v, 6) for v in self.feature_median],
            "feature_scale": [round(v, 6) for v in self.feature_scale],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> BaselineProfile:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            machine_name=str(d["machine_name"]),
            fs_hz=float(d["fs_hz"]),
            units=str(d["units"]),
            n_windows=int(d["n_windows"]),
            healthy_rms=float(d["healthy_rms"]),
            healthy_kurtosis=float(d["healthy_kurtosis"]),
            feature_median=tuple(float(v) for v in d["feature_median"]),
            feature_scale=tuple(float(v) for v in d["feature_scale"]),
        )


@dataclass(frozen=True)
class BaselineDrift:
    """Plain-English drift summary comparing a new recording to a healthy baseline."""

    machine_name: str
    drift_score: float  # 0 (identical to healthy) .. 100 (major anomaly)
    rms_ratio: float
    kurtosis_delta: float
    spectral_shift: float
    plain_reason: str


def _window_features(x: NDArray[np.float64], fs_hz: float) -> NDArray[np.float64]:
    """Slice signal into non-overlapping windows and extract 12 physical features each."""
    centered = x - float(np.mean(x))
    n_total = len(centered)
    # Aim for ~0.25 s windows, guaranteeing at least 6 windows on short clips.
    target_win = max(128, min(int(0.25 * fs_hz), n_total // 6))
    if target_win < 32:
        raise ValueError("Recording is too short to extract baseline windows.")
    n_win = max(1, n_total // target_win)
    rows: list[list[float]] = []
    hann = np.hanning(target_win)

    for i in range(n_win):
        w = centered[i * target_win : (i + 1) * target_win]
        w = w - float(np.mean(w))
        rms = float(np.sqrt(np.mean(w**2))) + EPS
        kurt = float(np.mean(w**4)) / (rms**4)
        crest = float(np.max(np.abs(w))) / rms
        shape = rms / (float(np.mean(np.abs(w))) + EPS)

        pwr = np.abs(np.fft.rfft(w * hann)) ** 2
        total_p = float(np.sum(pwr)) + EPS
        n_bins = len(pwr)
        width = max(1, n_bins // N_BANDS)
        bands = [
            float(np.log10(float(np.sum(pwr[b * width : (b + 1) * width])) / total_p + EPS))
            for b in range(N_BANDS)
        ]
        rows.append([float(np.log10(rms)), kurt, crest, shape, *bands])
    return np.asarray(rows, dtype=np.float64)


def learn_baseline(sig: SignalData, *, machine_name: str = "My Machine") -> BaselineProfile:
    """Fit a robust healthy-machine profile using per-window median and MAD."""
    feats = _window_features(sig.values, sig.fs_hz)
    med = np.median(feats, axis=0)
    mad = np.median(np.abs(feats - med), axis=0) * 1.4826
    # Floor scales so tiny baseline jitter does not create extreme z-scores on normal operation.
    floors = np.array([0.08, 0.6, 0.5, 0.15, *([0.25] * N_BANDS)], dtype=np.float64)
    scale = np.maximum(mad, floors)

    x = sig.values - float(np.mean(sig.values))
    rms = float(np.sqrt(np.mean(x**2)))
    kurt = float(np.mean(x**4)) / ((rms + EPS) ** 4)
    return BaselineProfile(
        machine_name=machine_name,
        fs_hz=sig.fs_hz,
        units=sig.units,
        n_windows=len(feats),
        healthy_rms=rms,
        healthy_kurtosis=kurt,
        feature_median=tuple(float(v) for v in med),
        feature_scale=tuple(float(v) for v in scale),
    )


def score_against_baseline(sig: SignalData, baseline: BaselineProfile) -> BaselineDrift:
    """Measure how far a recording has drifted from a saved healthy baseline (0-100)."""
    feats = _window_features(sig.values, sig.fs_hz)
    med = np.asarray(baseline.feature_median, dtype=np.float64)
    scale = np.asarray(baseline.feature_scale, dtype=np.float64)

    # Robust per-window standardized distance across physical + spectral dimensions
    z = np.abs(feats - med) / np.maximum(scale, 1e-6)
    win_dist = np.sqrt(np.mean(z**2, axis=1))
    median_dist = float(np.median(win_dist))

    # Map robust distance smoothly onto 0..100: dist=1.0 -> ~16, dist=3.0 -> ~50, dist>=6.0 -> ~90+
    drift_score = float(100.0 * (1.0 - np.exp(-0.24 * max(0.0, median_dist - 0.5))))
    drift_score = round(min(100.0, max(0.0, drift_score)), 1)

    x = sig.values - float(np.mean(sig.values))
    rms = float(np.sqrt(np.mean(x**2)))
    kurt = float(np.mean(x**4)) / ((rms + EPS) ** 4)
    rms_ratio = rms / (baseline.healthy_rms + EPS)
    kurt_delta = kurt - baseline.healthy_kurtosis
    spec_shift = float(np.mean(np.abs(np.median(feats[:, 4:], axis=0) - med[4:])))

    parts: list[str] = []
    if rms_ratio >= 1.6:
        parts.append(f"vibration energy is {rms_ratio:.1f}x higher than healthy baseline")
    elif rms_ratio <= 0.5:
        parts.append(
            f"vibration energy dropped to {rms_ratio:.1f}x of baseline (check sensor mount)"
        )
    if kurt_delta >= 2.5:
        parts.append(
            f"sharper repeating shocks (kurtosis {kurt:.1f} vs {baseline.healthy_kurtosis:.1f})"
        )
    if spec_shift >= 0.45:
        parts.append("noticeable shift in frequency content")

    if not parts:
        reason = (
            f"Matches '{baseline.machine_name}' healthy baseline closely "
            f"(drift score {drift_score:.0f}/100)."
        )
    else:
        reason = (
            f"Drifted from '{baseline.machine_name}' baseline (score {drift_score:.0f}/100): "
            + "; ".join(parts)
            + "."
        )

    return BaselineDrift(
        machine_name=baseline.machine_name,
        drift_score=drift_score,
        rms_ratio=round(rms_ratio, 2),
        kurtosis_delta=round(kurt_delta, 2),
        spectral_shift=round(spec_shift, 3),
        plain_reason=reason,
    )
