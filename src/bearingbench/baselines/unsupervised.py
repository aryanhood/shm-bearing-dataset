"""Zero-label unsupervised baselines: ISO 20816-3 severity, Median/MAD drift, and CSV/WAV I/O."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat, wavfile
from scipy.signal import decimate
from sklearn.metrics import average_precision_score, roc_auc_score

from ..data.cwru import CWRU_CATALOG, load_cwru_recording
from ..data.uottawa import available_uottawa_specs, load_uottawa_recording

G_TO_MS2 = 9.80665
EPS = 1e-12
N_BANDS = 8
Units = Literal["g", "m/s2", "audio", "unknown"]
IsoClass = Literal["group2_rigid", "group2_flexible", "group1_rigid", "group1_flexible"]

ISO_20816_3_LIMITS: dict[IsoClass, tuple[float, float, float]] = {
    "group2_rigid": (1.4, 2.8, 4.5),
    "group2_flexible": (2.3, 4.5, 7.1),
    "group1_rigid": (2.3, 4.5, 7.1),
    "group1_flexible": (3.5, 7.1, 11.0),
}

FEATURE_NAMES: tuple[str, ...] = (
    "log_rms",
    "kurtosis",
    "crest_factor",
    "shape_factor",
    *(f"band_{i}" for i in range(N_BANDS)),
)


@dataclass(frozen=True)
class SignalData:
    """Loaded 1D vibration or acoustic time-series recording."""

    values: NDArray[np.float64]
    fs_hz: float
    units: Units
    source_name: str
    channel: str

    @property
    def duration_s(self) -> float:
        return float(len(self.values)) / self.fs_hz if self.fs_hz > 0.0 else 0.0


@dataclass(frozen=True)
class SignalQuality:
    ok: bool
    clipped_ratio: float
    dropout_ratio: float
    dc_offset: float
    issues: tuple[str, ...]


@dataclass(frozen=True)
class IsoSeverity:
    velocity_rms_mm_s: float
    zone: Literal["A", "B", "C", "D"]
    machine_class: IsoClass
    limits_mm_s: tuple[float, float, float]


@dataclass(frozen=True)
class BaselineProfile:
    """Portable healthy-machine fingerprint fitted with Median + MAD."""

    machine_name: str
    fs_hz: float
    units: str
    n_windows: int
    healthy_rms: float
    healthy_kurtosis: float
    feature_median: tuple[float, ...]
    feature_scale: tuple[float, ...]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
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
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path


@dataclass(frozen=True)
class BaselineDrift:
    """Zero-label anomaly score comparing a recording to a healthy baseline."""

    machine_name: str
    drift_score: float
    raw_distance: float
    rms_ratio: float
    kurtosis_delta: float
    spectral_shift: float


def _is_float(text: str) -> bool:
    try:
        float(text.strip())
        return True
    except ValueError:
        return False


def load_signal(path: Path, *, fs_hz: float | None = None, units: str | None = None) -> SignalData:
    """Load vibration signal from CSV, WAV, or MAT file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        lines = [
            ln
            for ln in path.read_text(encoding="utf-8-sig").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        delim = ";" if lines[0].count(";") > lines[0].count(",") else ","
        rows = list(csv.reader(lines, delimiter=delim))
        has_hdr = not all(_is_float(c) for c in rows[0] if c.strip())
        headers = (
            [c.strip() for c in rows[0]] if has_hdr else [f"c{i}" for i in range(len(rows[0]))]
        )
        data = np.asarray(
            [
                [float(c.strip()) for c in r[: len(headers)]]
                for r in (rows[1:] if has_hdr else rows)
                if len(r) >= len(headers) and all(_is_float(c) for c in r[: len(headers)])
            ],
            dtype=np.float64,
        )
        t_idx = next(
            (i for i, h in enumerate(headers) if "time" in h.lower() or h.lower() == "t"), None
        )
        inferred_fs = fs_hz
        if inferred_fs is None:
            if t_idx is None:
                raise ValueError("Pass --fs for CSV without a time column")
            inferred_fs = float(len(data) - 1) / float(data[-1, t_idx] - data[0, t_idx])
        cands = [i for i in range(data.shape[1]) if i != t_idx]
        sig_idx = max(cands, key=lambda i: float(np.var(data[:, i])))
        return SignalData(
            values=data[:, sig_idx].copy(),
            fs_hz=float(inferred_fs),
            units="g" if units is None else "m/s2" if units == "m/s2" else "g",
            source_name=path.name,
            channel=headers[sig_idx],
        )
    if suffix == ".wav":
        sr, arr = wavfile.read(path)
        x = np.asarray(arr, dtype=np.float64)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            x = x / float(max(abs(info.min), info.max))
        return SignalData(
            values=x.ravel(), fs_hz=float(sr), units="audio", source_name=path.name, channel="mono"
        )
    if suffix == ".mat":
        mat = loadmat(path)
        keys = [k for k in mat if not k.startswith("__")]
        key = (
            "Channel_1"
            if "Channel_1" in mat
            else next((k for k in keys if k.endswith("_DE_time")), keys[0])
        )
        x = np.asarray(mat[key], dtype=np.float64).ravel()
        inferred_fs = fs_hz or (200_000.0 if key == "Channel_1" else 12_000.0)
        if (
            fs_hz is None
            and key.endswith("_DE_time")
            and path.stem.isdigit()
            and int(path.stem) in {97, 98, 99, 100}
        ):
            x = np.asarray(decimate(x, 4, zero_phase=True), dtype=np.float64)
            inferred_fs = 12_000.0
        return SignalData(
            values=x, fs_hz=float(inferred_fs), units="g", source_name=path.name, channel=key
        )
    raise ValueError(f"Unsupported format: {suffix}")


def audit_signal(sig: SignalData) -> SignalQuality:
    """Check raw signal for non-finite values, flatline dropout, or clipping."""
    x = sig.values
    issues: list[str] = []
    if len(x) == 0 or not np.all(np.isfinite(x)):
        return SignalQuality(False, 1.0, 1.0, 0.0, ("Signal contains NaN or Inf.",))
    dc = float(np.mean(x))
    centered = x - dc
    rms = float(np.sqrt(np.mean(centered**2)))
    dropout = float(np.mean(np.abs(np.diff(x)) < 1e-9)) if len(x) > 1 else 1.0
    if rms < 1e-7 or dropout > 0.30:
        issues.append("Signal is flat or disconnected.")
    peak = float(np.max(np.abs(x)))
    clipped = float(np.mean(np.abs(x) >= 0.998 * peak)) if peak > 0.0 else 0.0
    if clipped > 0.01:
        issues.append("Sensor clipping detected.")
    return SignalQuality(
        ok=len(issues) == 0,
        clipped_ratio=round(clipped, 5),
        dropout_ratio=round(dropout, 5),
        dc_offset=round(dc, 6),
        issues=tuple(issues),
    )


def compute_iso_velocity(
    sig: SignalData,
    *,
    machine_class: IsoClass = "group2_rigid",
) -> IsoSeverity | None:
    """Integrate acceleration to ISO 20816-3 broadband RMS velocity (10-1000 Hz)."""
    if sig.units not in {"g", "m/s2"} or sig.fs_hz < 40.0:
        return None
    x = sig.values - float(np.mean(sig.values))
    a_ms2 = x * G_TO_MS2 if sig.units == "g" else x
    n = len(a_ms2)
    spec = np.fft.rfft(a_ms2)
    freqs = np.fft.rfftfreq(n, d=1.0 / sig.fs_hz)
    f_hi = min(1000.0, 0.45 * sig.fs_hz)
    mask = (freqs >= 10.0) & (freqs <= f_hi)
    vel_spec = np.zeros_like(spec)
    vel_spec[mask] = spec[mask] / (1j * 2.0 * np.pi * freqs[mask])
    vel_ms = np.fft.irfft(vel_spec, n=n)
    v_rms_mm_s = 1000.0 * float(np.sqrt(np.mean(vel_ms**2)))

    lim_ab, lim_bc, lim_cd = ISO_20816_3_LIMITS[machine_class]
    zone: Literal["A", "B", "C", "D"]
    if v_rms_mm_s < lim_ab:
        zone = "A"
    elif v_rms_mm_s < lim_bc:
        zone = "B"
    elif v_rms_mm_s < lim_cd:
        zone = "C"
    else:
        zone = "D"
    return IsoSeverity(round(v_rms_mm_s, 3), zone, machine_class, (lim_ab, lim_bc, lim_cd))


def _window_features(x: NDArray[np.float64], fs_hz: float) -> NDArray[np.float64]:
    centered = x - float(np.mean(x))
    n_total = len(centered)
    target_win = max(128, min(int(0.25 * fs_hz), n_total // 6))
    n_win = max(1, n_total // target_win)
    hann = np.hanning(target_win)
    rows: list[list[float]] = []
    for i in range(n_win):
        w = centered[i * target_win : (i + 1) * target_win]
        w = w - float(np.mean(w))
        rms = float(np.sqrt(np.mean(w**2))) + EPS
        kurt = float(np.mean(w**4)) / (rms**4)
        crest = float(np.max(np.abs(w))) / rms
        shape = rms / (float(np.mean(np.abs(w))) + EPS)
        pwr = np.abs(np.fft.rfft(w * hann)) ** 2
        total_p = float(np.sum(pwr)) + EPS
        width = max(1, len(pwr) // N_BANDS)
        bands = [
            float(np.log10(float(np.sum(pwr[b * width : (b + 1) * width])) / total_p + EPS))
            for b in range(N_BANDS)
        ]
        rows.append([float(np.log10(rms)), kurt, crest, shape, *bands])
    return np.asarray(rows, dtype=np.float64)


def learn_baseline(sig: SignalData, *, machine_name: str = "Healthy Baseline") -> BaselineProfile:
    """Fit zero-label Median/MAD baseline profile from a healthy recording."""
    feats = _window_features(sig.values, sig.fs_hz)
    med = np.median(feats, axis=0)
    mad = np.median(np.abs(feats - med), axis=0) * 1.4826
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
    """Score anomaly drift against a single healthy reference recording."""
    feats = _window_features(sig.values, sig.fs_hz)
    med = np.asarray(baseline.feature_median, dtype=np.float64)
    scale = np.asarray(baseline.feature_scale, dtype=np.float64)
    z = np.abs(feats - med) / np.maximum(scale, 1e-6)
    median_dist = float(np.median(np.sqrt(np.mean(z**2, axis=1))))
    drift_score = float(100.0 * (1.0 - np.exp(-0.24 * max(0.0, median_dist - 0.5))))
    drift_score = round(min(100.0, max(0.0, drift_score)), 2)

    x = sig.values - float(np.mean(sig.values))
    rms = float(np.sqrt(np.mean(x**2)))
    kurt = float(np.mean(x**4)) / ((rms + EPS) ** 4)
    return BaselineDrift(
        machine_name=baseline.machine_name,
        drift_score=drift_score,
        raw_distance=round(median_dist, 4),
        rms_ratio=round(rms / (baseline.healthy_rms + EPS), 3),
        kurtosis_delta=round(kurt - baseline.healthy_kurtosis, 3),
        spectral_shift=round(float(np.mean(np.abs(np.median(feats[:, 4:], axis=0) - med[4:]))), 4),
    )


def evaluate_zero_label_cwru(raw_dir: Path) -> dict[str, object]:
    """Fit Median/MAD baseline strictly on 0 HP normal (97.mat), test on unseen 1..3 HP & faults.

    Also measures ISO 20816-3 Zone A false-negative trap across all 40 CWRU recordings.
    """
    ref = load_cwru_recording(raw_dir, 97)
    base_sig = SignalData(ref.signal, 12_000.0, "g", "97.mat", "DE")
    baseline = learn_baseline(base_sig, machine_name="CWRU 0 HP Healthy (97.mat)")

    y_true: list[int] = []
    scores: list[float] = []
    iso_zones: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
    fault_in_zone_a = 0
    total_faults = 0
    healthy_scores: list[float] = []
    fault_scores: list[float] = []

    for fid in sorted(CWRU_CATALOG):
        rec = load_cwru_recording(raw_dir, fid)
        sig = SignalData(rec.signal, 12_000.0, "g", f"{fid}.mat", "DE")
        iso = compute_iso_velocity(sig)
        if iso is not None:
            iso_zones[iso.zone] += 1
            if rec.label != "normal":
                total_faults += 1
                if iso.zone == "A":
                    fault_in_zone_a += 1

        if fid == 97:
            continue  # Exclude fitting reference recording from evaluation
        drift = score_against_baseline(sig, baseline)
        is_fault = 0 if rec.label == "normal" else 1
        y_true.append(is_fault)
        scores.append(drift.drift_score)
        if is_fault == 0:
            healthy_scores.append(drift.drift_score)
        else:
            fault_scores.append(drift.drift_score)

    return {
        "dataset": "CWRU",
        "reference_recording": "97.mat (0 HP normal)",
        "n_eval_recordings": len(y_true),
        "auroc": round(float(roc_auc_score(y_true, scores)), 4),
        "auprc": round(float(average_precision_score(y_true, scores)), 4),
        "healthy_max_drift": max(healthy_scores),
        "fault_min_drift": min(fault_scores),
        "clean_separation": max(healthy_scores) < min(fault_scores),
        "iso_zone_counts_all_40": iso_zones,
        "faults_masked_in_iso_zone_a": f"{fault_in_zone_a}/{total_faults}",
    }


def evaluate_zero_label_uottawa(raw_dir: Path) -> dict[str, object]:
    """Fit Median/MAD baseline on healthy A/B runs and evaluate on unseen C/D regimes."""
    specs = available_uottawa_specs(raw_dir)
    # Concatenate healthy profile A/B windows (Trial 1 only, zero label leakage) to build baseline
    ref_parts = [
        load_uottawa_recording(raw_dir, stem).vibration
        for stem in ("H-A-1", "H-B-1")
        if (raw_dir / f"{stem}.mat").exists()
    ]
    ref_concat = np.concatenate(ref_parts)
    baseline = learn_baseline(
        SignalData(ref_concat, 200_000.0, "g", "H-AB-1", "Channel_1"),
        machine_name="uOttawa Healthy Monotonic (H-A-1, H-B-1)",
    )

    y_true: list[int] = []
    scores: list[float] = []
    healthy_scores: list[float] = []
    fault_scores: list[float] = []
    for spec in specs:
        if spec.profile_code not in {"C", "D"}:
            continue  # Evaluate purely on unseen non-stationary reversing speed profiles (C/D)
        rec = load_uottawa_recording(raw_dir, spec.stem)
        drift = score_against_baseline(
            SignalData(rec.vibration, rec.fs_hz, "g", spec.filename, "Channel_1"),
            baseline,
        )
        is_fault = 0 if spec.label == "healthy" else 1
        y_true.append(is_fault)
        scores.append(drift.drift_score)
        if is_fault == 0:
            healthy_scores.append(drift.drift_score)
        else:
            fault_scores.append(drift.drift_score)

    return {
        "dataset": "uOttawa (Unseen Speed Profiles C/D)",
        "reference_recording": "H-A-1 + H-B-1",
        "n_eval_recordings": len(y_true),
        "auroc": round(float(roc_auc_score(y_true, scores)), 4),
        "auprc": round(float(average_precision_score(y_true, scores)), 4),
        "healthy_max_drift": max(healthy_scores),
        "fault_min_drift": min(fault_scores),
        "clean_separation": max(healthy_scores) < min(fault_scores),
    }
