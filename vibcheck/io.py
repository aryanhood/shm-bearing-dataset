"""Universal signal loader for CSV (Phyphox/ESP32/DAQ), WAV audio, and MAT files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat, wavfile
from scipy.signal import decimate

Units = Literal["g", "m/s2", "audio", "unknown"]


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


def _parse_units(raw: str | None) -> Units:
    if raw is None:
        return "unknown"
    low = raw.strip().lower()
    if low in {"g", "accel_g"}:
        return "g"
    if low in {"m/s2", "m/s^2", "ms2"}:
        return "m/s2"
    if low in {"audio", "wav", "pcm"}:
        return "audio"
    return "unknown"


def _infer_units_from_header(header: str) -> Units:
    low = header.lower()
    if "m/s^2" in low or "m/s2" in low or "m/s²" in low:
        return "m/s2"
    if "(g)" in low or "_g" in low or low.endswith(" g"):
        return "g"
    return "unknown"


def _is_float(text: str) -> bool:
    try:
        float(text.strip())
        return True
    except ValueError:
        return False


def _load_csv(
    path: Path,
    *,
    fs_hz: float | None,
    column: str | None,
    units: str | None,
) -> SignalData:
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError(f"{path.name}: file is empty")

    # Auto-detect common CSV delimiters.
    first = lines[0]
    delimiter = (
        ";"
        if first.count(";") > first.count(",")
        else ("\t" if first.count("\t") > first.count(",") else ",")
    )
    rows = list(csv.reader(lines, delimiter=delimiter))
    has_header = not all(_is_float(cell) for cell in rows[0] if cell.strip())
    headers = (
        [c.strip() for c in rows[0]] if has_header else [f"col_{i}" for i in range(len(rows[0]))]
    )
    data_rows = rows[1:] if has_header else rows

    numeric: list[list[float]] = []
    for r in data_rows:
        if len(r) < len(headers):
            continue
        vals = [float(c.strip()) for c in r[: len(headers)] if _is_float(c)]
        if len(vals) == len(headers):
            numeric.append(vals)

    if len(numeric) < 8:
        raise ValueError(f"{path.name}: found only {len(numeric)} numeric rows")

    arr = np.asarray(numeric, dtype=np.float64)
    time_idx: int | None = None
    for idx, h in enumerate(headers):
        if "time" in h.lower() or h.lower() in {"t", "t_s", "sec", "seconds"}:
            time_idx = idx
            break
    if time_idx is None and arr.shape[1] >= 2:
        # Heuristic: strictly increasing first column starting near 0 is likely time in seconds.
        diffs = np.diff(arr[:, 0])
        if np.all(diffs > 0.0) and float(np.std(diffs)) / (float(np.mean(diffs)) + 1e-12) < 0.25:
            time_idx = 0

    inferred_fs = fs_hz
    if inferred_fs is None:
        if time_idx is not None:
            t_col = arr[:, time_idx]
            span = float(t_col[-1] - t_col[0])
            if span <= 0.0:
                raise ValueError(f"{path.name}: non-monotonic time column")
            inferred_fs = float(len(t_col) - 1) / span
        else:
            raise ValueError(f"{path.name}: no time column detected; pass --fs <sample_rate_hz>")

    candidate_cols = [i for i in range(arr.shape[1]) if i != time_idx]
    if not candidate_cols:
        raise ValueError(f"{path.name}: no signal column found")

    if column is not None:
        matches = [i for i, h in enumerate(headers) if h.lower() == column.lower()]
        if not matches:
            raise KeyError(f"{path.name}: column '{column}' not in {headers}")
        sig_idx = matches[0]
    elif len(candidate_cols) == 1:
        sig_idx = candidate_cols[0]
    else:
        # Prefer explicit "absolute" magnitude or highest-variance axis (common in Phyphox exports).
        abs_matches = [i for i in candidate_cols if "abs" in headers[i].lower()]
        if abs_matches:
            sig_idx = abs_matches[0]
        else:
            sig_idx = max(candidate_cols, key=lambda i: float(np.var(arr[:, i])))

    resolved_units = (
        _parse_units(units) if units is not None else _infer_units_from_header(headers[sig_idx])
    )
    return SignalData(
        values=arr[:, sig_idx].copy(),
        fs_hz=float(inferred_fs),
        units=resolved_units,
        source_name=path.name,
        channel=headers[sig_idx],
    )


def _load_wav(path: Path, *, units: str | None) -> SignalData:
    sr, data = wavfile.read(path)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), info.max))
        arr = arr / scale
    resolved_units = _parse_units(units) if units is not None else "audio"
    return SignalData(
        values=arr.ravel(),
        fs_hz=float(sr),
        units=resolved_units,
        source_name=path.name,
        channel="audio_mono",
    )


def _load_mat(
    path: Path,
    *,
    fs_hz: float | None,
    column: str | None,
    units: str | None,
) -> SignalData:
    mat = loadmat(path)
    keys = [k for k in mat if not k.startswith("__")]
    if column is not None:
        if column not in mat:
            raise KeyError(f"{path.name}: key '{column}' not found in {keys}")
        key = column
        default_fs = fs_hz or 12_000.0
    elif "Channel_1" in mat:
        key = "Channel_1"
        default_fs = fs_hz or 200_000.0
    else:
        de_keys = [k for k in keys if k.endswith("_DE_time")]
        if de_keys:
            preferred = f"X{path.stem.zfill(3)}_DE_time"
            key = preferred if preferred in mat else de_keys[0]
            default_fs = fs_hz or 12_000.0
        elif keys:
            key = keys[0]
            default_fs = fs_hz or 12_000.0
        else:
            raise ValueError(f"{path.name}: no signal arrays found")

    sig = np.asarray(mat[key], dtype=np.float64).ravel()
    # CWRU normal baseline files (97..100.mat) are recorded at 48 kHz while fault files are 12 kHz.
    if (
        fs_hz is None
        and key.endswith("_DE_time")
        and path.stem.isdigit()
        and int(path.stem) in {97, 98, 99, 100}
    ):
        sig = np.asarray(decimate(sig, 4, zero_phase=True), dtype=np.float64)
        default_fs = 12_000.0

    resolved_units = _parse_units(units) if units is not None else "g"
    return SignalData(
        values=sig,
        fs_hz=float(default_fs),
        units=resolved_units,
        source_name=path.name,
        channel=key,
    )


def load_signal(
    path: Path,
    *,
    fs_hz: float | None = None,
    column: str | None = None,
    units: str | None = None,
) -> SignalData:
    """Load a vibration or acoustic recording from .csv, .txt, .wav, or .mat."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return _load_csv(path, fs_hz=fs_hz, column=column, units=units)
    if suffix == ".wav":
        return _load_wav(path, units=units)
    if suffix == ".mat":
        return _load_mat(path, fs_hz=fs_hz, column=column, units=units)
    raise ValueError(f"Unsupported file format '{suffix}'. Supported: .csv, .txt, .wav, .mat")
