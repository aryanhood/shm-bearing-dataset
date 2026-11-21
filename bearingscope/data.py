"""Load CWRU .mat recordings and cut them into labelled, non-overlapping windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.signal import decimate

from .physics import CLASSES, RPM_BY_LOAD, WINDOW

# file_id -> (label, fault diameter in mils, motor load in HP)
CATALOG: dict[int, tuple[str, int, int]] = {
    97: ("normal", 0, 0),
    98: ("normal", 0, 1),
    99: ("normal", 0, 2),
    100: ("normal", 0, 3),
    105: ("inner", 7, 0),
    106: ("inner", 7, 1),
    107: ("inner", 7, 2),
    108: ("inner", 7, 3),
    169: ("inner", 14, 0),
    170: ("inner", 14, 1),
    171: ("inner", 14, 2),
    172: ("inner", 14, 3),
    209: ("inner", 21, 0),
    210: ("inner", 21, 1),
    211: ("inner", 21, 2),
    212: ("inner", 21, 3),
    118: ("ball", 7, 0),
    119: ("ball", 7, 1),
    120: ("ball", 7, 2),
    121: ("ball", 7, 3),
    185: ("ball", 14, 0),
    186: ("ball", 14, 1),
    187: ("ball", 14, 2),
    188: ("ball", 14, 3),
    222: ("ball", 21, 0),
    223: ("ball", 21, 1),
    224: ("ball", 21, 2),
    225: ("ball", 21, 3),
    130: ("outer", 7, 0),
    131: ("outer", 7, 1),
    132: ("outer", 7, 2),
    133: ("outer", 7, 3),
    197: ("outer", 14, 0),
    198: ("outer", 14, 1),
    199: ("outer", 14, 2),
    200: ("outer", 14, 3),
    234: ("outer", 21, 0),
    235: ("outer", 21, 1),
    236: ("outer", 21, 2),
    237: ("outer", 21, 3),
}

# The CWRU normal-baseline recordings are sampled at 48 kHz; fault files at 12 kHz.
NORMAL_DECIMATION = 4


@dataclass(frozen=True)
class Recording:
    file_id: int
    label: str
    diameter_mils: int
    load_hp: int
    rpm: int
    signal: NDArray[np.float64]  # drive-end acceleration, 12 kHz


@dataclass(frozen=True)
class Windows:
    x: NDArray[np.float64]  # (n, WINDOW)
    y: NDArray[np.int64]  # class index into CLASSES
    load: NDArray[np.int64]
    diameter: NDArray[np.int64]  # fault diameter in mils, 0 for normal
    rpm: NDArray[np.float64]
    file_id: NDArray[np.int64]


def _drive_end_signal(mat: dict[str, object], file_id: int) -> NDArray[np.float64]:
    keys = [k for k in mat if k.endswith("_DE_time")]
    preferred = f"X{file_id:03d}_DE_time"
    key = preferred if preferred in mat else (keys[0] if len(keys) == 1 else None)
    if key is None:
        raise KeyError(f"{file_id}.mat: cannot pick a drive-end channel from {keys}")
    return np.asarray(mat[key], dtype=np.float64).ravel()


def load_recording(raw_dir: Path, file_id: int) -> Recording:
    label, diameter, load = CATALOG[file_id]
    path = raw_dir / f"{file_id}.mat"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python scripts/download_cwru.py` first")
    signal = _drive_end_signal(loadmat(path), file_id)
    if label == "normal":
        signal = decimate(signal, NORMAL_DECIMATION, zero_phase=True)
    return Recording(file_id, label, diameter, load, RPM_BY_LOAD[load], signal)


def window(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    n = len(signal) // WINDOW
    return signal[: n * WINDOW].reshape(n, WINDOW)


def load_windows(raw_dir: Path) -> Windows:
    parts: list[NDArray[np.float64]] = []
    meta: list[tuple[int, int, int, int, int]] = []
    for file_id in sorted(CATALOG):
        rec = load_recording(raw_dir, file_id)
        w = window(rec.signal)
        parts.append(w)
        meta += [
            (CLASSES.index(rec.label), rec.load_hp, rec.rpm, file_id, rec.diameter_mils)
        ] * len(w)
    m = np.array(meta)
    return Windows(
        x=np.concatenate(parts),
        y=m[:, 0].astype(np.int64),
        load=m[:, 1].astype(np.int64),
        rpm=m[:, 2].astype(np.float64),
        file_id=m[:, 3].astype(np.int64),
        diameter=m[:, 4].astype(np.int64),
    )
