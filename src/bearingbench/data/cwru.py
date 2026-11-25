"""Case Western Reserve University (CWRU) bearing dataset loader & representation cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.signal import decimate

from .signal_ops import (
    extract_cwru_gain_invariant_physics,
    extract_cwru_gain_sensitive_16,
    extract_cwru_multiband_cnn_input,
)

CWRU_CLASSES: tuple[str, ...] = ("normal", "inner", "outer", "ball")
CWRU_CLASS_TITLES: tuple[str, ...] = (
    "Normal",
    "Inner race",
    "Outer race",
    "Ball",
)
CWRU_WINDOW_LEN = 2048
CWRU_RPM_BY_LOAD: dict[int, int] = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
NORMAL_DECIMATION = 4
CWRU_CACHE_VERSION = 1

# file_id -> (label, fault diameter in mils, motor load in HP)
CWRU_CATALOG: dict[int, tuple[str, int, int]] = {
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

CWRU_PROTOCOLS: dict[str, str] = {
    "random_window": "5-fold random window split (leakage trap)",
    "unseen_load": "4-fold grouped by motor load (0..3 HP)",
    "unseen_fault_size": "3-fold grouped by fault diameter (7, 14, 21 mils)",
}


@dataclass(frozen=True)
class CWRURecording:
    """Single CWRU drive-end accelerometer recording standardized to 12 kHz."""

    file_id: int
    label: str
    diameter_mils: int
    load_hp: int
    rpm: int
    signal: NDArray[np.float64]


@dataclass(frozen=True)
class CWRUDataset:
    """Cached multi-representation CWRU window dataset."""

    raw_x: NDArray[np.float64]  # (N, 2048)
    gain16_x: NDArray[np.float64]  # (N, 16) BearingScope gain-sensitive features
    phys27_x: NDArray[np.float64]  # (N, 27) Gain-invariant multi-band envelope features
    cnn_x: NDArray[np.float32]  # (N, 3, 256) Gain-invariant 3-band envelope spectrum
    y: NDArray[np.int64]  # (N,)
    load: NDArray[np.int64]  # (N,)
    diameter: NDArray[np.int64]  # (N,)
    rpm: NDArray[np.float64]  # (N,)
    file_id: NDArray[np.int64]  # (N,)

    def save_cache(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            cache_version=np.array([CWRU_CACHE_VERSION], dtype=np.int64),
            raw_x=self.raw_x,
            gain16_x=self.gain16_x,
            phys27_x=self.phys27_x,
            cnn_x=self.cnn_x,
            y=self.y,
            load=self.load,
            diameter=self.diameter,
            rpm=self.rpm,
            file_id=self.file_id,
        )

    @classmethod
    def load_cache(cls, path: Path) -> CWRUDataset | None:
        with np.load(path, allow_pickle=False) as d:
            if "cache_version" not in d or int(d["cache_version"][0]) != CWRU_CACHE_VERSION:
                return None
            return cls(
                raw_x=d["raw_x"],
                gain16_x=d["gain16_x"],
                phys27_x=d["phys27_x"],
                cnn_x=d["cnn_x"],
                y=d["y"],
                load=d["load"],
                diameter=d["diameter"],
                rpm=d["rpm"],
                file_id=d["file_id"],
            )


def _drive_end_signal(mat: dict[str, object], file_id: int) -> NDArray[np.float64]:
    keys = [k for k in mat if k.endswith("_DE_time")]
    preferred = f"X{file_id:03d}_DE_time"
    key = preferred if preferred in mat else (keys[0] if len(keys) == 1 else None)
    if key is None:
        raise KeyError(f"{file_id}.mat: cannot pick a drive-end channel from {keys}")
    return np.asarray(mat[key], dtype=np.float64).ravel()


def load_cwru_recording(raw_dir: Path, file_id: int) -> CWRURecording:
    """Load one CWRU .mat file and decimate 48 kHz normal files to 12 kHz."""
    label, diameter, load = CWRU_CATALOG[file_id]
    path = raw_dir / f"{file_id}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Missing CWRU recording: {path}")
    signal = _drive_end_signal(loadmat(path), file_id)
    if label == "normal":
        signal = np.asarray(decimate(signal, NORMAL_DECIMATION, zero_phase=True), dtype=np.float64)
    return CWRURecording(
        file_id=file_id,
        label=label,
        diameter_mils=diameter,
        load_hp=load,
        rpm=CWRU_RPM_BY_LOAD[load],
        signal=signal,
    )


def prepare_cwru_dataset(raw_dir: Path, *, use_cache: bool = True) -> CWRUDataset:
    """Load all 40 CWRU .mat recordings and extract aligned multi-representation windows."""
    cache_path = raw_dir / "cwru_cache.npz"
    if use_cache and cache_path.exists():
        cached = CWRUDataset.load_cache(cache_path)
        if cached is not None:
            return cached

    raw_parts: list[NDArray[np.float64]] = []
    gain16_parts: list[NDArray[np.float64]] = []
    phys27_parts: list[NDArray[np.float64]] = []
    cnn_parts: list[NDArray[np.float32]] = []
    meta_rows: list[tuple[int, int, int, int, int]] = []

    for file_id in sorted(CWRU_CATALOG):
        rec = load_cwru_recording(raw_dir, file_id)
        n_win = len(rec.signal) // CWRU_WINDOW_LEN
        wins = rec.signal[: n_win * CWRU_WINDOW_LEN].reshape(n_win, CWRU_WINDOW_LEN)
        raw_parts.append(wins)
        gain16_parts.append(
            np.stack([extract_cwru_gain_sensitive_16(w, float(rec.rpm)) for w in wins])
        )
        phys27_parts.append(
            np.stack([extract_cwru_gain_invariant_physics(w, float(rec.rpm)) for w in wins])
        )
        cnn_parts.append(np.stack([extract_cwru_multiband_cnn_input(w) for w in wins]))
        cls_idx = CWRU_CLASSES.index(rec.label)
        meta_rows.extend([(cls_idx, rec.load_hp, rec.rpm, file_id, rec.diameter_mils)] * n_win)

    meta = np.asarray(meta_rows, dtype=np.int64)
    ds = CWRUDataset(
        raw_x=np.concatenate(raw_parts, axis=0),
        gain16_x=np.concatenate(gain16_parts, axis=0),
        phys27_x=np.concatenate(phys27_parts, axis=0),
        cnn_x=np.concatenate(cnn_parts, axis=0),
        y=meta[:, 0],
        load=meta[:, 1],
        rpm=meta[:, 2].astype(np.float64),
        file_id=meta[:, 3],
        diameter=meta[:, 4],
    )
    if use_cache:
        ds.save_cache(cache_path)
    return ds


def cwru_fold_splits(
    data: CWRUDataset, protocol: str
) -> list[tuple[str, NDArray[np.bool_], NDArray[np.bool_]]]:
    """Return leak-free (fold_name, train_mask, test_mask) splits for CWRU."""
    n = len(data.y)
    if protocol == "random_window":
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        out: list[tuple[str, NDArray[np.bool_], NDArray[np.bool_]]] = []
        for k, (tr, te) in enumerate(skf.split(np.zeros(n), data.y), start=1):
            tr_m = np.zeros(n, dtype=bool)
            te_m = np.zeros(n, dtype=bool)
            tr_m[tr] = True
            te_m[te] = True
            out.append((f"rand_{k}", tr_m, te_m))
        return out

    if protocol == "unseen_load":
        return [(f"load_{hp}hp", data.load != hp, data.load == hp) for hp in sorted(set(data.load.tolist()))]

    if protocol == "unseen_fault_size":
        # Normal recordings have diameter=0; hold out load_hp in {0,1,2} across folds, keep 3 HP in train.
        normal_idx = CWRU_CLASSES.index("normal")
        out_fs: list[tuple[str, NDArray[np.bool_], NDArray[np.bool_]]] = []
        for fold_idx, diam in enumerate((7, 14, 21)):
            te_fault = data.diameter == diam
            te_norm = (data.y == normal_idx) & (data.load == fold_idx)
            te_m = te_fault | te_norm
            tr_m = ~te_m
            out_fs.append((f"diam_{diam}mil", tr_m, te_m))
        return out_fs

    raise ValueError(f"Unknown CWRU protocol: {protocol!r}")
