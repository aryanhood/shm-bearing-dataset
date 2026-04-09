"""
Data Loader  ·  CWRU Bearing Dataset
======================================
Dataset: Case Western Reserve University Bearing Dataset
URL    : https://engineering.case.edu/bearingdatacenter/download-data

Physical motivation
-------------------
Vibration signals from a test rig with a 2 HP motor are recorded at
12 kHz on the drive-end (DE) accelerometer.  Bearing faults are seeded
at three locations (inner race, outer race, ball) at three diameters
(0.007 in, 0.014 in, 0.021 in).  We collapse diameter into a single
"fault present" label, giving four classes:

    0 — Normal          (no fault)
    1 — Inner Race Fault
    2 — Outer Race Fault
    3 — Ball Fault

Structural-dynamics link
------------------------
Resonance peaks in the vibration spectrum shift as bearing stiffness
K(d) = K₀(1 − αd) degrades.  FFT features capture this shift directly.

If ``use_synthetic = true`` in config (the default), this module generates
CWRU-like synthetic signals so the repo runs without downloading anything.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.config import CFG
from ..utils.logger import get_logger
from ..utils.seed   import set_all_seeds

log = get_logger("data.loader")

# ─── CWRU characteristic fault frequencies at 1750 rpm ───────────────────────
_BPFI = 162.2   # ball-pass freq inner race  (Hz)
_BPFO =  107.4   # ball-pass freq outer race
_BSF  =  141.2   # ball spin freq
_FTF  =  14.9    # fundamental train freq


class BearingDataLoader:
    """
    Loads (or synthesises) raw bearing vibration signals and returns them
    as a dict ``{class_id: np.ndarray of shape (n_samples, window_size)}``.

    Parameters
    ----------
    config : dict  — project config (uses ``data`` sub-section)
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.cfg        = (config or CFG)["data"]
        self.sr         = int(self.cfg["sampling_rate"])
        self.win        = int(self.cfg["window_size"])
        self.overlap    = float(self.cfg["overlap"])
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError("data.overlap must be in [0.0, 1.0).")
        self.raw_dir    = Path(self.cfg["raw_dir"])
        self.classes: Dict[int, str] = {
            int(k): v for k, v in self.cfg["classes"].items()
        }
        set_all_seeds(int((config or CFG).get("project.seed") or 42))

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (X, y) where X.shape == (N, window_size), y.shape == (N,).

        Falls back to synthetic data if config says so or real files are absent.
        """
        if self.cfg.get("use_synthetic", True) or not self._cwru_present():
            if not self.cfg.get("use_synthetic", True):
                log.warning("CWRU raw files not found — using synthetic data.")
            return self._load_synthetic()
        return self._load_cwru()

    # ── Synthetic data ─────────────────────────────────────────────────────

    def _load_synthetic(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate CWRU-like 4-class bearing signals.

        Each class adds characteristic spectral content on top of a
        structural resonance carrier, mirroring the physical fault modes.
        """
        log.info("Generating synthetic CWRU-like bearing data…")
        rng   = np.random.default_rng(42)
        n_per = 500
        windows, labels = [], []

        for cid, cname in self.classes.items():
            for _ in range(n_per):
                sig = self._synthesise_window(cid, rng)
                windows.append(sig)
                labels.append(cid)

        X = np.stack(windows).astype(np.float32)      # (N, win)
        y = np.array(labels, dtype=np.int64)
        # shuffle
        idx = rng.permutation(len(y))
        log.info(f"Synthetic dataset: {X.shape[0]} windows, {len(self.classes)} classes")
        return X[idx], y[idx]

    def _synthesise_window(self, class_id: int, rng: np.random.Generator) -> np.ndarray:
        """Create one bearing-vibration window for the given fault class."""
        t    = np.linspace(0, self.win / self.sr, self.win, endpoint=False)
        f0   = 100.0 + rng.uniform(-5, 5)             # structural resonance
        sig  = np.sin(2 * np.pi * f0 * t)             # carrier

        if class_id == 0:   # Normal — clean resonance + light noise
            sig += 0.25 * np.sin(2 * np.pi * f0 * 2 * t)
            noise_std = 0.04

        elif class_id == 1:  # Inner Race — BPFI sidebands + amplitude mod
            bpfi = _BPFI + rng.uniform(-3, 3)
            mod  = 1.0 + 0.4 * np.sin(2 * np.pi * bpfi * t)
            sig  = mod * sig
            sig += 0.35 * np.sin(2 * np.pi * bpfi * t)
            sig += 0.15 * np.sin(2 * np.pi * (f0 + bpfi) * t)
            noise_std = 0.07

        elif class_id == 2:  # Outer Race — BPFO modulation + impulse train
            bpfo = _BPFO + rng.uniform(-3, 3)
            impulse = np.zeros_like(t)
            period  = int(self.sr / bpfo)
            impulse[::period] = rng.uniform(0.5, 1.5)
            sig += 0.4 * np.sin(2 * np.pi * bpfo * t) + impulse * 0.6
            noise_std = 0.09

        elif class_id == 3:  # Ball Fault — BSF + frequency doubling
            bsf = _BSF + rng.uniform(-3, 3)
            sig += 0.3 * np.sin(2 * np.pi * bsf * t)
            sig += 0.2 * np.sin(2 * np.pi * bsf * 2 * t)
            # slight amplitude modulation from cage rotation
            sig *= 1.0 + 0.25 * np.sin(2 * np.pi * _FTF * t)
            noise_std = 0.08
        else:
            noise_std = 0.05

        sig += rng.normal(0, noise_std, len(t))
        # normalise to unit variance
        sig /= (sig.std() + 1e-8)
        return sig.astype(np.float32)

    # ── Real CWRU loader (requires .mat files) ─────────────────────────────

    def _cwru_present(self) -> bool:
        return any(self.raw_dir.glob("*.mat"))

    def _load_cwru(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the pre-downloaded CWRU .mat files.

        Expected file naming pattern (one example per class):
          Normal_0.mat, IR007_0.mat, OR007@6_0.mat, B007_0.mat

        The loader reads the DE-side accelerometer channel (key ending
        in '_DE_time') and applies sliding-window segmentation.
        """
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy is required to load .mat files: pip install scipy")

        file_map: Dict[int, List[str]] = {
            0: ["Normal_0.mat",     "Normal_1.mat",     "Normal_2.mat"],
            1: ["IR007_0.mat",      "IR014_0.mat",      "IR021_0.mat"],
            2: ["OR007@6_0.mat",    "OR014@6_0.mat",    "OR021@6_0.mat"],
            3: ["B007_0.mat",       "B014_0.mat",       "B021_0.mat"],
        }
        windows, labels = [], []
        step = int(self.win * (1 - self.overlap))

        for cid, fnames in file_map.items():
            for fname in fnames:
                fpath = self.raw_dir / fname
                if not fpath.exists():
                    log.warning(f"Missing: {fpath} — skipping")
                    continue
                mat = loadmat(str(fpath))
                de_key = next(
                    (k for k in mat if k.endswith("_DE_time") and not k.startswith("_")),
                    None,
                )
                if de_key is None:
                    log.warning(f"No DE channel in {fname}")
                    continue
                signal = mat[de_key].squeeze().astype(np.float32)
                signal /= (signal.std() + 1e-8)
                # sliding windows
                start = 0
                while start + self.win <= len(signal):
                    windows.append(signal[start : start + self.win])
                    labels.append(cid)
                    start += step

        if not windows:
            raise RuntimeError(
                "No usable CWRU windows were extracted. "
                "Check data/raw/*.mat naming, DE channel keys, and overlap/window_size settings."
            )
        X = np.stack(windows)
        y = np.array(labels, dtype=np.int64)
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(y))
        log.info(f"CWRU dataset loaded: {X.shape[0]} windows")
        return X[idx], y[idx]
