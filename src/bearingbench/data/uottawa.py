"""University of Ottawa time-varying speed bearing dataset loader & ablation cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.io import loadmat

from .encoder import EncoderTrack, audit_sensors, decode_encoder, degrade_encoder_track
from .signal_ops import (
    CARRIER_BANDS_HZ,
    SAMPLES_PER_REV,
    WINDOW_REVS,
    extract_multiband_cnn_input,
    extract_multiband_order_physics,
    extract_multiband_time_env_baseline,
    extract_time_cnn_input,
    multiband_order_envelopes,
    order_first_cot_envelopes,
    slice_order_windows,
)

UOTTAWA_SAMPLE_RATE_HZ = 200_000.0
UOTTAWA_CLASSES: tuple[str, ...] = ("healthy", "inner", "outer", "ball", "combined")
UOTTAWA_CLASS_TITLES: tuple[str, ...] = (
    "Healthy",
    "Inner race",
    "Outer race",
    "Ball",
    "Combined",
)
CLASS_CODES: dict[str, str] = {
    "H": "healthy",
    "I": "inner",
    "O": "outer",
    "B": "ball",
    "C": "combined",
}
SPEED_PROFILES: dict[str, str] = {
    "A": "increasing",
    "B": "decreasing",
    "C": "inc_dec",
    "D": "dec_inc",
}
TRIALS: tuple[int, ...] = (1, 2, 3)

UOTTAWA_PROTOCOLS: dict[str, str] = {
    "random_window": "5-fold random window split (leakage trap)",
    "unseen_trial": "3-fold grouped by independent recording trial",
    "unseen_speed_profile": "train monotonic ramps (A/B) <-> test reversing ramps (C/D)",
}

ABLATION_CACHE_VERSION = 1


@dataclass(frozen=True)
class RecordingSpec:
    """Metadata parsed from a uOttawa recording filename (e.g. 'I-A-1')."""

    stem: str
    label: str
    class_idx: int
    profile_code: str
    profile: str
    trial: int

    @property
    def filename(self) -> str:
        return f"{self.stem}.mat"


@dataclass(frozen=True)
class Recording:
    """Loaded uOttawa vibration & encoder channels."""

    spec: RecordingSpec
    vibration: NDArray[np.float64]
    encoder: NDArray[np.float64]
    fs_hz: float


def _build_catalog() -> tuple[RecordingSpec, ...]:
    specs: list[RecordingSpec] = []
    for class_code, label in CLASS_CODES.items():
        class_idx = UOTTAWA_CLASSES.index(label)
        for profile_code, profile in SPEED_PROFILES.items():
            for trial in TRIALS:
                stem = f"{class_code}-{profile_code}-{trial}"
                specs.append(
                    RecordingSpec(
                        stem=stem,
                        label=label,
                        class_idx=class_idx,
                        profile_code=profile_code,
                        profile=profile,
                        trial=trial,
                    )
                )
    return tuple(specs)


UOTTAWA_CATALOG: tuple[RecordingSpec, ...] = _build_catalog()
SPEC_BY_STEM: dict[str, RecordingSpec] = {s.stem: s for s in UOTTAWA_CATALOG}


def available_uottawa_specs(raw_dir: Path) -> tuple[RecordingSpec, ...]:
    """Return catalog entries present on disk in canonical order."""
    return tuple(s for s in UOTTAWA_CATALOG if (raw_dir / s.filename).exists())


def load_uottawa_recording(raw_dir: Path, stem: str) -> Recording:
    """Load one uOttawa .mat recording (200 kHz Channel_1 vib, Channel_2 encoder)."""
    spec = SPEC_BY_STEM[stem]
    path = raw_dir / spec.filename
    mat = loadmat(path)
    vib = np.asarray(mat["Channel_1"], dtype=np.float64).ravel()
    enc = np.asarray(mat["Channel_2"], dtype=np.float64).ravel()
    return Recording(spec=spec, vibration=vib, encoder=enc, fs_hz=UOTTAWA_SAMPLE_RATE_HZ)


@dataclass(frozen=True)
class UottawaDataset:
    """Unified uOttawa multi-representation & ablation dataset."""

    stems: tuple[str, ...]
    window_stem: NDArray[np.str_]  # (N,) recording stem per window (for chronological ACI)
    time_x: NDArray[np.float64]  # (N, 96) time-domain envelope RF features
    order_x: NDArray[np.float64]  # (N, 132) order-domain physics RF features
    time_cnn_x: NDArray[np.float32]  # (N, 4, 400) time-domain env spectrum (for Time-Env CNN)
    cnn_x: NDArray[np.float32]  # (N, 4, 400) envelope-first Order-SES (clean 1024 PPR)
    order_first_x: NDArray[np.float32]  # (N, 4, 400) order-first COT ablation
    cpw_cnn_x: NDArray[np.float32]  # (N, 4, 400) envelope-first + Cepstrum Pre-Whitening
    tach_256_x: NDArray[np.float32]  # (N, 4, 400) degraded encoder 256 PPR
    tach_64_x: NDArray[np.float32]  # (N, 4, 400) degraded encoder 64 PPR
    tach_32_x: NDArray[np.float32]  # (N, 4, 400) degraded encoder 32 PPR
    tach_jitter_x: NDArray[np.float32]  # (N, 4, 400) 1024 PPR + 2% edge jitter
    y: NDArray[np.int64]  # (N,)
    trial: NDArray[np.int64]  # (N,)
    profile_code: NDArray[np.str_]  # (N,)
    rpm: NDArray[np.float64]  # (N,)


def _resample_envelopes_on_degraded_track(
    time_envs: list[NDArray[np.float64]],
    fs_hz: float,
    deg_track: EncoderTrack,
    n_target_windows: int,
    *,
    samples_per_rev: int = SAMPLES_PER_REV,
    window_revs: int = WINDOW_REVS,
) -> NDArray[np.float32]:
    """Extract (n_target_windows, 4, 400) Order-SES aligned 1:1 with clean windows."""
    theta_start = float(deg_track.edge_revs[0]) + 0.05
    step = 1.0 / float(samples_per_rev)
    win_len = window_revs * samples_per_rev
    total_samples_needed = n_target_windows * win_len
    theta_grid = theta_start + np.arange(total_samples_needed, dtype=np.float64) * step
    time_of_rev = PchipInterpolator(deg_track.edge_revs, deg_track.edge_times_s, extrapolate=True)
    t_max_sig = float(len(time_envs[0]) - 1) / fs_hz
    t_theta = np.clip(np.asarray(time_of_rev(theta_grid), dtype=np.float64), 0.0, t_max_sig)
    t_axis = np.arange(len(time_envs[0]), dtype=np.float64) / fs_hz

    band_windows: list[NDArray[np.float64]] = []
    for env_t in time_envs:
        env_theta = np.asarray(CubicSpline(t_axis, env_t, extrapolate=False)(t_theta), dtype=np.float64)
        env_theta = np.nan_to_num(env_theta, nan=0.0)
        band_windows.append(env_theta.reshape(n_target_windows, win_len))

    return np.stack(
        [extract_multiband_cnn_input([bw[k] for bw in band_windows]) for k in range(n_target_windows)]
    )


def prepare_uottawa_dataset(raw_dir: Path, *, use_cache: bool = True) -> UottawaDataset:
    """Load/verify primary & full ablation representations across all 60 uOttawa recordings."""
    specs = available_uottawa_specs(raw_dir)
    if not specs:
        raise FileNotFoundError(f"No uOttawa .mat recordings found in {raw_dir}")
    expected_stems = tuple(s.stem for s in specs)

    primary_cache = raw_dir / "prepared_cache.npz"
    ablation_cache = raw_dir / "ablation_cache.npz"

    primary_loaded: dict[str, Any] | None = None
    if use_cache and primary_cache.exists():
        with np.load(primary_cache, allow_pickle=False) as d:
            if tuple(str(s) for s in d["stems"]) == expected_stems:
                primary_loaded = {k: d[k] for k in d.files}

    if use_cache and primary_loaded is not None and ablation_cache.exists():
        with np.load(ablation_cache, allow_pickle=False) as ab:
            if (
                "cache_version" in ab
                and int(ab["cache_version"][0]) == ABLATION_CACHE_VERSION
                and tuple(str(s) for s in ab["stems"]) == expected_stems
                and len(ab["y"]) == len(primary_loaded["y"])
            ):
                return UottawaDataset(
                    stems=expected_stems,
                    window_stem=np.asarray(ab["window_stem"], dtype=np.str_),
                    time_x=np.asarray(primary_loaded["time_x"], dtype=np.float64),
                    order_x=np.asarray(primary_loaded["order_x"], dtype=np.float64),
                    time_cnn_x=np.asarray(ab["time_cnn_x"], dtype=np.float32),
                    cnn_x=np.asarray(primary_loaded["cnn_x"], dtype=np.float32),
                    order_first_x=np.asarray(ab["order_first_x"], dtype=np.float32),
                    cpw_cnn_x=np.asarray(ab["cpw_cnn_x"], dtype=np.float32),
                    tach_256_x=np.asarray(ab["tach_256_x"], dtype=np.float32),
                    tach_64_x=np.asarray(ab["tach_64_x"], dtype=np.float32),
                    tach_32_x=np.asarray(ab["tach_32_x"], dtype=np.float32),
                    tach_jitter_x=np.asarray(ab["tach_jitter_x"], dtype=np.float32),
                    y=np.asarray(primary_loaded["y"], dtype=np.int64),
                    trial=np.asarray(primary_loaded["trial"], dtype=np.int64),
                    profile_code=np.asarray(primary_loaded["profile_code"], dtype=np.str_),
                    rpm=np.asarray(primary_loaded["rpm"], dtype=np.float64),
                )

    # Extract full representations directly from the 60 real .mat recordings
    time_parts: list[NDArray[np.float64]] = []
    order_parts: list[NDArray[np.float64]] = []
    time_cnn_parts: list[NDArray[np.float32]] = []
    cnn_parts: list[NDArray[np.float32]] = []
    order_first_parts: list[NDArray[np.float32]] = []
    cpw_parts: list[NDArray[np.float32]] = []
    t256_parts: list[NDArray[np.float32]] = []
    t64_parts: list[NDArray[np.float32]] = []
    t32_parts: list[NDArray[np.float32]] = []
    tjitter_parts: list[NDArray[np.float32]] = []

    w_stems: list[str] = []
    labels: list[int] = []
    trials: list[int] = []
    profiles: list[str] = []
    rpms: list[float] = []

    for i, spec in enumerate(specs, start=1):
        rec = load_uottawa_recording(raw_dir, spec.stem)
        track = decode_encoder(rec.encoder, rec.fs_hz)
        audit = audit_sensors(rec.vibration, track)
        if not audit.ok:
            raise RuntimeError(f"{spec.stem} failed sensor audit: {audit.issues}")

        time_envs, order_sigs = multiband_order_envelopes(rec.vibration, rec.fs_hz, track, use_cpw=False)
        band_wins: list[NDArray[np.float64]] = []
        win_rpms = np.empty(0, dtype=np.float64)
        for osig in order_sigs:
            wins, _, wrpms = slice_order_windows(osig)
            band_wins.append(wins)
            win_rpms = wrpms
        n_win = len(band_wins[0])

        # Time-domain baselines (96-d RF + 4x400 CNN)
        t_rf = extract_multiband_time_env_baseline(time_envs, rec.fs_hz, n_win)
        t_cnn = extract_time_cnn_input(time_envs, rec.fs_hz, n_win)

        # Clean Envelope-First Order-SES (132-d RF + 4x400 CNN)
        o_rf = np.stack([extract_multiband_order_physics([bw[k] for bw in band_wins]) for k in range(n_win)])
        c_cnn = np.stack([extract_multiband_cnn_input([bw[k] for bw in band_wins]) for k in range(n_win)])

        # Order-First COT ablation
        of_sigs = order_first_cot_envelopes(rec.vibration, rec.fs_hz, track, bands_hz=CARRIER_BANDS_HZ)
        of_wins = [slice_order_windows(osig)[0][:n_win] for osig in of_sigs]
        of_cnn = np.stack([extract_multiband_cnn_input([bw[k] for bw in of_wins]) for k in range(n_win)])

        # Cepstrum Pre-Whitening + Envelope-First COT ablation
        _, cpw_sigs = multiband_order_envelopes(rec.vibration, rec.fs_hz, track, use_cpw=True)
        cpw_wins = [slice_order_windows(osig)[0][:n_win] for osig in cpw_sigs]
        cpw_cnn = np.stack([extract_multiband_cnn_input([bw[k] for bw in cpw_wins]) for k in range(n_win)])

        # Tachometer degradation sweeps using identical time_envs
        t256 = _resample_envelopes_on_degraded_track(
            time_envs, rec.fs_hz, degrade_encoder_track(track, target_ppr=256), n_win
        )
        t64 = _resample_envelopes_on_degraded_track(
            time_envs, rec.fs_hz, degrade_encoder_track(track, target_ppr=64), n_win
        )
        t32 = _resample_envelopes_on_degraded_track(
            time_envs, rec.fs_hz, degrade_encoder_track(track, target_ppr=32), n_win
        )
        tjit = _resample_envelopes_on_degraded_track(
            time_envs,
            rec.fs_hz,
            degrade_encoder_track(track, target_ppr=1024, jitter_fraction=0.02, seed=i),
            n_win,
        )

        time_parts.append(t_rf)
        order_parts.append(o_rf)
        time_cnn_parts.append(t_cnn)
        cnn_parts.append(c_cnn)
        order_first_parts.append(of_cnn)
        cpw_parts.append(cpw_cnn)
        t256_parts.append(t256)
        t64_parts.append(t64)
        t32_parts.append(t32)
        tjitter_parts.append(tjit)

        w_stems.extend([spec.stem] * n_win)
        labels.extend([spec.class_idx] * n_win)
        trials.extend([spec.trial] * n_win)
        profiles.extend([spec.profile_code] * n_win)
        rpms.extend(float(r) for r in win_rpms)
        if i % 10 == 0 or i == len(specs):
            print(f"[uOttawa {i:02d}/{len(specs):02d}] {spec.stem}: {n_win} windows")

    ds = UottawaDataset(
        stems=expected_stems,
        window_stem=np.asarray(w_stems, dtype=np.str_),
        time_x=(
            np.asarray(primary_loaded["time_x"], dtype=np.float64)
            if primary_loaded is not None
            else np.concatenate(time_parts, axis=0)
        ),
        order_x=(
            np.asarray(primary_loaded["order_x"], dtype=np.float64)
            if primary_loaded is not None
            else np.concatenate(order_parts, axis=0)
        ),
        time_cnn_x=np.concatenate(time_cnn_parts, axis=0),
        cnn_x=(
            np.asarray(primary_loaded["cnn_x"], dtype=np.float32)
            if primary_loaded is not None
            else np.concatenate(cnn_parts, axis=0)
        ),
        order_first_x=np.concatenate(order_first_parts, axis=0),
        cpw_cnn_x=np.concatenate(cpw_parts, axis=0),
        tach_256_x=np.concatenate(t256_parts, axis=0),
        tach_64_x=np.concatenate(t64_parts, axis=0),
        tach_32_x=np.concatenate(t32_parts, axis=0),
        tach_jitter_x=np.concatenate(tjitter_parts, axis=0),
        y=(
            np.asarray(primary_loaded["y"], dtype=np.int64)
            if primary_loaded is not None
            else np.asarray(labels, dtype=np.int64)
        ),
        trial=(
            np.asarray(primary_loaded["trial"], dtype=np.int64)
            if primary_loaded is not None
            else np.asarray(trials, dtype=np.int64)
        ),
        profile_code=(
            np.asarray(primary_loaded["profile_code"], dtype=np.str_)
            if primary_loaded is not None
            else np.asarray(profiles, dtype=np.str_)
        ),
        rpm=(
            np.asarray(primary_loaded["rpm"], dtype=np.float64)
            if primary_loaded is not None
            else np.asarray(rpms, dtype=np.float64)
        ),
    )

    if use_cache:
        np.savez_compressed(
            ablation_cache,
            cache_version=np.array([ABLATION_CACHE_VERSION], dtype=np.int64),
            stems=np.array(ds.stems),
            window_stem=ds.window_stem,
            time_cnn_x=ds.time_cnn_x,
            order_first_x=ds.order_first_x,
            cpw_cnn_x=ds.cpw_cnn_x,
            tach_256_x=ds.tach_256_x,
            tach_64_x=ds.tach_64_x,
            tach_32_x=ds.tach_32_x,
            tach_jitter_x=ds.tach_jitter_x,
            y=ds.y,
        )
    return ds


def uottawa_fold_splits(
    data: UottawaDataset, protocol: str
) -> list[tuple[str, NDArray[np.bool_], NDArray[np.bool_]]]:
    """Return (fold_name, train_mask, test_mask) splits for uOttawa."""
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

    if protocol == "unseen_trial":
        return [
            (f"trial_{t}", data.trial != t, data.trial == t)
            for t in sorted(set(data.trial.tolist()))
        ]

    if protocol == "unseen_speed_profile":
        mono = np.isin(data.profile_code, ["A", "B"])
        rev = np.isin(data.profile_code, ["C", "D"])
        return [
            ("train_AB_test_CD", mono, rev),
            ("train_CD_test_AB", rev, mono),
        ]

    raise ValueError(f"Unknown uOttawa protocol: {protocol!r}")
