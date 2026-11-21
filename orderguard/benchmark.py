"""Leak-free cross-trial and cross-speed-profile benchmarking on uOttawa data."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from .data import CLASSES, available_specs, load_recording
from .encoder import audit_sensors, decode_encoder
from .features import (
    extract_multiband_cnn_input,
    extract_multiband_order_physics,
    extract_multiband_time_env_baseline,
)
from .model import export_onnx, train_order_cnn, train_random_forest
from .order_tracking import multiband_order_envelopes, slice_order_windows

CACHE_VERSION = 2

PROTOCOLS: dict[str, str] = {
    "random_window": "5-fold random window split (leakage trap)",
    "unseen_trial": "3-fold grouped by independent trial",
    "unseen_speed_profile": "train monotonic ramps (A/B) <-> test reversing ramps (C/D)",
}

MODEL_NAMES: tuple[str, ...] = ("time_env_rf", "order_physics_rf", "order_cnn_onnx")


@dataclass(frozen=True)
class PreparedDataset:
    """Cached extracted window representations across all available recordings."""

    stems: tuple[str, ...]
    time_x: NDArray[np.float64]  # (N, 96) 4-band time-env FFT (no encoder)
    order_x: NDArray[np.float64]  # (N, 132) 4-band Order-SES physics features
    cnn_x: NDArray[np.float32]  # (N, 4, 400) 4-band Order-SES tensor
    y: NDArray[np.int64]  # (N,)
    trial: NDArray[np.int64]  # (N,) in {1, 2, 3}
    profile_code: NDArray[np.str_]  # (N,) in {'A', 'B', 'C', 'D'}
    rpm: NDArray[np.float64]  # (N,)

    def save_cache(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            cache_version=np.array([CACHE_VERSION], dtype=np.int64),
            stems=np.array(self.stems),
            time_x=self.time_x,
            order_x=self.order_x,
            cnn_x=self.cnn_x,
            y=self.y,
            trial=self.trial,
            profile_code=self.profile_code,
            rpm=self.rpm,
        )

    @classmethod
    def load_cache(cls, path: Path) -> PreparedDataset | None:
        with np.load(path, allow_pickle=False) as d:
            if "cache_version" not in d or int(d["cache_version"][0]) != CACHE_VERSION:
                return None
            return cls(
                stems=tuple(str(s) for s in d["stems"]),
                time_x=d["time_x"],
                order_x=d["order_x"],
                cnn_x=d["cnn_x"],
                y=d["y"],
                trial=d["trial"],
                profile_code=d["profile_code"],
                rpm=d["rpm"],
            )


def prepare_dataset(raw_dir: Path, *, use_cache: bool = True) -> PreparedDataset:
    """Load all uOttawa recordings in raw_dir and extract time & order windows."""
    specs = available_specs(raw_dir)
    if not specs:
        raise FileNotFoundError(f"no uOttawa .mat recordings found in {raw_dir}")

    cache_path = raw_dir / "prepared_cache.npz"
    expected_stems = tuple(s.stem for s in specs)
    if use_cache and cache_path.exists():
        cached = PreparedDataset.load_cache(cache_path)
        if cached is not None and cached.stems == expected_stems:
            return cached

    time_parts: list[NDArray[np.float64]] = []
    order_parts: list[NDArray[np.float64]] = []
    cnn_parts: list[NDArray[np.float32]] = []
    labels: list[int] = []
    trials: list[int] = []
    profiles: list[str] = []
    rpms: list[float] = []

    for i, spec in enumerate(specs, start=1):
        rec = load_recording(raw_dir, spec.stem)
        track = decode_encoder(rec.encoder, rec.fs_hz)
        audit = audit_sensors(rec.vibration, track)
        if not audit.ok:
            raise RuntimeError(f"{spec.stem} failed sensor audit: {audit.issues}")

        time_envs, order_sigs = multiband_order_envelopes(rec.vibration, rec.fs_hz, track)
        band_windows: list[NDArray[np.float64]] = []
        win_rpms = np.empty(0, dtype=np.float64)
        for osig in order_sigs:
            wins, _, wrpms = slice_order_windows(osig)
            band_windows.append(wins)
            win_rpms = wrpms

        n_win = len(band_windows[0])
        t_feats = extract_multiband_time_env_baseline(time_envs, rec.fs_hz, n_win)
        o_feats = np.stack(
            [extract_multiband_order_physics([bw[k] for bw in band_windows]) for k in range(n_win)]
        )
        c_feats = np.stack(
            [extract_multiband_cnn_input([bw[k] for bw in band_windows]) for k in range(n_win)]
        )

        time_parts.append(t_feats)
        order_parts.append(o_feats)
        cnn_parts.append(c_feats)
        labels.extend([spec.class_idx] * n_win)
        trials.extend([spec.trial] * n_win)
        profiles.extend([spec.profile_code] * n_win)
        rpms.extend(float(r) for r in win_rpms)
        print(
            f"[{i:02d}/{len(specs):02d}] {spec.stem}: {n_win} windows, "
            f"RPM {audit.rpm_min:.0f}-{audit.rpm_max:.0f}"
        )

    dataset = PreparedDataset(
        stems=expected_stems,
        time_x=np.concatenate(time_parts, axis=0),
        order_x=np.concatenate(order_parts, axis=0),
        cnn_x=np.concatenate(cnn_parts, axis=0),
        y=np.asarray(labels, dtype=np.int64),
        trial=np.asarray(trials, dtype=np.int64),
        profile_code=np.asarray(profiles),
        rpm=np.asarray(rpms, dtype=np.float64),
    )
    if use_cache:
        dataset.save_cache(cache_path)
    return dataset


def fold_splits(
    dataset: PreparedDataset,
    protocol: str,
) -> list[tuple[str, NDArray[np.bool_], NDArray[np.bool_]]]:
    """Generate (fold_name, train_mask, test_mask) splits for a benchmark protocol."""
    n = len(dataset.y)
    if protocol == "random_window":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        out: list[tuple[str, NDArray[np.bool_], NDArray[np.bool_]]] = []
        for k, (tr, te) in enumerate(skf.split(np.zeros(n), dataset.y), start=1):
            tr_m = np.zeros(n, dtype=bool)
            te_m = np.zeros(n, dtype=bool)
            tr_m[tr] = True
            te_m[te] = True
            out.append((f"rand_{k}", tr_m, te_m))
        return out

    if protocol == "unseen_trial":
        out = []
        for t in sorted(set(dataset.trial.tolist())):
            te_m = dataset.trial == t
            tr_m = ~te_m
            out.append((f"trial_{t}", tr_m, te_m))
        return out

    if protocol == "unseen_speed_profile":
        mono = np.isin(dataset.profile_code, ["A", "B"])
        rev = np.isin(dataset.profile_code, ["C", "D"])
        return [
            ("train_AB_test_CD", mono, rev),
            ("train_CD_test_AB", rev, mono),
        ]

    raise ValueError(f"unknown protocol {protocol!r}")


def _eval_onnx_fold(
    x_tr: NDArray[np.float32],
    y_tr: NDArray[np.int64],
    x_te: NDArray[np.float32],
) -> NDArray[np.int64]:
    from .edge import EdgeClassifier

    pt_model = train_order_cnn(x_tr, y_tr)
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "fold.onnx"
        export_onnx(pt_model, onnx_path)
        edge = EdgeClassifier(onnx_path)
        probs = edge.predict_proba(x_te)
    return np.asarray(probs.argmax(axis=1), dtype=np.int64)


def run_benchmark(dataset: PreparedDataset) -> dict[str, object]:
    """Run all 3 protocols x 3 pipelines and return structured metrics."""
    results: list[dict[str, object]] = []
    for protocol in PROTOCOLS:
        splits = fold_splits(dataset, protocol)
        for model_name in MODEL_NAMES:
            y_true_all: list[int] = []
            y_pred_all: list[int] = []
            fold_scores: list[dict[str, float | str | int]] = []

            for fold_name, tr_m, te_m in splits:
                y_tr = dataset.y[tr_m]
                y_te = dataset.y[te_m]
                if model_name == "time_env_rf":
                    clf = train_random_forest(dataset.time_x[tr_m], y_tr)
                    pred = np.asarray(clf.predict(dataset.time_x[te_m]), dtype=np.int64)
                elif model_name == "order_physics_rf":
                    clf = train_random_forest(dataset.order_x[tr_m], y_tr)
                    pred = np.asarray(clf.predict(dataset.order_x[te_m]), dtype=np.int64)
                elif model_name == "order_cnn_onnx":
                    pred = _eval_onnx_fold(dataset.cnn_x[tr_m], y_tr, dataset.cnn_x[te_m])
                else:
                    raise ValueError(model_name)

                acc_f = float(accuracy_score(y_te, pred))
                fold_scores.append(
                    {"fold": fold_name, "n_test": int(te_m.sum()), "accuracy": round(acc_f, 4)}
                )
                y_true_all.extend(y_te.tolist())
                y_pred_all.extend(pred.tolist())

            cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(CLASSES))))
            row_sum = np.maximum(cm.sum(axis=1), 1)
            recall = cm.diagonal() / row_sum
            results.append(
                {
                    "protocol": protocol,
                    "description": PROTOCOLS[protocol],
                    "model": model_name,
                    "n_windows": len(y_true_all),
                    "accuracy": round(float(accuracy_score(y_true_all, y_pred_all)), 4),
                    "macro_f1": round(float(f1_score(y_true_all, y_pred_all, average="macro")), 4),
                    "per_class_recall": {
                        c: round(float(r), 4) for c, r in zip(CLASSES, recall, strict=True)
                    },
                    "confusion": cm.tolist(),
                    "folds": fold_scores,
                }
            )

    return {
        "n_recordings": len(dataset.stems),
        "n_windows": len(dataset.y),
        "classes": list(CLASSES),
        "results": results,
    }


def save_benchmark(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
