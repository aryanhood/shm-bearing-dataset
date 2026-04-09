"""Training pipeline for SHM models and reproducible artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import BearingDataLoader
from src.data.preprocessor import Preprocessor
from src.inference.pipeline import build_artifact_manifest
from src.models.anomaly import AnomalyDetector
from src.models.cnn1d import CNN1DModel
from src.models.random_forest import RandomForestModel
from src.utils.config import get_config, init_config
from src.utils.logger import get_logger, setup_root_logger
from src.utils.metrics import compute_all, print_report
from src.utils.paths import get_artifact_paths
from src.utils.seed import set_all_seeds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHM training pipeline")
    parser.add_argument("--model", choices=["rf", "cnn", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args(argv)


def _jsonify(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in metrics.items()
        if key != "report"
    }


def _write_config_snapshot(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(cfg), sort_keys=False), encoding="utf-8")


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = init_config(args.config, force=bool(args.config))

    setup_root_logger(
        level=cfg.get("project.log_level") or "INFO",
        log_file=cfg.get("artifacts.log_file"),
    )
    log = get_logger("pipelines.train")

    seed = int(cfg.get("project.seed") or 42)
    set_all_seeds(seed)

    artifact_paths = get_artifact_paths(cfg).ensure()
    run_paths = artifact_paths.create_run_paths("train").ensure()

    log.info("Loading and preprocessing data...")
    loader = BearingDataLoader(cfg)
    X, y = loader.load()

    preprocessor = Preprocessor(
        train_frac=float(cfg["data"]["train_frac"]),
        val_frac=float(cfg["data"]["val_frac"]),
        test_frac=float(cfg["data"]["test_frac"]),
        seed=seed,
    )
    splits = preprocessor.fit_transform(X, y)
    X_tr, y_tr = splits["train"]
    X_val, y_val = splits["val"]
    X_te, y_te = splits["test"]

    artifact_paths.ensure()
    preprocessor.save(artifact_paths.preprocessor_path)
    class_names = [cfg["data"]["classes"][i] for i in range(len(cfg["data"]["classes"]))]
    results: dict[str, Any] = {
        "run_id": run_paths.root.name,
        "seed": seed,
        "dataset": {
            "n_samples": int(len(y)),
            "window_size": int(cfg["data"]["window_size"]),
            "sampling_rate": int(cfg["data"]["sampling_rate"]),
        },
    }

    if args.model in ("rf", "all"):
        log.info("Training Random Forest baseline...")
        start = time.time()
        rf = RandomForestModel(
            cfg=cfg,
            sampling_rate=int(cfg["data"]["sampling_rate"]),
            seed=seed,
        )
        rf.fit(X_tr, y_tr)
        rf_metrics = compute_all(y_te, rf.predict(X_te), rf.predict_proba(X_te), class_names)
        rf_payload = _jsonify(rf_metrics)
        rf_payload["train_time_s"] = round(time.time() - start, 2)
        results["random_forest"] = rf_payload
        rf.save(artifact_paths.rf_model_path)
        print_report(rf_metrics, title="Random Forest - Test Set")

    if args.model in ("cnn", "all"):
        log.info("Training CNN1D advanced model...")
        merged_cfg = {**cfg["models"], **cfg["training"]}
        if args.epochs is not None:
            merged_cfg["epochs"] = args.epochs
        start = time.time()
        cnn = CNN1DModel(
            cfg=merged_cfg,
            n_classes=len(cfg["data"]["classes"]),
            seed=seed,
        )
        cnn.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        cnn_metrics = compute_all(y_te, cnn.predict(X_te), cnn.predict_proba(X_te), class_names)
        cnn_payload = _jsonify(cnn_metrics)
        cnn_payload["train_time_s"] = round(time.time() - start, 2)
        results["cnn1d"] = cnn_payload
        cnn.save(artifact_paths.cnn_model_path)
        print_report(cnn_metrics, title="CNN1D - Test Set")

    if args.model == "all":
        log.info("Training Isolation Forest anomaly detector...")
        normal_windows = X_tr[y_tr == 0]
        anomaly = AnomalyDetector(
            cfg=cfg,
            sampling_rate=int(cfg["data"]["sampling_rate"]),
        )
        anomaly.fit(normal_windows)
        anomaly.save(artifact_paths.anomaly_model_path)
        scores = anomaly.score(X_te)
        threshold = float(cfg["agent"]["anomaly_threshold"])
        results["anomaly"] = {
            "threshold": threshold,
            "flagged_windows": int((scores < threshold).sum()),
            "test_windows": int(len(X_te)),
        }

    available_models = {
        "rf": artifact_paths.rf_model_path.exists(),
        "cnn": artifact_paths.cnn_model_path.exists(),
        "anomaly": artifact_paths.anomaly_model_path.exists(),
        "preprocessor": artifact_paths.preprocessor_path.exists(),
    }
    _write_config_snapshot(artifact_paths.config_snapshot_path, cfg)
    artifact_hashes = {
        "preprocessor": _sha256(artifact_paths.preprocessor_path),
        "config_snapshot": _sha256(artifact_paths.config_snapshot_path),
    }
    if available_models["rf"]:
        artifact_hashes["rf_model"] = _sha256(artifact_paths.rf_model_path)
    if available_models["cnn"]:
        artifact_hashes["cnn_model"] = _sha256(artifact_paths.cnn_model_path)
    if available_models["anomaly"]:
        artifact_hashes["anomaly_model"] = _sha256(artifact_paths.anomaly_model_path)

    manifest = build_artifact_manifest(
        cfg,
        available_models=available_models,
        artifact_hashes=artifact_hashes,
    )
    artifact_paths.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_path = artifact_paths.reports_dir / "train_results.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    run_paths.metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_config_snapshot(run_paths.config_snapshot_path, cfg)
    (run_paths.reports_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _copy_if_exists(artifact_paths.preprocessor_path, run_paths.models_dir / artifact_paths.preprocessor_path.name)
    _copy_if_exists(artifact_paths.rf_model_path, run_paths.models_dir / artifact_paths.rf_model_path.name)
    _copy_if_exists(artifact_paths.cnn_model_path, run_paths.models_dir / artifact_paths.cnn_model_path.name)
    _copy_if_exists(artifact_paths.anomaly_model_path, run_paths.models_dir / artifact_paths.anomaly_model_path.name)

    log.info("Training artifacts saved to %s", artifact_paths.models_dir)
    log.info("Run snapshot saved to %s", run_paths.root)


if __name__ == "__main__":
    main()
