"""Evaluation pipeline using the centralized inference pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import ModelChoice
from src.data.loader import BearingDataLoader
from src.data.preprocessor import Preprocessor
from src.inference.pipeline import InferencePipeline
from src.utils.config import init_config
from src.utils.logger import get_logger, setup_root_logger
from src.utils.metrics import compute_all, print_report
from src.utils.paths import get_artifact_paths
from src.utils.seed import set_all_seeds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved SHM artifacts")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args(argv)


def save_confusion_matrix(cm, class_names, title: str, save_path: Path, log) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("Confusion matrix saved to %s", save_path)


def save_roc_curves(y_true, y_proba, class_names, title: str, save_path: Path, log) -> None:
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))  # type: ignore[attr-defined]

    for index, (label, color) in enumerate(zip(class_names, colors)):
        RocCurveDisplay.from_predictions(
            y_bin[:, index],
            y_proba[:, index],
            name=label,
            ax=ax,
            curve_kwargs={"color": color},
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("ROC curves saved to %s", save_path)


def build_markdown_report(all_metrics: dict, class_names: list[str], cfg) -> str:
    lines = [
        "# SHM Evaluation Report\n\n",
        "## Dataset\n\n",
        f"- Dataset: CWRU bearing family ({', '.join(class_names)})\n",
        f"- Window size: {cfg['data']['window_size']} samples @ {cfg['data']['sampling_rate']} Hz\n",
        f"- Split: {cfg['data']['train_frac']:.0%} train / {cfg['data']['val_frac']:.0%} val / {cfg['data']['test_frac']:.0%} test\n\n",
        "## Results\n\n",
        "| Model | Accuracy | F1 Weighted | ROC-AUC | False Alarm Rate |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for name, metrics in all_metrics.items():
        lines.append(
            f"| {name} | {metrics.get('accuracy', 0):.4f} | {metrics.get('f1_weighted', 0):.4f} | "
            f"{metrics.get('roc_auc', 0):.4f} | {metrics.get('false_alarm_rate', 0):.4f} |\n"
        )
    return "".join(lines)


def evaluate_model(
    pipeline: InferencePipeline,
    *,
    model_choice: ModelChoice,
    X_te: np.ndarray,
    y_te: np.ndarray,
    class_names: list[str],
) -> tuple[dict, np.ndarray]:
    predictions = pipeline.predict_batch(X_te, model=model_choice)
    y_pred = np.array([item.model_output.predicted_class for item in predictions], dtype=np.int64)
    y_proba = np.array(
        [
            [item.model_output.class_probs[label] for label in class_names]
            for item in predictions
        ],
        dtype=np.float32,
    )
    metrics = compute_all(y_te, y_pred, y_proba, class_names)
    return metrics, y_proba


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = init_config(args.config, force=bool(args.config))

    setup_root_logger(
        level=cfg.get("project.log_level") or "INFO",
        log_file=cfg.get("artifacts.log_file"),
    )
    log = get_logger("pipelines.evaluate")

    seed = int(cfg.get("project.seed") or 42)
    set_all_seeds(seed)

    paths = get_artifact_paths(cfg).ensure()
    loader = BearingDataLoader(cfg)
    X, y = loader.load()
    splitter = Preprocessor(
        train_frac=float(cfg["data"]["train_frac"]),
        val_frac=float(cfg["data"]["val_frac"]),
        test_frac=float(cfg["data"]["test_frac"]),
        seed=seed,
    )
    splits = splitter.split(X, y)
    X_te, y_te = splits["test"]
    class_names = [cfg["data"]["classes"][i] for i in range(len(cfg["data"]["classes"]))]

    pipeline = InferencePipeline(cfg).load_artifacts()
    status = pipeline.status()
    if not status.compatible:
        raise RuntimeError(f"Artifacts are not compatible with the current config: {status.issues}")

    all_metrics: dict[str, dict] = {}

    if status.rf:
        rf_metrics, rf_proba = evaluate_model(
            pipeline,
            model_choice=ModelChoice.RF,
            X_te=X_te,
            y_te=y_te,
            class_names=class_names,
        )
        print_report(rf_metrics, title="Random Forest")
        save_confusion_matrix(
            rf_metrics["confusion_matrix"],
            class_names,
            "Random Forest - Confusion Matrix",
            paths.plots_dir / "confusion_matrix_rf.png",
            log,
        )
        save_roc_curves(
            y_te,
            rf_proba,
            class_names,
            "Random Forest - ROC",
            paths.plots_dir / "roc_rf.png",
            log,
        )
        all_metrics["RandomForest"] = {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in rf_metrics.items()
            if key != "report"
        }

    if status.cnn:
        cnn_metrics, cnn_proba = evaluate_model(
            pipeline,
            model_choice=ModelChoice.CNN,
            X_te=X_te,
            y_te=y_te,
            class_names=class_names,
        )
        print_report(cnn_metrics, title="CNN1D")
        save_confusion_matrix(
            cnn_metrics["confusion_matrix"],
            class_names,
            "CNN1D - Confusion Matrix",
            paths.plots_dir / "confusion_matrix_cnn.png",
            log,
        )
        save_roc_curves(
            y_te,
            cnn_proba,
            class_names,
            "CNN1D - ROC",
            paths.plots_dir / "roc_cnn.png",
            log,
        )
        all_metrics["CNN1D"] = {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in cnn_metrics.items()
            if key != "report"
        }

    if all_metrics:
        report = build_markdown_report(all_metrics, class_names, cfg)
        (paths.reports_dir / "evaluation_report.md").write_text(report, encoding="utf-8")
        (paths.reports_dir / "evaluation_metrics.json").write_text(
            json.dumps(all_metrics, indent=2),
            encoding="utf-8",
        )
        log.info("Evaluation outputs saved to %s", paths.reports_dir)


if __name__ == "__main__":
    main()
