"""
Ablation Study
==============
Systematically varies three factors to show contribution of each component:

  Factor A — Model type:      RF  vs  CNN
  Factor B — Feature set:     time-only  vs  freq-only  vs  time+freq
  Factor C — Window size:     256  /  512  /  1024

Results are saved to:
  artifacts/reports/ablation_results.json
  artifacts/plots/ablation_heatmap.png

Usage
-----
    python -m shm ablation
    python -m pipelines.ablation --quick   # 3 configs only, for CI
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader          import BearingDataLoader
from src.data.preprocessor    import Preprocessor
from src.features.extractor   import FeatureExtractor
from src.models.random_forest  import RandomForestModel
from src.models.cnn1d          import CNN1DModel
from src.utils.config          import CFG
from src.utils.logger          import setup_root_logger, get_logger
from src.utils.seed            import set_all_seeds
from src.utils.metrics         import compute_all

setup_root_logger(level="INFO")
log = get_logger("pipelines.ablation")


# ─── grid definition ──────────────────────────────────────────────────────────
WINDOW_SIZES = [256, 512, 1024]
FEATURE_SETS = {
    "time_only":  dict(include_time=True, include_freq=False),
    "freq_only":  dict(include_time=False, include_freq=True),
    "time+freq":  dict(include_time=True, include_freq=True),
}
MODELS = ["rf", "cnn"]


def run_one(
    model_type:  str,
    feat_key:    str,
    window_size: int,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    seed: int,
    quick: bool,
) -> Dict:
    """Train one ablation configuration and return its test metrics."""
    sr       = CFG["data"]["sampling_rate"]
    n_cls    = len(CFG["data"]["classes"])
    feat_kw  = FEATURE_SETS[feat_key]

    # Slice windows to the target size
    w = min(window_size, X_tr.shape[1])
    X_tr_w = X_tr[:, :w]
    X_te_w = X_te[:, :w]

    t0 = time.time()

    if model_type == "rf":
        # Override FeatureExtractor settings inside RF
        m = RandomForestModel(CFG["models"], sampling_rate=sr, seed=seed)
        m.extractor = FeatureExtractor(
            sampling_rate = sr,
            n_fft_bins    = 64,
            include_time  = feat_kw["include_time"],
            include_freq  = feat_kw["include_freq"],
        )
        m.clf.set_params(n_estimators=50 if quick else 300)
        m.fit(X_tr_w, y_tr)
        y_pred  = m.predict(X_te_w)
        y_proba = m.predict_proba(X_te_w)

    else:  # cnn
        full_cfg = {
            **CFG["models"], **CFG["training"],
            "epochs": 5 if quick else CFG["training"]["epochs"],
        }
        m2 = CNN1DModel(full_cfg, n_classes=n_cls, seed=seed)
        m2.fit(X_tr_w, y_tr)
        y_pred  = m2.predict(X_te_w)
        y_proba = m2.predict_proba(X_te_w)

    elapsed = time.time() - t0
    metrics = compute_all(y_te, y_pred, y_proba)

    return {
        "model":        model_type,
        "feature_set":  feat_key,
        "window_size":  w,
        "accuracy":     round(metrics["accuracy"], 4),
        "f1_weighted":  round(metrics["f1_weighted"], 4),
        "roc_auc":      round(metrics.get("roc_auc", float("nan")), 4),
        "false_alarm":  round(metrics["false_alarm_rate"], 4),
        "time_s":       round(elapsed, 1),
    }


def plot_heatmap(records: List[Dict], save_path: Path) -> None:
    """F1-weighted heatmap: (window × feature-set) per model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    feat_keys = list(FEATURE_SETS.keys())
    win_sizes = sorted(set(r["window_size"] for r in records))

    for ax, model_type in zip(axes, MODELS):
        grid = np.zeros((len(win_sizes), len(feat_keys)))
        for r in records:
            if r["model"] != model_type:
                continue
            ri = win_sizes.index(r["window_size"])
            ci = feat_keys.index(r["feature_set"])
            grid[ri, ci] = r["f1_weighted"]

        im = ax.imshow(grid, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
        ax.set_xticks(range(len(feat_keys))); ax.set_xticklabels(feat_keys, rotation=20, ha="right")
        ax.set_yticks(range(len(win_sizes))); ax.set_yticklabels([str(w) for w in win_sizes])
        ax.set_xlabel("Feature set"); ax.set_ylabel("Window size")
        ax.set_title(f"{model_type.upper()} — F1 Weighted", fontweight="bold")

        for ri in range(len(win_sizes)):
            for ci in range(len(feat_keys)):
                ax.text(ci, ri, f"{grid[ri, ci]:.3f}", ha="center", va="center",
                        fontsize=9, color="black")

    plt.colorbar(im, ax=axes, label="F1 Weighted", shrink=0.8)
    plt.suptitle("Ablation Study — F1 Weighted Score", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Ablation heatmap → {save_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run 3 configs only (for fast CI checks)")
    args = parser.parse_args(argv)

    seed = int(CFG.get("project.seed") or 42)
    set_all_seeds(seed)

    # Load full dataset at max window size
    loader   = BearingDataLoader(CFG)
    X, y     = loader.load()
    cfg_data = CFG["data"]
    prep     = Preprocessor(
        train_frac=cfg_data["train_frac"],
        val_frac=cfg_data["val_frac"],
        test_frac=cfg_data["test_frac"],
        seed=seed,
    )
    splits   = prep.fit_transform(X, y)
    X_tr, y_tr = splits["train"]
    X_te, y_te = splits["test"]

    # ── build run list ─────────────────────────────────────────────────────────
    grid = list(product(MODELS, FEATURE_SETS.keys(), WINDOW_SIZES))
    if args.quick:
        grid = [("rf", "time+freq", 1024), ("cnn", "time+freq", 1024), ("rf", "freq_only", 512)]

    log.info(f"Running {len(grid)} ablation configs…")
    records = []
    for i, (model_type, feat_key, window_size) in enumerate(grid, 1):
        log.info(f"[{i}/{len(grid)}] model={model_type}  features={feat_key}  window={window_size}")
        try:
            row = run_one(model_type, feat_key, window_size,
                          X_tr, y_tr, X_te, y_te, seed, args.quick)
            records.append(row)
            log.info(f"      → f1={row['f1_weighted']:.4f}  acc={row['accuracy']:.4f}")
        except Exception as exc:
            log.error(f"Config failed: {exc}")

    # ── save JSON ──────────────────────────────────────────────────────────────
    rep_dir  = Path(CFG["artifacts"]["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)
    out = rep_dir / "ablation_results.json"
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    log.info(f"Ablation results → {out}")

    # ── heatmap ────────────────────────────────────────────────────────────────
    if not args.quick:
        plot_dir = Path(CFG["artifacts"]["plots_dir"])
        plot_heatmap(records, plot_dir / "ablation_heatmap.png")

    # ── print summary table ────────────────────────────────────────────────────
    log.info("\n{:<6} {:<12} {:<8} {:>8} {:>8} {:>8}".format(
        "model", "feat_set", "window", "acc", "f1_w", "roc_auc"))
    for r in sorted(records, key=lambda x: -x["f1_weighted"]):
        log.info("{model:<6} {feature_set:<12} {window_size:<8} "
                 "{accuracy:>8.4f} {f1_weighted:>8.4f} {roc_auc:>8.4f}".format(**r))


if __name__ == "__main__":
    main()
