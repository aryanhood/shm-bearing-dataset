"""Local prediction pipeline backed by the centralized inference pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import ModelChoice, PredictRequest
from src.data.loader import BearingDataLoader
from src.data.preprocessor import Preprocessor
from src.inference.pipeline import InferencePipeline
from src.utils.config import init_config
from src.utils.logger import get_logger, setup_root_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one local SHM prediction")
    parser.add_argument("--model", choices=["rf", "cnn"], default="cnn")
    parser.add_argument("--input", type=str, default=None, help="Optional path to a .npy file containing one 1-D signal window")
    parser.add_argument("--sample-index", type=int, default=0, help="Fallback test-set sample index when --input is not provided")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Optional path to write the JSON response")
    return parser.parse_args(argv)


def load_signal(cfg, input_path: str | None, sample_index: int) -> np.ndarray:
    if input_path:
        signal = np.load(input_path).astype(np.float32).flatten()
    else:
        loader = BearingDataLoader(cfg)
        X, y = loader.load()
        splitter = Preprocessor(
            train_frac=float(cfg["data"]["train_frac"]),
            val_frac=float(cfg["data"]["val_frac"]),
            test_frac=float(cfg["data"]["test_frac"]),
            seed=int(cfg.get("project.seed") or 42),
        )
        X_te, _ = splitter.split(X, y)["test"]
        if sample_index < 0 or sample_index >= len(X_te):
            raise IndexError(f"sample-index must be in [0, {len(X_te) - 1}]")
        signal = X_te[sample_index]

    expected = int(cfg["data"]["window_size"])
    if signal.shape[0] != expected:
        raise ValueError(f"Expected exactly {expected} samples, received {signal.shape[0]}")
    return signal


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = init_config(args.config, force=bool(args.config))
    setup_root_logger(
        level=cfg.get("project.log_level") or "INFO",
        log_file=cfg.get("artifacts.log_file"),
    )
    log = get_logger("pipelines.predict")

    signal = load_signal(cfg, args.input, args.sample_index)
    pipeline = InferencePipeline(cfg).load_artifacts()
    response = pipeline.predict(
        PredictRequest(signal=signal.tolist(), model=ModelChoice(args.model))
    )
    payload = response.model_dump()
    serialized = json.dumps(payload, indent=2)

    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
        log.info("Prediction written to %s", args.output)
    else:
        print(serialized)


if __name__ == "__main__":
    main()
