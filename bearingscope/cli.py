"""Command line entry point: `bearingscope <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__

RAW = Path("data/raw")
ARTIFACTS = Path("artifacts")
WEB_DATA = Path("web/public/data")


def cmd_download(_: argparse.Namespace) -> None:
    from .download import download

    download(RAW)


def cmd_evaluate(_: argparse.Namespace) -> None:
    from .data import load_windows
    from .evaluate import PROTOCOLS, evaluate, plot_confusion, save
    from .features import extract_many

    data = load_windows(RAW)
    feats = extract_many(data.x, data.rpm)
    reports = [evaluate(data, feats, p) for p in PROTOCOLS]
    save(reports, ARTIFACTS / "metrics.json")
    plot_confusion(reports, Path("assets/confusion_matrix.png"))
    print(f"{len(data.y)} windows from {len(set(data.file_id.tolist()))} recordings")
    for r in reports:
        folds = "  ".join(f"{f['accuracy']:.1%}" for f in r.folds)
        print(
            f"  {PROTOCOLS[r.protocol]:<44} acc {r.accuracy:.2%}  F1 {r.macro_f1:.3f}"
            f"   folds: {folds}"
        )


def cmd_train(_: argparse.Namespace) -> None:
    from .export import export_web

    export_web(RAW, WEB_DATA, ARTIFACTS / "model.json")


def cmd_predict(args: argparse.Namespace) -> None:
    from .decision import decide
    from .features import extract
    from .model import Model
    from .physics import WINDOW

    signal = np.loadtxt(args.file, delimiter=",", dtype=np.float64).ravel()
    if len(signal) < WINDOW:
        sys.exit(f"need at least {WINDOW} samples at 12 kHz, got {len(signal)}")
    model = Model.load(ARTIFACTS / "model.json")
    n = len(signal) // WINDOW
    for i in range(n):
        p = model.predict_proba(extract(signal[i * WINDOW : (i + 1) * WINDOW], args.rpm))[0]
        d = decide(p)
        print(
            json.dumps(
                {
                    "window": i,
                    "status": d.status,
                    "label": d.label,
                    "confidence": round(d.confidence, 3),
                    "action": d.action,
                }
            )
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bearingscope", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="fetch the CWRU recordings").set_defaults(fn=cmd_download)
    sub.add_parser("evaluate", help="leave-one-load-out benchmark").set_defaults(fn=cmd_evaluate)
    sub.add_parser("train", help="train and export the browser model").set_defaults(fn=cmd_train)
    p = sub.add_parser("predict", help="diagnose a CSV of 12 kHz samples")
    p.add_argument("file", type=Path)
    p.add_argument("--rpm", type=float, default=1772.0)
    p.set_defaults(fn=cmd_predict)
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
