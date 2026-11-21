"""Command line entry point: `orderguard <command>`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import __version__

DEFAULT_DATA_DIR = Path("data/uottawa")
DEFAULT_ARTIFACTS = Path("artifacts")
DEFAULT_ASSETS = Path("assets")


def cmd_download(args: argparse.Namespace) -> None:
    from .data import CATALOG, download_uottawa

    stems = [s.stem for s in CATALOG[: args.subset]] if args.subset else None
    paths = download_uottawa(args.data_dir, stems=stems)
    print(f"Downloaded/verified {len(paths)} files in {args.data_dir}")


def cmd_inspect(args: argparse.Namespace) -> None:
    from .data import load_mat_file
    from .edge import EdgeClassifier
    from .encoder import audit_sensors, decode_encoder
    from .features import extract_multiband_cnn_input
    from .order_tracking import (
        multiband_order_envelopes,
        order_envelope_spectrum,
        slice_order_windows,
    )

    rec = load_mat_file(args.file)
    track = decode_encoder(rec.encoder, rec.fs_hz, ppr=args.ppr)
    audit = audit_sensors(rec.vibration, track)
    _, order_sigs = multiband_order_envelopes(rec.vibration, rec.fs_hz, track)

    band_windows = []
    for osig in order_sigs:
        wins, _, _ = slice_order_windows(osig)
        band_windows.append(wins)

    orders = np.empty(0, dtype=np.float64)
    specs = []
    for w in band_windows[0]:
        orders, sp = order_envelope_spectrum(w)
        specs.append(sp)
    mean_spec = np.mean(specs, axis=0)
    mask = orders >= 0.4
    top_idx = np.argsort(mean_spec[mask])[::-1][:5]
    top_orders = [round(float(orders[mask][i]), 3) for i in top_idx]

    payload: dict[str, object] = {
        "file": rec.spec.filename,
        "sensor_audit": {
            "ok": audit.ok,
            "rms": audit.rms,
            "encoder_pulses": audit.encoder_pulses,
            "rpm_min": audit.rpm_min,
            "rpm_max": audit.rpm_max,
            "issues": list(audit.issues),
        },
        "n_order_windows": len(band_windows[0]),
        "top_orders": top_orders,
    }

    onnx_path = args.artifacts / "orderguard.onnx"
    meta_path = args.artifacts / "model_meta.json"
    if onnx_path.exists():
        edge = EdgeClassifier(onnx_path, meta_path)
        n_win = len(band_windows[0])
        x_cnn = np.stack(
            [extract_multiband_cnn_input([bw[k] for bw in band_windows]) for k in range(n_win)]
        )
        probs = edge.predict_proba(x_cnn).mean(axis=0)
        pred = edge.classes[int(np.argmax(probs))]
        payload["edge_prediction"] = {
            "class": pred,
            "mean_probabilities": {
                c: round(float(p), 4) for c, p in zip(edge.classes, probs, strict=True)
            },
        }
    print(json.dumps(payload, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    from .benchmark import prepare_dataset
    from .edge import EdgeClassifier
    from .model import export_onnx, predict_proba_torch, train_order_cnn

    ds = prepare_dataset(args.data_dir)
    # Train on Trials 1 and 2 only so Trial 3 stays completely unseen for closed-loop replay.
    train_mask = np.isin(ds.trial, [1, 2])
    if not np.any(train_mask):
        train_mask = np.ones(len(ds.y), dtype=bool)
    model = train_order_cnn(ds.cnn_x[train_mask], ds.y[train_mask])

    onnx_path = args.artifacts / "orderguard.onnx"
    meta_path = args.artifacts / "model_meta.json"
    export_onnx(model, onnx_path, meta_path)

    edge = EdgeClassifier(onnx_path, meta_path)
    pt_probs = predict_proba_torch(model, ds.cnn_x)
    ort_probs = edge.predict_proba(ds.cnn_x)
    max_diff = float(np.max(np.abs(pt_probs - ort_probs)))
    latency = edge.benchmark_latency()
    print(
        json.dumps(
            {
                "train_windows": int(train_mask.sum()),
                "onnx_path": str(onnx_path),
                "onnx_bytes": onnx_path.stat().st_size,
                "pytorch_onnx_max_abs_diff": max_diff,
                "cpu_latency_ms": latency,
            },
            indent=2,
        )
    )


def cmd_benchmark(args: argparse.Namespace) -> None:
    from .benchmark import prepare_dataset, run_benchmark, save_benchmark

    ds = prepare_dataset(args.data_dir)
    report = run_benchmark(ds)
    out = args.artifacts / "benchmark.json"
    save_benchmark(report, out)
    print(f"Saved {out} ({report['n_recordings']} recordings, {report['n_windows']} windows)")
    for row in report["results"]:  # type: ignore[attr-defined]
        print(
            f"  {row['protocol']:<22} {row['model']:<18} "
            f"acc={row['accuracy']:.2%}  macro_f1={row['macro_f1']:.3f}"
        )


def cmd_replay(args: argparse.Namespace) -> None:
    from dataclasses import asdict

    from .plots import run_closed_loop_replay

    cmds = run_closed_loop_replay(
        args.data_dir,
        args.artifacts / "orderguard.onnx",
        args.artifacts / "model_meta.json",
    )
    out = args.artifacts / "replay_trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(c) for c in cmds], indent=2) + "\n")
    prev_mode = None
    for c in cmds:
        if c.mode != prev_mode:
            print(
                f"[t={c.time_s:5.2f}s rpm={c.rpm:6.1f}] {prev_mode or 'INIT'} -> {c.mode} "
                f"(vel={c.velocity_scale:.2f}, trq={c.torque_limit:.2f}, reason={c.reason})"
            )
            prev_mode = c.mode


def cmd_plots(args: argparse.Namespace) -> None:
    from .plots import (
        plot_benchmark,
        plot_closed_loop_replay,
        plot_fft_vs_order_tracking,
        run_closed_loop_replay,
    )

    plot_fft_vs_order_tracking(args.data_dir, args.assets / "fft_vs_order_tracking.png")
    plot_benchmark(args.artifacts / "benchmark.json", args.assets / "cross_speed_benchmark.png")
    cmds = run_closed_loop_replay(
        args.data_dir,
        args.artifacts / "orderguard.onnx",
        args.artifacts / "model_meta.json",
    )
    plot_closed_loop_replay(cmds, args.assets / "closed_loop_replay.png")
    print(f"Generated 3 plots in {args.assets}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="orderguard", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="download uOttawa dataset from Mendeley Data")
    p_dl.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_dl.add_argument("--subset", type=int, default=None, help="optional number of files")
    p_dl.set_defaults(fn=cmd_download)

    p_in = sub.add_parser("inspect", help="audit sensors and run order-tracked diagnosis")
    p_in.add_argument("file", type=Path)
    p_in.add_argument("--ppr", type=int, default=1024)
    p_in.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    p_in.set_defaults(fn=cmd_inspect)

    p_tr = sub.add_parser("train", help="train OrderCNN on Trials 1-2 and export ONNX")
    p_tr.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_tr.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    p_tr.set_defaults(fn=cmd_train)

    p_bm = sub.add_parser("benchmark", help="run 3-protocol leak-free benchmark")
    p_bm.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_bm.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    p_bm.set_defaults(fn=cmd_benchmark)

    p_rp = sub.add_parser("replay", help="replay unseen Trial-3 degradation through supervisor")
    p_rp.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_rp.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    p_rp.set_defaults(fn=cmd_replay)

    p_pl = sub.add_parser("plots", help="generate all 3 figures from real uOttawa outputs")
    p_pl.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_pl.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    p_pl.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    p_pl.set_defaults(fn=cmd_plots)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
