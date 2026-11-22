"""Command-line interface for BearingBench reproduction, inspection & edge verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines.unsupervised import audit_signal, compute_iso_velocity, load_signal
from .data.kinematics import get_bearing


def _cmd_inspect(args: argparse.Namespace) -> int:
    sig = load_signal(Path(args.file), fs_hz=args.fs)
    qual = audit_signal(sig)
    iso = compute_iso_velocity(sig)
    bearing = get_bearing(args.bearing) if args.bearing else None
    out: dict[str, object] = {
        "source": sig.source_name,
        "fs_hz": sig.fs_hz,
        "duration_s": round(sig.duration_s, 3),
        "quality_ok": qual.ok,
        "iso_velocity_mm_s": iso.velocity_rms_mm_s if iso else None,
        "iso_zone": iso.zone if iso else None,
        "bearing": bearing.code if bearing else None,
        "fault_hz": (
            {k: round(v, 2) for k, v in bearing.frequencies_hz(args.rpm).items()}
            if bearing and args.rpm
            else None
        ),
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_reproduce(args: argparse.Namespace) -> int:
    from .evaluation.plots import generate_all_figures
    from .evaluation.runner import run_all_experiments

    root = Path(args.root).resolve()
    run_all_experiments(root)
    generate_all_figures(root)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bearingbench", description="BearingBench CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    ins = sub.add_parser("inspect", help="Inspect a vibration CSV/WAV/MAT file")
    ins.add_argument("file", help="Path to signal file")
    ins.add_argument("--fs", type=float, default=None, help="Sample rate in Hz")
    ins.add_argument("--rpm", type=float, default=None, help="Shaft RPM")
    ins.add_argument("--bearing", type=str, default=None, help="Bearing code (e.g. 6205, ER16K)")
    ins.set_defaults(func=_cmd_inspect)

    rep = sub.add_parser("reproduce", help="Reproduce full benchmark + plots from raw .mat data")
    rep.add_argument("--root", default=".", help="Repository root containing data/")
    rep.set_defaults(func=_cmd_reproduce)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
