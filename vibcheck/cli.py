"""Command line interface: `vibcheck <check|baseline|compare|bearings>`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from . import __version__
from .baseline import BaselineProfile, learn_baseline
from .bearings import BEARINGS
from .compare import compare_signals
from .diagnostics import ISO_20816_3_LIMITS, IsoClass, analyze
from .io import load_signal
from .report import format_terminal_report, write_comparison_html, write_html_report


def cmd_check(args: argparse.Namespace) -> None:
    sig = load_signal(args.file, fs_hz=args.fs, column=args.column, units=args.units)
    base = BaselineProfile.load(args.baseline) if args.baseline is not None else None
    iso_cls = cast(IsoClass, args.iso_class)
    res = analyze(sig, rpm=args.rpm, bearing=args.bearing, iso_class=iso_cls, baseline=base)

    if args.report is not None:
        write_html_report(res, args.report)

    if args.json:
        payload = {
            "source": res.source_name,
            "status": res.status,
            "summary": res.summary,
            "rms": res.rms,
            "units": res.units,
            "kurtosis": res.kurtosis,
            "crest_factor": res.crest_factor,
            "iso_mm_s": res.iso.velocity_rms_mm_s if res.iso else None,
            "iso_zone": res.iso.zone if res.iso else None,
            "baseline_drift": res.baseline_drift.drift_score if res.baseline_drift else None,
            "findings": list(res.findings),
            "next_steps": list(res.next_steps),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(format_terminal_report(res))
        if args.report is not None:
            print(f"\nHTML report saved to: {args.report}")


def cmd_baseline(args: argparse.Namespace) -> None:
    sig = load_signal(args.file, fs_hz=args.fs, column=args.column, units=args.units)
    profile = learn_baseline(sig, machine_name=args.name)
    profile.save(args.output)
    print(
        f"Saved healthy baseline for '{profile.machine_name}' to {args.output} "
        f"({profile.n_windows} windows, RMS={profile.healthy_rms:.4f} {profile.units}, "
        f"kurtosis={profile.healthy_kurtosis:.2f})."
    )


def cmd_compare(args: argparse.Namespace) -> None:
    before = load_signal(args.before, fs_hz=args.fs, column=args.column, units=args.units)
    after = load_signal(args.after, fs_hz=args.fs, column=args.column, units=args.units)
    iso_cls = cast(IsoClass, args.iso_class)
    comp = compare_signals(before, after, rpm=args.rpm, bearing=args.bearing, iso_class=iso_cls)
    print(f"[{comp.verdict}] {comp.summary}")
    if args.report is not None:
        write_comparison_html(comp, args.report)
        print(f"Comparison HTML report saved to: {args.report}")


def cmd_bearings(args: argparse.Namespace) -> None:
    if args.rpm is not None:
        print(
            f"Bearing defect frequencies at {args.rpm:.0f} RPM ({args.rpm / 60.0:.2f} Hz shaft):\n"
        )
        print(
            f"{'Code':<8} {'BPFO (Hz)':>10} {'BPFI (Hz)':>10} "
            f"{'2xBSF (Hz)':>11} {'FTF (Hz)':>9}  Notes"
        )
        print("-" * 82)
        for code, spec in BEARINGS.items():
            f = spec.frequencies_hz(args.rpm)
            print(
                f"{code:<8} {f['bpfo']:10.2f} {f['bpfi']:10.2f} "
                f"{f['bsf_2x']:11.2f} {f['ftf']:9.2f}  {spec.description}"
            )
    else:
        print("Built-in bearing presets (pass --rpm <RPM> to see exact Hz):\n")
        print(f"{'Code':<8} {'BPFO':>7} {'BPFI':>7} {'2xBSF':>7} {'FTF':>6}  Common uses")
        print("-" * 78)
        for code, spec in BEARINGS.items():
            o = spec.orders()
            print(
                f"{code:<8} {o.bpfo:7.3f} {o.bpfi:7.3f} {o.bsf_2x:7.3f} {o.ftf:6.3f}  "
                f"{spec.description}"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vibcheck",
        description="Plain-English vibration and bearing health diagnostics from CSV, WAV, or MAT.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_chk = sub.add_parser("check", help="diagnose a CSV, WAV, or MAT recording")
    p_chk.add_argument("file", type=Path)
    p_chk.add_argument(
        "--fs", type=float, default=None, help="sampling rate in Hz (if no time col)"
    )
    p_chk.add_argument("--column", type=str, default=None, help="CSV column or MAT key")
    p_chk.add_argument("--units", type=str, default=None, choices=["g", "m/s2", "audio"])
    p_chk.add_argument("--rpm", type=float, default=None, help="shaft speed in RPM")
    p_chk.add_argument("--bearing", type=str, default=None, help="bearing preset (e.g. 608, 6205)")
    p_chk.add_argument(
        "--iso-class",
        type=str,
        default="group2_rigid",
        choices=list(ISO_20816_3_LIMITS),
    )
    p_chk.add_argument("--baseline", type=Path, default=None, help="saved healthy baseline JSON")
    p_chk.add_argument("--report", type=Path, default=None, help="output single-file HTML report")
    p_chk.add_argument("--json", action="store_true", help="output JSON instead of text report")
    p_chk.set_defaults(fn=cmd_check)

    p_base = sub.add_parser("baseline", help="learn a healthy baseline profile for your machine")
    p_base.add_argument("file", type=Path)
    p_base.add_argument("-o", "--output", type=Path, required=True, help="output baseline .json")
    p_base.add_argument("--name", type=str, default="My Machine", help="friendly machine name")
    p_base.add_argument("--fs", type=float, default=None)
    p_base.add_argument("--column", type=str, default=None)
    p_base.add_argument("--units", type=str, default=None, choices=["g", "m/s2", "audio"])
    p_base.set_defaults(fn=cmd_baseline)

    p_cmp = sub.add_parser("compare", help="compare before vs after maintenance recordings")
    p_cmp.add_argument("before", type=Path)
    p_cmp.add_argument("after", type=Path)
    p_cmp.add_argument("--fs", type=float, default=None)
    p_cmp.add_argument("--column", type=str, default=None)
    p_cmp.add_argument("--units", type=str, default=None, choices=["g", "m/s2", "audio"])
    p_cmp.add_argument("--rpm", type=float, default=None)
    p_cmp.add_argument("--bearing", type=str, default=None)
    p_cmp.add_argument(
        "--iso-class",
        type=str,
        default="group2_rigid",
        choices=list(ISO_20816_3_LIMITS),
    )
    p_cmp.add_argument("--report", type=Path, default=None, help="output comparison HTML report")
    p_cmp.set_defaults(fn=cmd_compare)

    p_brg = sub.add_parser("bearings", help="list built-in bearings and fault frequencies")
    p_brg.add_argument("--rpm", type=float, default=None, help="optional shaft RPM for Hz table")
    p_brg.set_defaults(fn=cmd_bearings)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
