"""Verify README.md, LaTeX, and HTML Technical Report match artifacts/benchmark_results.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast


def _pct(val: float) -> str:
    return f"{100.0 * val:.2f}%"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    res_path = root / "artifacts" / "benchmark_results.json"
    if not res_path.exists():
        print(f"MISSING {res_path}", file=sys.stderr)
        return 1

    res = cast(dict[str, Any], json.loads(res_path.read_text(encoding="utf-8")))
    readme = (
        (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else ""
    )
    tex_path = root / "paper" / "technical_report.tex"
    html_path = root / "paper" / "technical_report.html"
    tex = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    html = html_path.read_text(encoding="utf-8") if html_path.exists() else ""

    if not readme or not tex or not html:
        print(
            "README.md, paper/technical_report.tex, or paper/technical_report.html missing.",
            file=sys.stderr,
        )
        return 1

    # Anchor checks against exact computed metrics from artifacts/benchmark_results.json
    uo_sp = res["uottawa"]["protocols"]["unseen_speed_profile"]
    uo_tr = res["uottawa"]["protocols"]["unseen_trial"]
    cw_fs = res["cwru"]["protocols"]["unseen_fault_size"]
    cw_ld = res["cwru"]["protocols"]["unseen_load"]
    pareto_128 = res["uottawa"]["mrl_onnx_pareto"]["128"]
    pareto_8 = res["uottawa"]["mrl_onnx_pareto"]["8"]
    unc_sp = res["uottawa"]["uncertainty_and_conformal"]["unseen_speed_profile"]
    unc_tr = res["uottawa"]["uncertainty_and_conformal"]["unseen_trial"]

    required_tokens = [
        _pct(uo_sp["order_mrl_cnn"]["accuracy_mean"]),
        _pct(uo_sp["time_env_cnn"]["accuracy_mean"]),
        _pct(uo_tr["order_mrl_cnn"]["accuracy_mean"]),
        _pct(cw_ld["cwru_mrl_cnn"]["accuracy_mean"]),
        _pct(cw_fs["gain_sensitive_mlp"]["accuracy_mean"]),
        _pct(cw_fs["cwru_mrl_cnn"]["accuracy_mean"]),
        f"{pareto_128['int8']['size_kb']:.2f}",
        _pct(pareto_8["int8"]["accuracy_mean"]),
        _pct(unc_sp["online_aci"]["coverage_mean"]),
        _pct(unc_tr["online_aci"]["coverage_mean"]),
    ]

    missing: list[str] = []
    for tok in required_tokens:
        tok_plain = tok.rstrip("%")
        if tok_plain not in readme:
            missing.append(f"README.md missing {tok}")
        if tok_plain not in tex:
            missing.append(f"technical_report.tex missing {tok}")
        if tok_plain not in html:
            missing.append(f"technical_report.html missing {tok}")

    if missing:
        for msg in missing:
            print("MISMATCH:", msg, file=sys.stderr)
        return 1

    print(
        "OK: All verified benchmark metrics match README.md, "
        "technical_report.tex, and technical_report.html."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
