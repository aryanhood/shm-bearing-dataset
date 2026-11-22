"""Tests for terminal formatting, single-file HTML reports, and all 4 CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from pytest import CaptureFixture

from vibcheck.cli import main


def test_cli_end_to_end_on_bundled_examples(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    # 1. bearings catalog
    main(["bearings", "--rpm", "1750"])
    out_b = capsys.readouterr().out
    assert "6205" in out_b and "608" in out_b

    # 2. learn baseline from healthy example
    base_json = tmp_path / "motor.json"
    main(["baseline", "examples/motor_healthy.csv", "-o", str(base_json), "--name", "Test Motor"])
    assert base_json.exists()
    capsys.readouterr()

    # 3. check faulty example with baseline + HTML report + JSON output
    report_html = tmp_path / "report.html"
    main(
        [
            "check",
            "examples/motor_inner_fault.csv",
            "--rpm",
            "1750",
            "--bearing",
            "6205",
            "--baseline",
            str(base_json),
            "--report",
            str(report_html),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] in {"INSPECT SOON", "CRITICAL"}
    assert payload["baseline_drift"] > 40.0
    assert report_html.exists()
    html_text = report_html.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html_text
    assert "What To Check Next" in html_text

    # 4. compare before (faulty) vs after (healthy) with HTML diff
    diff_html = tmp_path / "diff.html"
    main(
        [
            "compare",
            "examples/motor_inner_fault.csv",
            "examples/motor_healthy.csv",
            "--rpm",
            "1750",
            "--bearing",
            "6205",
            "--report",
            str(diff_html),
        ]
    )
    out_c = capsys.readouterr().out
    assert "IMPROVED" in out_c
    assert diff_html.exists()
