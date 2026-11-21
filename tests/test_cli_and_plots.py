"""CLI smoke test on synthetic .mat files and replay plotting verification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pytest import CaptureFixture
from scipy.io import savemat

from orderguard.cli import main
from orderguard.data import load_mat_file
from orderguard.kinematics import ER16K
from orderguard.plots import plot_closed_loop_replay
from orderguard.supervisor import ActuatorCommand


def test_load_mat_and_cli_inspect(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    fs_hz = 100_000.0
    duration_s = 1.6
    t = np.arange(int(fs_hz * duration_s), dtype=np.float64) / fs_hz
    revs = 16.0 * t + 4.0 * (t**2) / duration_s
    ttl = np.where(((revs * 1024) % 1.0) < 0.5, 4.3, 0.04)
    vib = np.exp(-((revs * ER16K.orders().bpfo) % 1.0) * 20.0) * np.sin(2.0 * np.pi * 8000.0 * t)

    mat_path = tmp_path / "O-A-1.mat"
    savemat(mat_path, {"Channel_1": vib[:, None], "Channel_2": ttl[:, None]})

    rec = load_mat_file(mat_path, fs_hz=fs_hz)
    assert rec.spec.label == "outer"
    assert rec.spec.profile == "increasing"
    assert rec.spec.trial == 1

    # CLI inspect uses SAMPLE_RATE_HZ=200_000, so write a 2.0s 200 kHz recording for CLI.
    fs_cli = 200_000.0
    t_cli = np.arange(int(fs_cli * 2.0), dtype=np.float64) / fs_cli
    revs_cli = 15.0 * t_cli
    ttl_cli = np.where(((revs_cli * 1024) % 1.0) < 0.5, 4.3, 0.04)
    vib_cli = np.exp(-((revs_cli * ER16K.orders().bpfi) % 1.0) * 20.0) * np.sin(
        2.0 * np.pi * 8000.0 * t_cli
    )
    cli_mat = tmp_path / "I-A-1.mat"
    savemat(cli_mat, {"Channel_1": vib_cli[:, None], "Channel_2": ttl_cli[:, None]})

    main(["inspect", str(cli_mat), "--artifacts", str(tmp_path / "no_artifacts")])
    out = json.loads(capsys.readouterr().out)
    assert out["sensor_audit"]["ok"] is True
    assert out["n_order_windows"] >= 1


def test_plot_closed_loop_replay_writes_png(tmp_path: Path) -> None:
    cmds = [
        ActuatorCommand(
            step=i,
            time_s=float(i) * 1.0,
            rpm=900.0 + 30.0 * i,
            mode="NOMINAL" if i < 10 else ("DERATE" if i < 20 else "SAFE_STOP"),
            predicted_class="healthy" if i < 10 else ("inner" if i < 20 else "combined"),
            fault_prob=0.05 if i < 10 else (0.75 if i < 20 else 0.95),
            velocity_scale=1.0 if i < 10 else (0.6 if i < 20 else 0.0),
            torque_limit=1.0 if i < 10 else (0.7 if i < 20 else 0.0),
            brake_engage=i >= 20,
            reason="test",
        )
        for i in range(30)
    ]
    png = tmp_path / "replay.png"
    plot_closed_loop_replay(cmds, png)
    assert png.exists()
    assert png.stat().st_size > 10_000
