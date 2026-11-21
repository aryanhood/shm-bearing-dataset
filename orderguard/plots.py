"""Reproducible engineering figures generated from real uOttawa recordings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from numpy.typing import NDArray

from .data import CLASS_TITLES, CLASSES, load_recording
from .encoder import audit_sensors, decode_encoder
from .features import extract_multiband_cnn_input
from .kinematics import ER16K
from .order_tracking import (
    multiband_order_envelopes,
    order_envelope_spectrum,
    resample_to_angle,
    slice_order_windows,
    squared_envelope_time,
    time_envelope_fft,
)
from .supervisor import ActuatorCommand, SupervisoryController

MODEL_LABELS = {
    "time_env_rf": "4-band Time-Env FFT + RF (no encoder)",
    "order_physics_rf": "4-band Order-SES Physics + RF (encoder COT)",
    "order_cnn_onnx": "4-band Order-SES 1D-CNN (ONNX edge)",
}

PROTOCOL_LABELS = {
    "random_window": "Random window\n(leakage trap)",
    "unseen_trial": "Unseen trial\n(3-fold CV)",
    "unseen_speed_profile": "Unseen speed profile\n(A/B <-> C/D)",
}


def plot_fft_vs_order_tracking(raw_dir: Path, out_path: Path, stem: str = "I-A-1") -> None:
    """Compare smeared time-domain envelope FFT against encoder-locked Order-SES."""
    rec = load_recording(raw_dir, stem)
    track = decode_encoder(rec.encoder, rec.fs_hz)
    env_t = squared_envelope_time(rec.vibration, rec.fs_hz, bandpass_hz=(500.0, 5000.0))
    order_sig = resample_to_angle(env_t, rec.fs_hz, track)
    windows_theta, _, _ = slice_order_windows(order_sig)

    freqs, time_spec = time_envelope_fft(env_t, rec.fs_hz, max_hz=350.0)
    orders = np.empty(0, dtype=np.float64)
    specs: list[NDArray[np.float64]] = []
    for w in windows_theta:
        orders, sp = order_envelope_spectrum(w, max_order=20.0)
        specs.append(sp)
    mean_order_spec = np.mean(specs, axis=0)
    orders_mask = (orders >= 0.3) & (orders <= 20.0)
    freqs_mask = freqs >= 5.0

    bpfi = ER16K.orders().bpfi
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=160)

    # Panel A: encoder RPM profile
    ax = axes[0]
    ax.plot(track.rpm_times_s, track.rpm_values, color="#0f766e", lw=1.8)
    ax.set_title(f"Decoded 1024-PPR Encoder Speed ({stem})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Shaft Speed (RPM)")
    ax.grid(alpha=0.25)

    # Panel B: time-domain envelope FFT (smeared)
    ax = axes[1]
    t_vals = time_spec[freqs_mask]
    t_norm = t_vals / (float(np.max(t_vals)) + 1e-12)
    ax.plot(freqs[freqs_mask], t_norm, color="#b45309", lw=1.2)
    ax.set_title("Time-Domain Envelope FFT (No Encoder)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized Magnitude")
    ax.text(
        0.96,
        0.90,
        f"BPFI drifts\n{bpfi * track.rpm_min / 60:.0f} -> {bpfi * track.rpm_max / 60:.0f} Hz",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fffbeb", "edgecolor": "#f59e0b"},
    )
    ax.grid(alpha=0.25)

    # Panel C: order-domain squared envelope spectrum (sharp peaks)
    ax = axes[2]
    o_vals = mean_order_spec[orders_mask]
    o_norm = o_vals / (float(np.max(o_vals)) + 1e-12)
    ax.plot(orders[orders_mask], o_norm, color="#1d4ed8", lw=1.4)
    for h in (1, 2, 3):
        ax.axvline(h * bpfi, color="#dc2626", ls="--", lw=1.0, alpha=0.8)
        ax.text(
            h * bpfi + 0.18,
            0.92 - 0.08 * (h - 1),
            f"{h}xBPFI",
            color="#dc2626",
            fontsize=7.5,
        )
    ax.axvline(bpfi - 1.0, color="#64748b", ls=":", lw=0.9)
    ax.axvline(bpfi + 1.0, color="#64748b", ls=":", lw=0.9)
    ax.text(bpfi + 1.15, 0.62, "BPFI+/-1x", color="#475569", fontsize=7.5)
    ax.set_title("Encoder-Locked Order-SES (Angular Domain)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Shaft Order (x shaft rev)")
    ax.set_ylabel("Normalized Magnitude")
    ax.set_xlim(0.3, 20.0)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_benchmark(benchmark_json: Path, out_path: Path) -> None:
    """Plot 3-protocol generalization bars and unseen-speed confusion matrix."""
    data: dict[str, Any] = json.loads(benchmark_json.read_text())
    results: list[dict[str, Any]] = data["results"]
    protocols = ["random_window", "unseen_trial", "unseen_speed_profile"]
    models = ["time_env_rf", "order_physics_rf", "order_cnn_onnx"]

    lookup = {(str(r["protocol"]), str(r["model"])): r for r in results}

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.4),
        dpi=160,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    x = np.arange(len(protocols), dtype=np.float64)
    width = 0.24
    colors = ["#94a3b8", "#0f766e", "#1d4ed8"]
    for i, m in enumerate(models):
        vals = [float(lookup[(p, m)]["accuracy"]) * 100.0 for p in protocols]
        bars = ax1.bar(x + (i - 1) * width, vals, width, label=MODEL_LABELS[m], color=colors[i])
        for b, v in zip(bars, vals, strict=True):
            ax1.text(
                b.get_x() + b.get_width() / 2.0,
                v + 1.2,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax1.set_xticks(x, [PROTOCOL_LABELS[p] for p in protocols], fontsize=8.5)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 134)
    n_win = int(data["n_windows"])
    ax1.set_title(
        f"60-Recording uOttawa Benchmark ({n_win} Windows)",
        fontsize=10,
        fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=7.8, framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25)

    # Panel B: confusion matrix for Order-CNN on unseen_speed_profile
    cnn_res = lookup[("unseen_speed_profile", "order_cnn_onnx")]
    cm = np.asarray(cnn_res["confusion"], dtype=np.float64)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    ax2.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            c = "white" if norm[i, j] > 0.58 else "#0f172a"
            ax2.text(
                j,
                i,
                f"{norm[i, j]:.0%}\n{int(cm[i, j])}",
                ha="center",
                va="center",
                color=c,
                fontsize=7.8,
            )
    ticks = list(range(len(CLASSES)))
    ax2.set_xticks(ticks, CLASS_TITLES, rotation=22, ha="right", fontsize=8)
    ax2.set_yticks(ticks, CLASS_TITLES, fontsize=8)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title(
        f"Order-CNN (ONNX) on Unseen Speed Profiles\n"
        f"acc {float(cnn_res['accuracy']):.1%} | macro F1 {float(cnn_res['macro_f1']):.3f}",
        fontsize=9.5,
        fontweight="bold",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def run_closed_loop_replay(
    raw_dir: Path,
    onnx_path: Path,
    meta_path: Path,
    *,
    stems: tuple[str, ...] = ("H-A-3", "I-C-3", "C-D-3"),
) -> list[ActuatorCommand]:
    """Run held-out Trial-3 recordings sequentially through ONNX + SupervisoryController."""
    from .edge import EdgeClassifier

    edge = EdgeClassifier(onnx_path, meta_path)
    ctrl = SupervisoryController()
    commands: list[ActuatorCommand] = []
    time_offset = 0.0

    for stem in stems:
        rec = load_recording(raw_dir, stem)
        track = decode_encoder(rec.encoder, rec.fs_hz)
        audit = audit_sensors(rec.vibration, track)
        _, order_sigs = multiband_order_envelopes(rec.vibration, rec.fs_hz, track)

        band_windows: list[NDArray[np.float64]] = []
        times_s = np.empty(0, dtype=np.float64)
        win_rpms = np.empty(0, dtype=np.float64)
        for osig in order_sigs:
            wins, ts, wrpms = slice_order_windows(osig)
            band_windows.append(wins)
            times_s = ts
            win_rpms = wrpms

        n_win = len(band_windows[0])
        x_cnn = np.stack(
            [extract_multiband_cnn_input([bw[k] for bw in band_windows]) for k in range(n_win)]
        )
        probs = edge.predict_proba(x_cnn)

        for p_row, t_s, rpm in zip(probs, times_s, win_rpms, strict=True):
            cmd = ctrl.update(
                p_row.tolist(),
                audit,
                time_s=time_offset + float(t_s),
                rpm=float(rpm),
            )
            commands.append(cmd)
        time_offset += float(len(rec.vibration)) / rec.fs_hz

    return commands


def plot_closed_loop_replay(
    commands: list[ActuatorCommand],
    out_path: Path,
    *,
    segment_labels: tuple[str, ...] = (
        "Healthy\n(H-A-3, accel)",
        "Inner Fault\n(I-C-3, inc-dec)",
        "Combined Fault\n(C-D-3, dec-inc)",
    ),
) -> None:
    """Plot end-to-end replay: decoded RPM -> ONNX fault prob -> actuator derating."""
    t = np.array([c.time_s for c in commands])
    rpm = np.array([c.rpm for c in commands])
    fp = np.array([c.fault_prob for c in commands])
    vel = np.array([c.velocity_scale for c in commands])
    trq = np.array([c.torque_limit for c in commands])
    modes = [c.mode for c in commands]

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 6.8), dpi=160, sharex=True)
    mode_colors = {"NOMINAL": "#dcfce7", "DERATE": "#fef3c7", "SAFE_STOP": "#fee2e2"}

    for ax in axes:
        for i in range(len(commands)):
            t0 = t[i] - 0.35 if i == 0 else 0.5 * (t[i - 1] + t[i])
            t1 = t[i] + 0.35 if i == len(commands) - 1 else 0.5 * (t[i] + t[i + 1])
            ax.axvspan(t0, t1, color=mode_colors[modes[i]], alpha=0.55, lw=0)
        for boundary in (10.0, 20.0):
            ax.axvline(boundary, color="#334155", ls="--", lw=1.0, alpha=0.7)
        ax.grid(alpha=0.25)

    axes[0].plot(t, rpm, color="#0f766e", lw=1.8)
    axes[0].set_ylabel("Encoder RPM")
    axes[0].set_title(
        "Held-Out Trial-3 Closed-Loop Replay (ONNX Edge + Supervisory Controller)",
        fontsize=10.5,
        fontweight="bold",
    )
    rpm_top = float(np.max(rpm))
    for idx, label in enumerate(segment_labels):
        axes[0].text(
            5.0 + 10.0 * idx,
            rpm_top * 0.97,
            label,
            ha="center",
            va="top",
            fontsize=8.2,
            fontweight="bold",
            color="#1e293b",
        )

    axes[1].plot(
        t,
        fp,
        color="#1d4ed8",
        lw=1.8,
        marker="o",
        ms=3.5,
        label="Fault probability (1 - P(healthy))",
    )
    axes[1].axhline(0.55, color="#d97706", ls=":", lw=1.1, label="Derate threshold (0.55)")
    axes[1].axhline(0.85, color="#dc2626", ls=":", lw=1.1, label="Stop threshold (0.85)")
    axes[1].set_ylabel("P(fault)")
    axes[1].set_ylim(-0.05, 1.08)
    axes[1].legend(loc="center left", fontsize=7.8, framealpha=0.9)

    axes[2].step(t, vel, where="mid", color="#0f172a", lw=2.0, label="velocity_scale")
    axes[2].step(t, trq, where="mid", color="#7c3aed", lw=1.6, ls="--", label="torque_limit")
    axes[2].set_ylabel("Actuator Scale")
    axes[2].set_xlabel("Stream Time (s)")
    axes[2].set_ylim(-0.08, 1.12)
    legend_patches = [
        Patch(facecolor=mode_colors["NOMINAL"], label="NOMINAL"),
        Patch(facecolor=mode_colors["DERATE"], label="DERATE"),
        Patch(facecolor=mode_colors["SAFE_STOP"], label="SAFE_STOP"),
    ]
    h1, l1 = axes[2].get_legend_handles_labels()
    axes[2].legend(
        h1 + legend_patches,
        [*l1, "NOMINAL", "DERATE", "SAFE_STOP"],
        loc="lower left",
        ncol=5,
        fontsize=7.8,
        framealpha=0.92,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
