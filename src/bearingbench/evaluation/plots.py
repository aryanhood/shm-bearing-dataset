"""Publication-grade vector and 300-DPI figure generation from verified artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..data.encoder import decode_encoder
from ..data.kinematics import ER16K
from ..data.signal_ops import (
    CARRIER_BANDS_HZ,
    multiband_order_envelopes,
    order_envelope_spectrum,
    order_first_cot_envelopes,
    slice_order_windows,
    time_envelope_fft,
)
from ..data.uottawa import load_uottawa_recording


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#cbd5e1",
        }
    )


def write_pipeline_svg(out_path: Path) -> None:
    """Write a crisp vector SVG schematic of the BearingBench pipeline."""
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 265" width="980" height="265">
  <defs>
    <style>
      .bg { fill: #f8fafc; }
      .box { fill: #ffffff; stroke: #334155; stroke-width: 1.6; rx: 8; }
      .accent1 { fill: #eff6ff; stroke: #1d4ed8; stroke-width: 1.8; rx: 8; }
      .accent2 { fill: #ecfdf5; stroke: #047857; stroke-width: 1.8; rx: 8; }
      .accent3 { fill: #fffbeb; stroke: #b45309; stroke-width: 1.8; rx: 8; }
      .title {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 12.5px; font-weight: 700; fill: #0f172a;
      }
      .sub { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5px; fill: #334155; }
      .hdr {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 13.5px; font-weight: 700; fill: #0f172a;
      }
      .arrow { stroke: #475569; stroke-width: 1.8; fill: none; marker-end: url(#ah); }
    </style>
    <marker id="ah" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#475569"/>
    </marker>
  </defs>
  <rect width="980" height="265" class="bg"/>
  <text x="24" y="28" class="hdr">
    BearingBench: Physics-Informed COT + Matryoshka 1D-CNN + Conformal INT8 Supervisor
  </text>

  <rect x="24" y="52" width="170" height="175" class="box"/>
  <text x="109" y="78" text-anchor="middle" class="title">1. Multi-Sensor Input</text>
  <text x="109" y="103" text-anchor="middle" class="sub">200 kHz Accelerometer</text>
  <text x="109" y="121" text-anchor="middle" class="sub">+ 1024-PPR Encoder</text>
  <line x1="38" y1="138" x2="180" y2="138" stroke="#cbd5e1"/>
  <text x="109" y="158" text-anchor="middle" class="sub">Schmitt Trigger +</text>
  <text x="109" y="175" text-anchor="middle" class="sub">Sub-Sample Edge Interp.</text>
  <text x="109" y="195" text-anchor="middle" class="sub">Sensor Health Audit</text>
  <text x="109" y="213" text-anchor="middle" class="sub">(Clipping / Dropout)</text>

  <path d="M194 140 L216 140" class="arrow"/>

  <rect x="220" y="52" width="215" height="175" class="accent1"/>
  <text x="327" y="78" text-anchor="middle" class="title">2. Physics Inductive Bias</text>
  <text x="327" y="103" text-anchor="middle" class="sub">4 Structural Carrier Bands</text>
  <text x="327" y="121" text-anchor="middle" class="sub">(0.5-5, 5-15, 15-35, 35-65 kHz)</text>
  <line x1="234" y1="138" x2="420" y2="138" stroke="#bfdbfe"/>
  <text x="327" y="158" text-anchor="middle" class="sub">Time-Domain Hilbert Sq. Env</text>
  <text x="327" y="176" text-anchor="middle" class="sub">-&gt; PCHIP/Cubic Angle COT</text>
  <text x="327" y="196" text-anchor="middle" class="sub">-&gt; Gain-Invariant Order-SES</text>
  <text x="327" y="214" text-anchor="middle" class="sub">Tensor Shape: (4, 400)</text>

  <path d="M435 140 L457 140" class="arrow"/>

  <rect x="461" y="52" width="235" height="175" class="accent2"/>
  <text x="578" y="78" text-anchor="middle" class="title">3. Matryoshka Edge Backbone</text>
  <text x="578" y="103" text-anchor="middle" class="sub">3-Block 1D-CNN (~24.8K params)</text>
  <text x="578" y="121" text-anchor="middle" class="sub">Nested D in {8, 16, 32, 64, 128}</text>
  <line x1="475" y1="138" x2="682" y2="138" stroke="#a7f3d0"/>
  <text x="578" y="158" text-anchor="middle" class="sub">Elastic Prefix Slicing</text>
  <text x="578" y="176" text-anchor="middle" class="sub">Static QDQ INT8 ONNX Export</text>
  <text x="578" y="196" text-anchor="middle" class="sub">Zero-PyTorch CPU Runtime</text>
  <text x="578" y="214" text-anchor="middle" class="sub">(~36-125 KB, Sub-ms Latency)</text>

  <path d="M696 140 L718 140" class="arrow"/>

  <rect x="722" y="52" width="234" height="175" class="accent3"/>
  <text x="839" y="78" text-anchor="middle" class="title">4. Calibrated Conformal Guard</text>
  <text x="839" y="103" text-anchor="middle" class="sub">OOF Temp. Scaling + Ensemble</text>
  <text x="839" y="121" text-anchor="middle" class="sub">Split APS / Online ACI (90% Target)</text>
  <line x1="736" y1="138" x2="942" y2="138" stroke="#fde68a"/>
  <text x="839" y="158" text-anchor="middle" class="sub">Ambiguity Set |C(x)| &gt; 1 Trigger</text>
  <text x="839" y="176" text-anchor="middle" class="sub">3-of-4 Hysteresis State Machine</text>
  <text x="839" y="196" text-anchor="middle" class="sub">NOMINAL -&gt; DERATE -&gt; SAFE_STOP</text>
  <text x="839" y="214" text-anchor="middle" class="sub">Slew-Limited Torque / Velocity</text>
</svg>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")


def plot_physics_order_vs_time(uottawa_dir: Path, out_path: Path) -> None:
    """Plot Fig 2: Real O-C-1 speed ramp vs Time-Env, Order-First & Envelope-First COT."""
    _style()
    rec = load_uottawa_recording(uottawa_dir, "O-C-1")
    track = decode_encoder(rec.encoder, rec.fs_hz)
    time_envs, env_first_sigs = multiband_order_envelopes(rec.vibration, rec.fs_hz, track)
    order_first_sigs = order_first_cot_envelopes(
        rec.vibration, rec.fs_hz, track, bands_hz=CARRIER_BANDS_HZ
    )

    # Use carrier band 1 (5-15 kHz) which strongly carries housing structural impact modulation
    b_idx = 1
    ef_wins, _, _ = slice_order_windows(env_first_sigs[b_idx])
    of_wins, _, _ = slice_order_windows(order_first_sigs[b_idx])

    mid = len(ef_wins) // 2
    orders, ef_spec = order_envelope_spectrum(ef_wins[mid])
    _, of_spec = order_envelope_spectrum(of_wins[mid])
    ef_spec = ef_spec / (float(np.median(ef_spec[orders >= 0.25])) + 1e-12)
    of_spec = of_spec / (float(np.median(of_spec[orders >= 0.25])) + 1e-12)

    seg_len = len(time_envs[b_idx]) // len(ef_wins)
    f_hz, t_spec = time_envelope_fft(
        time_envs[b_idx][mid * seg_len : (mid + 1) * seg_len], rec.fs_hz, max_hz=350.0
    )
    t_spec = t_spec / (float(np.median(t_spec[f_hz >= 5.0])) + 1e-12)

    fig = plt.figure(figsize=(10.5, 6.2), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.35], hspace=0.38, wspace=0.25)

    ax_rpm = fig.add_subplot(gs[0, :])
    ax_rpm.plot(track.rpm_times_s, track.rpm_values, color="#1d4ed8", linewidth=1.8)
    ax_rpm.set_title("uOttawa O-C-1 (Outer-Race Defect, Non-Stationary Inc-Dec Speed Profile)")
    ax_rpm.set_xlabel("Time (s)")
    ax_rpm.set_ylabel("Instantaneous Speed (RPM)")

    ax_t = fig.add_subplot(gs[1, 0])
    ax_t.plot(f_hz, np.log1p(t_spec), color="#b45309", linewidth=1.4)
    ax_t.set_title("Time-Domain Envelope FFT (Smeared Peaks)")
    ax_t.set_xlabel("Envelope Frequency (Hz)")
    ax_t.set_ylabel("log1p(SNR over Median Floor)")
    ax_t.set_xlim(5.0, 320.0)

    ax_o = fig.add_subplot(gs[1, 1])
    bpfo = ER16K.orders().bpfo
    ax_o.plot(
        orders,
        np.log1p(of_spec),
        color="#64748b",
        linewidth=1.2,
        alpha=0.85,
        label="Order-First COT (Resonance Shift)",
    )
    ax_o.plot(
        orders, np.log1p(ef_spec), color="#047857", linewidth=1.6, label="Envelope-First COT (Ours)"
    )
    for h in (1, 2, 3):
        ax_o.axvline(h * bpfo, color="#dc2626", linestyle=":", linewidth=1.1, alpha=0.75)
        ax_o.text(
            h * bpfo + 0.25,
            ax_o.get_ylim()[1] * 0.85 if h == 1 else 2.2,
            f"{h}x BPFO",
            color="#dc2626",
            fontsize=7.5,
        )
    ax_o.set_title("Order-Domain SES (Locked to Kinematic Orders)")
    ax_o.set_xlabel("Shaft Order (cycles / rev)")
    ax_o.set_ylabel("log1p(SNR over Median Floor)")
    ax_o.set_xlim(0.25, 16.0)
    ax_o.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_dual_dataset_benchmark(results: dict[str, Any], out_path: Path) -> None:
    """Plot Fig 3: 5-seed accuracy +/- std across CWRU and uOttawa protocols."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=300)

    cw_protos = ["random_window", "unseen_load", "unseen_fault_size"]
    cw_labels = [
        "Random Window\n(Leakage Trap)",
        "Unseen Motor Load\n(0-3 HP OOD)",
        "Unseen Fault Size\n(7/14/21 mil OOD)",
    ]
    cw_models = [
        ("gain_sensitive_mlp", "Gain-Sens. 16-d MLP", "#94a3b8"),
        ("gain_invariant_rf", "Gain-Inv. 27-d RF", "#3b82f6"),
        ("cwru_env_cnn", "Env 1D-CNN", "#f59e0b"),
        ("cwru_mrl_cnn", "MRL Env 1D-CNN", "#047857"),
    ]
    x1 = np.arange(len(cw_protos))
    w1 = 0.19
    for idx, (mkey, label, col) in enumerate(cw_models):
        means = [100.0 * results["cwru"]["protocols"][p][mkey]["accuracy_mean"] for p in cw_protos]
        stds = [100.0 * results["cwru"]["protocols"][p][mkey]["accuracy_std"] for p in cw_protos]
        ax1.bar(
            x1 + (idx - 1.5) * w1,
            means,
            width=w1,
            yerr=stds,
            capsize=3,
            label=label,
            color=col,
            alpha=0.92,
        )
    ax1.set_xticks(x1, cw_labels, fontsize=8.5)
    ax1.set_ylim(35.0, 105.0)
    ax1.set_ylabel("Accuracy (% ± 1 SD, 5 seeds)")
    ax1.set_title("CWRU (40 Recordings): Gain Shift & Severity Wall")
    ax1.legend(loc="lower left", fontsize=7.8)

    uo_protos = ["random_window", "unseen_trial", "unseen_speed_profile"]
    uo_labels = [
        "Random Window\n(Leakage Trap)",
        "Unseen Trial\n(Session OOD)",
        "Unseen Speed Profile\n(A/B <-> C/D OOD)",
    ]
    uo_models = [
        ("time_env_rf", "Time-Env RF", "#94a3b8"),
        ("time_env_cnn", "Time-Env 1D-CNN", "#f97316"),
        ("order_physics_rf", "Order-Physics RF", "#3b82f6"),
        ("order_cnn", "Order 1D-CNN", "#0ea5e9"),
        ("order_mrl_cnn", "Order-MRL 1D-CNN", "#047857"),
    ]
    x2 = np.arange(len(uo_protos))
    w2 = 0.16
    for idx, (mkey, label, col) in enumerate(uo_models):
        means = [
            100.0 * results["uottawa"]["protocols"][p][mkey]["accuracy_mean"] for p in uo_protos
        ]
        stds = [100.0 * results["uottawa"]["protocols"][p][mkey]["accuracy_std"] for p in uo_protos]
        ax2.bar(
            x2 + (idx - 2.0) * w2,
            means,
            width=w2,
            yerr=stds,
            capsize=2.5,
            label=label,
            color=col,
            alpha=0.92,
        )
    ax2.set_xticks(x2, uo_labels, fontsize=8.5)
    ax2.set_ylim(75.0, 102.0)
    ax2.set_ylabel("Accuracy (% ± 1 SD, 5 seeds)")
    ax2.set_title("uOttawa (60 Recordings): Non-Stationary Regime Shift")
    ax2.legend(loc="lower left", fontsize=7.6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_ablations_and_pareto(results: dict[str, Any], out_path: Path) -> None:
    """Plot Fig 4: Factorial ablations, tachometer degradation, and Matryoshka INT8 Pareto."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=300)

    abl = results["uottawa"]["ablations_unseen_speed_profile"]
    items = [
        ("envelope_first_4band", "Envelope-First 4-Band (Ours)"),
        ("cpw_enabled_4band", "+ Cepstrum Pre-Whitening"),
        ("order_first_4band", "Order-First COT (Resonance Shift)"),
        ("single_band_5_15khz", "Single Band (5-15 kHz only)"),
        ("single_band_0_5_5khz", "Single Band (0.5-5 kHz only)"),
        ("tach_256ppr", "Encoder 256 PPR"),
        ("tach_64ppr", "Encoder 64 PPR"),
        ("tach_32ppr", "Encoder 32 PPR"),
        ("tach_1024ppr_2pct_jitter", "1024 PPR + 2% Edge Jitter"),
    ]
    labels = [lbl for _, lbl in items][::-1]
    means = [100.0 * abl[k]["accuracy_mean"] for k, _ in items][::-1]
    stds = [100.0 * abl[k]["accuracy_std"] for k, _ in items][::-1]
    colors = [
        "#047857"
        if i == len(items) - 1
        else ("#dc2626" if "order_first" in items[::-1][i][0] else "#3b82f6")
        for i in range(len(items))
    ]

    y_pos = np.arange(len(items))
    ax1.barh(y_pos, means, xerr=stds, capsize=3, color=colors, alpha=0.88)
    ax1.set_yticks(y_pos, labels, fontsize=8.3)
    ax1.set_xlim(min(60.0, min(means) - 5.0), 101.0)
    ax1.set_xlabel("Unseen Speed Profile Accuracy (% ± 1 SD, 5 seeds)")
    ax1.set_title("Factorial Signal-Processing & Tachometer Ablations")

    pareto = results["uottawa"]["mrl_onnx_pareto"]
    dims = [int(d) for d in results["meta"]["nested_dims"]]
    fp32_kb = [pareto[str(d)]["fp32"]["size_kb"] for d in dims]
    fp32_acc = [100.0 * pareto[str(d)]["fp32"]["accuracy_mean"] for d in dims]
    fp32_std = [100.0 * pareto[str(d)]["fp32"]["accuracy_std"] for d in dims]
    int8_kb = [pareto[str(d)]["int8"]["size_kb"] for d in dims]
    int8_acc = [100.0 * pareto[str(d)]["int8"]["accuracy_mean"] for d in dims]
    int8_std = [100.0 * pareto[str(d)]["int8"]["accuracy_std"] for d in dims]

    ax2.errorbar(
        fp32_kb,
        fp32_acc,
        yerr=fp32_std,
        marker="o",
        capsize=3,
        linewidth=1.8,
        color="#1d4ed8",
        label="ONNX FP32 (Matryoshka d=8..128)",
    )
    ax2.errorbar(
        int8_kb,
        int8_acc,
        yerr=int8_std,
        marker="s",
        capsize=3,
        linewidth=1.8,
        color="#047857",
        label="ONNX Static QDQ INT8 (d=8..128)",
    )
    for d, x_v, y_v in zip(dims, int8_kb, int8_acc, strict=True):
        ax2.annotate(
            f"d={d}",
            (x_v, y_v),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=7.8,
            color="#065f46",
        )
    for d, x_v, y_v in zip(dims, fp32_kb, fp32_acc, strict=True):
        ax2.annotate(
            f"d={d}",
            (x_v, y_v),
            textcoords="offset points",
            xytext=(-18, 6),
            fontsize=7.5,
            color="#1e40af",
        )
    ax2.set_xlabel("ONNX Model Artifact Size (KB)")
    ax2.set_ylabel("Unseen Speed Profile Accuracy (%)")
    ax2.set_title("Matryoshka Nested Widths x Static INT8 Pareto Frontier")
    ax2.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_and_conformal(results: dict[str, Any], out_path: Path) -> None:
    """Plot Fig 5: Reliability diagrams under shift + Online ACI vs Split APS rolling coverage."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5), dpi=300)

    unc_sp = results["uottawa"]["uncertainty_and_conformal"]["unseen_speed_profile"]
    unc_rw = results["uottawa"]["uncertainty_and_conformal"]["random_window"]

    categories = ["Raw Single", "OOF TempScale", "5-Seed Ensemble"]
    iid_ece = [
        unc_rw["raw_single"]["ece_mean"],
        unc_rw["temp_scaled_single"]["ece_mean"],
        unc_rw["ensemble_5seed"]["ece"],
    ]
    ood_ece = [
        unc_sp["raw_single"]["ece_mean"],
        unc_sp["temp_scaled_single"]["ece_mean"],
        unc_sp["ensemble_5seed"]["ece"],
    ]

    x = np.arange(len(categories))
    w = 0.34
    ax1.bar(
        x - w / 2,
        [100.0 * v for v in iid_ece],
        width=w,
        color="#64748b",
        label="i.i.d. Random Window",
    )
    ax1.bar(
        x + w / 2,
        [100.0 * v for v in ood_ece],
        width=w,
        color="#dc2626",
        alpha=0.85,
        label="OOD Unseen Speed Profile",
    )
    for idx, (vi, vo) in enumerate(zip(iid_ece, ood_ece, strict=True)):
        ax1.text(idx - w / 2, 100.0 * vi + 0.15, f"{100 * vi:.2f}%", ha="center", fontsize=8)
        ax1.text(idx + w / 2, 100.0 * vo + 0.15, f"{100 * vo:.2f}%", ha="center", fontsize=8)
    ax1.set_xticks(x, categories)
    ax1.set_ylabel("Expected Calibration Error (% ECE, Lower is Better)")
    ax1.set_title("Calibration Under Regime Shift (ECE)")
    ax1.legend(loc="upper right", fontsize=8.2)

    aci_data = results["uottawa"]["plot_data"]["aci"]
    roll = aci_data["rolling_aci_coverage"]
    ax2.plot(
        range(len(roll)),
        [100.0 * v for v in roll],
        color="#047857",
        linewidth=1.7,
        label=f"Online ACI Rolling (Overall {100 * unc_sp['online_aci']['coverage_mean']:.1f}%)",
    )
    ax2.axhline(
        90.0, color="#0f172a", linestyle="--", linewidth=1.2, label="Target Coverage (90.0%)"
    )
    ax2.axhline(
        100.0 * unc_sp["split_aps"]["coverage_mean"],
        color="#b45309",
        linestyle=":",
        linewidth=1.6,
        label=f"Static Split APS Mean ({100 * unc_sp['split_aps']['coverage_mean']:.1f}%)",
    )
    ax2.set_ylim(60.0, 102.0)
    ax2.set_xlabel("Chronological Test Window Step (Unseen Speed Profiles)")
    ax2.set_ylabel("Empirical Coverage (%)")
    ax2.set_title("Split APS vs Online ACI Under Non-Stationary Shift")
    ax2.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_closed_loop_replay(replay_json: Path, out_path: Path) -> None:
    """Plot Fig 6: Trial-3 closed-loop INT8 + Conformal Supervisory Hysteresis trace."""
    _style()
    payload = cast(dict[str, Any], json.loads(replay_json.read_text(encoding="utf-8")))
    rows = cast(list[dict[str, Any]], payload["timeline"])
    ts = [float(r["time_s"]) for r in rows]
    rpms = [float(r["rpm"]) for r in rows]
    p_fault = [float(r["fault_prob"]) for r in rows]
    vel = [float(r["velocity_scale"]) for r in rows]
    csize = [int(r["conformal_set_size"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 5.2), dpi=300, sharex=True)
    ax1.plot(
        ts,
        p_fault,
        color="#dc2626",
        marker="o",
        markersize=3.5,
        linewidth=1.6,
        label="P(fault) INT8 Calibrated",
    )
    ax1_r = ax1.twinx()
    ax1_r.plot(
        ts, rpms, color="#1d4ed8", linestyle="--", alpha=0.75, linewidth=1.3, label="Shaft RPM"
    )
    ax1_r.set_ylabel("Shaft RPM", color="#1d4ed8")
    ax1.axvline(10.0, color="#475569", linestyle="-.", linewidth=1.2)
    ax1.text(10.15, 0.9, "Combined Fault\nInjected (C-C-3)", fontsize=8, color="#0f172a")
    ax1.set_ylim(-0.05, 1.08)
    ax1.set_ylabel("Fault Probability")
    ax1.set_title("Closed-Loop Trial-3 Edge Replay (H-C-3 Healthy -> C-C-3 Combined Fault)")
    ax1.legend(loc="upper left", fontsize=8)

    ax2.step(ts, vel, where="post", color="#047857", linewidth=2.0, label="Actuator Velocity Scale")
    ax2_r = ax2.twinx()
    ax2_r.bar(ts, csize, width=0.35, color="#f59e0b", alpha=0.35, label="Conformal Set Size |C(x)|")
    ax2_r.set_ylabel("Conformal Set Size |C(x)|", color="#b45309")
    ax2_r.set_ylim(0, 5.5)
    ax2.axvline(10.0, color="#475569", linestyle="-.", linewidth=1.2)
    ax2.set_ylim(-0.05, 1.12)
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_ylabel("Velocity Scale (1.0 -> 0.0)")
    ax2.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(repo_root: Path) -> None:
    """Generate all 6 publication figures from artifacts/benchmark_results.json and real data."""
    assets_dir = repo_root / "assets"
    artifacts_dir = repo_root / "artifacts"
    uottawa_dir = repo_root / "data" / "uottawa"

    results = cast(
        dict[str, Any],
        json.loads((artifacts_dir / "benchmark_results.json").read_text(encoding="utf-8")),
    )
    write_pipeline_svg(assets_dir / "fig1_pipeline_architecture.svg")
    plot_physics_order_vs_time(uottawa_dir, assets_dir / "fig2_physics_order_vs_time.png")
    plot_dual_dataset_benchmark(results, assets_dir / "fig3_dual_dataset_benchmark.png")
    plot_ablations_and_pareto(results, assets_dir / "fig4_ablations_and_mrl_pareto.png")
    plot_calibration_and_conformal(results, assets_dir / "fig5_calibration_and_conformal.png")
    plot_closed_loop_replay(
        artifacts_dir / "replay_trace.json", assets_dir / "fig6_closed_loop_supervisor.png"
    )
    print(f"Generated all 6 figures in {assets_dir}", flush=True)
