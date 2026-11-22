"""Plain-English terminal summaries and self-contained single-file HTML reports."""

from __future__ import annotations

import base64
import html
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .compare import ComparisonResult
from .diagnostics import DiagnosisResult

STATUS_COLORS: dict[str, tuple[str, str]] = {
    "HEALTHY": ("#065f46", "#d1fae5"),
    "WATCH": ("#92400e", "#fef3c7"),
    "INSPECT SOON": ("#9a3412", "#ffedd5"),
    "CRITICAL": ("#991b1b", "#fee2e2"),
}


def format_terminal_report(res: DiagnosisResult) -> str:
    """Format a clean, readable plain-English summary for the terminal."""
    lines = [
        "=" * 68,
        f"  VIBCHECK HEALTH REPORT   [{res.status}]",
        "=" * 68,
        f"  File       : {res.source_name} ({res.channel}, {res.fs_hz:.0f} Hz, {res.duration_s:.2f} s)",
        f"  Summary    : {res.summary}",
        "",
        "  Key Measurements:",
        f"    Overall RMS     : {res.rms:.4f} {res.units}   (Peak: {res.peak:.4f})",
        f"    Impulsiveness   : kurtosis {res.kurtosis:.2f}  |  crest factor {res.crest_factor:.2f}",
    ]
    if res.iso is not None:
        lines.append(
            f"    ISO 20816-3     : {res.iso.velocity_rms_mm_s:.2f} mm/s RMS "
            f"(Zone {res.iso.zone}, {res.iso.machine_class})"
        )
    if res.baseline_drift is not None:
        lines.append(
            f"    Baseline Drift  : {res.baseline_drift.drift_score:.0f}/100 "
            f"vs '{res.baseline_drift.machine_name}' (RMS {res.baseline_drift.rms_ratio:.1f}x)"
        )
    if res.rpm is not None:
        b_txt = f" | Bearing: {res.bearing.code}" if res.bearing is not None else ""
        lines.append(f"    Operating Point : {res.rpm:.0f} RPM ({res.rpm / 60.0:.1f} Hz){b_txt}")

    lines.append("")
    lines.append("  What We Found:")
    for f in res.findings:
        lines.append(f"    - {f}")

    lines.append("")
    lines.append("  What To Check Next:")
    for s in res.next_steps:
        lines.append(f"    * {s}")
    lines.append("=" * 68)
    return "\n".join(lines)


def render_diagnostic_figure(res: DiagnosisResult, out_path: Path | None = None) -> bytes:
    """Render a 3-panel diagnostic figure (waveform, FFT, envelope spectrum) as PNG bytes."""
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 7.2), dpi=150)

    # Panel 1: time-domain waveform preview
    t = np.arange(len(res.waveform), dtype=np.float64) / res.fs_hz
    axes[0].plot(t, res.waveform, color="#1d4ed8", lw=0.8)
    axes[0].set_title(
        f"{res.source_name} — {res.status}  "
        f"(RMS={res.rms:.4f} {res.units}, Kurtosis={res.kurtosis:.1f})",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(f"Amplitude ({res.units})")
    axes[0].grid(alpha=0.25)

    # Panel 2: frequency spectrum with 1x / 2x shaft markers
    axes[1].plot(res.fft_freqs, res.fft_mag, color="#0f766e", lw=1.1)
    if res.rpm is not None and res.rpm > 0.0:
        f_shaft = res.rpm / 60.0
        for h, lbl in ((1, "1x RPM"), (2, "2x RPM")):
            if h * f_shaft <= float(np.max(res.fft_freqs)):
                axes[1].axvline(h * f_shaft, color="#d97706", ls="--", lw=1.0, alpha=0.8)
                axes[1].text(
                    h * f_shaft + 4.0,
                    float(np.max(res.fft_mag)) * (0.88 - 0.12 * (h - 1)),
                    lbl,
                    color="#b45309",
                    fontsize=8,
                )
    axes[1].set_title(
        "Frequency Spectrum (Unbalance & Misalignment Check)", fontsize=9.8, fontweight="bold"
    )
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(alpha=0.25)

    # Panel 3: Hilbert squared-envelope spectrum with bearing fault markers
    env_max = float(np.max(res.env_mag)) if len(res.env_mag) > 0 else 1.0
    env_norm = res.env_mag / (env_max + 1e-12)
    axes[2].plot(res.env_freqs, env_norm, color="#334155", lw=1.1)
    if res.rpm is not None and res.bearing is not None:
        fmap = res.bearing.frequencies_hz(res.rpm)
        markers = [
            ("BPFO", fmap["bpfo"], "#2563eb", 0.92),
            ("2xBSF", fmap["bsf_2x"], "#7c3aed", 0.78),
            ("BPFI", fmap["bpfi"], "#dc2626", 0.92),
            ("2xBPFI", 2.0 * fmap["bpfi"], "#dc2626", 0.84),
            ("3xBPFI", 3.0 * fmap["bpfi"], "#dc2626", 0.76),
        ]
        max_f = float(np.max(res.env_freqs)) if len(res.env_freqs) > 0 else 0.0
        for lbl, fhz, col, y_pos in markers:
            if 0.0 < fhz <= max_f:
                axes[2].axvline(fhz, color=col, ls="--", lw=1.0, alpha=0.8)
                x_off = -4.0 if lbl == "BPFO" else 4.0
                ha = "right" if lbl == "BPFO" else "left"
                axes[2].text(
                    fhz + x_off,
                    y_pos,
                    f"{lbl} ({fhz:.0f} Hz)",
                    color=col,
                    fontsize=7.8,
                    ha=ha,
                )
    lo, hi = res.envelope_band_hz
    axes[2].set_title(
        f"Envelope Spectrum — Repeating Bearing Impacts (Carrier Band {lo:.0f}-{hi:.0f} Hz)",
        fontsize=9.8,
        fontweight="bold",
    )
    axes[2].set_xlabel("Impact Repetition Frequency (Hz)")
    axes[2].set_ylabel("Normalized Magnitude")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(buf.getvalue())
    plt.close(fig)
    return buf.getvalue()


def write_html_report(res: DiagnosisResult, path: Path) -> Path:
    """Write a standalone, shareable single-file HTML diagnostic report."""
    png_b64 = base64.b64encode(render_diagnostic_figure(res)).decode("ascii")
    fg, bg = STATUS_COLORS[res.status]

    iso_val = (
        f"{res.iso.velocity_rms_mm_s:.2f} mm/s (Zone {res.iso.zone})"
        if res.iso is not None
        else "N/A (audio)"
    )
    drift_val = (
        f"{res.baseline_drift.drift_score:.0f} / 100"
        if res.baseline_drift is not None
        else "No baseline"
    )
    findings_html = "".join(f"<li>{html.escape(f)}</li>" for f in res.findings)
    steps_html = "".join(f"<li>{html.escape(s)}</li>" for s in res.next_steps)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>vibcheck Report — {html.escape(res.source_name)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f8fafc; color: #0f172a; margin: 0; padding: 28px; }}
  .wrap {{ max-width: 960px; margin: 0 auto; background: #fff; border-radius: 12px;
           box-shadow: 0 2px 14px rgba(15,23,42,0.08); padding: 28px 32px; }}
  .head {{ display: flex; justify-content: space-between; align-items: center;
           border-bottom: 1px solid #e2e8f0; padding-bottom: 16px; gap: 16px; }}
  .badge {{ font-weight: 700; padding: 8px 16px; border-radius: 999px;
            color: {fg}; background: {bg}; font-size: 14px; white-space: nowrap; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
  .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px; }}
  .card .lbl {{ font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }}
  .card .val {{ font-size: 18px; font-weight: 700; margin-top: 4px; color: #0f172a; }}
  h2 {{ font-size: 16px; margin: 22px 0 10px; color: #1e293b; }}
  ul {{ margin: 0; padding-left: 20px; line-height: 1.65; }}
  img {{ width: 100%; border: 1px solid #e2e8f0; border-radius: 8px; margin-top: 8px; }}
  .meta {{ font-size: 13px; color: #64748b; margin-top: 4px; }}
  .foot {{ margin-top: 24px; font-size: 12px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 12px; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="head">
    <div>
      <h1 style="margin:0;font-size:22px;">Machine Vibration Health Report</h1>
      <div class="meta">{html.escape(res.source_name)} &middot; {html.escape(res.channel)} &middot;
        {res.fs_hz:.0f} Hz &middot; {res.duration_s:.2f} s</div>
    </div>
    <div class="badge">{html.escape(res.status)}</div>
  </div>
  <p style="font-size:15px;margin:16px 0 0;"><strong>Summary:</strong> {html.escape(res.summary)}</p>
  <div class="cards">
    <div class="card"><div class="lbl">Overall RMS</div><div class="val">{res.rms:.4f} {html.escape(res.units)}</div></div>
    <div class="card"><div class="lbl">ISO 20816-3</div><div class="val">{html.escape(iso_val)}</div></div>
    <div class="card"><div class="lbl">Impulsiveness</div><div class="val">Kurtosis {res.kurtosis:.1f}</div></div>
    <div class="card"><div class="lbl">Baseline Drift</div><div class="val">{html.escape(drift_val)}</div></div>
  </div>
  <h2>What We Found</h2>
  <ul>{findings_html}</ul>
  <h2>What To Check Next</h2>
  <ul>{steps_html}</ul>
  <h2>Diagnostic Spectra</h2>
  <img src="data:image/png;base64,{png_b64}" alt="Vibration diagnostic spectra">
  <div class="foot">Generated locally by vibcheck (ISO 20816-3 + Hilbert envelope analysis). No cloud upload.</div>
</div>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path


def write_comparison_html(comp: ComparisonResult, path: Path) -> Path:
    """Write a standalone before-vs-after repair HTML comparison report."""
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 5.4), dpi=150)
    axes[0].plot(
        comp.before.fft_freqs, comp.before.fft_mag, color="#dc2626", lw=1.1, label="Before"
    )
    axes[0].plot(comp.after.fft_freqs, comp.after.fft_mag, color="#059669", lw=1.1, label="After")
    axes[0].set_title("Frequency Spectrum — Before vs After", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Magnitude")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        comp.before.env_freqs, comp.before.env_mag, color="#dc2626", lw=1.1, label="Before"
    )
    axes[1].plot(comp.after.env_freqs, comp.after.env_mag, color="#059669", lw=1.1, label="After")
    axes[1].set_title("Envelope Spectrum — Before vs After", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Impact Repetition Frequency (Hz)")
    axes[1].set_ylabel("Envelope Magnitude")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    vel_str = f"{comp.velocity_change_pct:+.1f}%" if comp.velocity_change_pct is not None else "N/A"
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>vibcheck Comparison — {html.escape(comp.verdict)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f8fafc; color: #0f172a; margin: 0; padding: 28px; }}
  .wrap {{ max-width: 940px; margin: 0 auto; background: #fff; border-radius: 12px;
           box-shadow: 0 2px 14px rgba(15,23,42,0.08); padding: 28px 32px; }}
  .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 18px 0; }}
  .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px; }}
  .lbl {{ font-size: 12px; color: #64748b; text-transform: uppercase; }}
  .val {{ font-size: 18px; font-weight: 700; margin-top: 4px; }}
  img {{ width: 100%; border: 1px solid #e2e8f0; border-radius: 8px; margin-top: 12px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1 style="margin:0;font-size:22px;">Before vs After Comparison — {html.escape(comp.verdict)}</h1>
  <p>{html.escape(comp.summary)}</p>
  <div class="cards">
    <div class="card"><div class="lbl">RMS Change</div><div class="val">{comp.rms_change_pct:+.1f}%</div></div>
    <div class="card"><div class="lbl">ISO Velocity Change</div><div class="val">{vel_str}</div></div>
    <div class="card"><div class="lbl">Kurtosis Change</div><div class="val">{comp.kurtosis_delta:+.2f}</div></div>
  </div>
  <img src="data:image/png;base64,{png_b64}" alt="Before vs After comparison">
</div>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
