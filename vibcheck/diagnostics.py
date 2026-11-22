"""Deterministic signal audit, ISO 20816-3 severity, envelope analysis & plain-English findings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.fft import next_fast_len
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from .baseline import BaselineDrift, BaselineProfile, score_against_baseline
from .bearings import BearingSpec, get_bearing
from .io import SignalData

G_TO_MS2 = 9.80665
EPS = 1e-12

Status = Literal["HEALTHY", "WATCH", "INSPECT SOON", "CRITICAL"]
IsoClass = Literal["group2_rigid", "group2_flexible", "group1_rigid", "group1_flexible"]

# ISO 20816-3 broadband RMS velocity boundaries (mm/s): (A/B, B/C, C/D)
ISO_20816_3_LIMITS: dict[IsoClass, tuple[float, float, float]] = {
    "group2_rigid": (1.4, 2.8, 4.5),
    "group2_flexible": (2.3, 4.5, 7.1),
    "group1_rigid": (2.3, 4.5, 7.1),
    "group1_flexible": (3.5, 7.1, 11.0),
}

ISO_ZONE_LABELS: dict[str, str] = {
    "A": "Good (newly commissioned / smooth operation)",
    "B": "Acceptable (unrestricted long-term operation)",
    "C": "Unsatisfactory (schedule maintenance soon)",
    "D": "Unacceptable (risk of machine damage)",
}


@dataclass(frozen=True)
class SignalQuality:
    ok: bool
    clipped_ratio: float
    dropout_ratio: float
    dc_offset: float
    issues: tuple[str, ...]


@dataclass(frozen=True)
class IsoSeverity:
    velocity_rms_mm_s: float
    zone: Literal["A", "B", "C", "D"]
    zone_description: str
    machine_class: IsoClass
    limits_mm_s: tuple[float, float, float]


@dataclass(frozen=True)
class FaultMatch:
    name: str
    expected_hz: float
    snr_ratio: float
    detected: bool
    detail: str


@dataclass(frozen=True)
class DiagnosisResult:
    source_name: str
    channel: str
    fs_hz: float
    duration_s: float
    units: str
    rpm: float | None
    bearing: BearingSpec | None
    status: Status
    summary: str
    rms: float
    peak: float
    crest_factor: float
    kurtosis: float
    quality: SignalQuality
    iso: IsoSeverity | None
    baseline_drift: BaselineDrift | None
    fault_matches: tuple[FaultMatch, ...]
    findings: tuple[str, ...]
    next_steps: tuple[str, ...]
    envelope_band_hz: tuple[float, float]
    # Arrays kept for clean plotting / HTML report rendering
    waveform: NDArray[np.float64]
    fft_freqs: NDArray[np.float64]
    fft_mag: NDArray[np.float64]
    env_freqs: NDArray[np.float64]
    env_mag: NDArray[np.float64]


def audit_signal(sig: SignalData) -> SignalQuality:
    """Check raw signal for dead channels, NaNs, clipping, or insufficient length."""
    x = sig.values
    issues: list[str] = []
    if len(x) == 0 or not np.all(np.isfinite(x)):
        return SignalQuality(False, 1.0, 1.0, 0.0, ("Signal contains NaN or Inf values.",))

    if sig.duration_s < 0.25:
        issues.append(f"Recording is very short ({sig.duration_s:.2f} s); aim for at least 2-5 s.")

    dc = float(np.mean(x))
    centered = x - dc
    rms = float(np.sqrt(np.mean(centered**2)))
    dropout = float(np.mean(np.abs(np.diff(x)) < 1e-9)) if len(x) > 1 else 1.0
    if rms < 1e-7 or dropout > 0.30:
        issues.append("Signal looks flat or disconnected (very low energy or repeated samples).")

    peak = float(np.max(np.abs(x)))
    clipped = float(np.mean(np.abs(x) >= 0.998 * peak)) if peak > 0.0 else 0.0
    if clipped > 0.01:
        issues.append(
            f"Sensor clipping detected ({clipped:.1%} of samples hit the peak rail); lower gain."
        )

    return SignalQuality(
        ok=len(issues) == 0,
        clipped_ratio=round(clipped, 5),
        dropout_ratio=round(dropout, 5),
        dc_offset=round(dc, 6),
        issues=tuple(issues),
    )


def compute_iso_velocity(
    sig: SignalData,
    *,
    machine_class: IsoClass = "group2_rigid",
) -> IsoSeverity | None:
    """Integrate acceleration (g or m/s^2) to ISO 20816-3 velocity RMS (10-1000 Hz)."""
    if sig.units not in {"g", "m/s2"}:
        return None
    if sig.fs_hz < 40.0:
        return None

    x = sig.values - float(np.mean(sig.values))
    a_ms2 = x * G_TO_MS2 if sig.units == "g" else x
    n = len(a_ms2)
    spec = np.fft.rfft(a_ms2)
    freqs = np.fft.rfftfreq(n, d=1.0 / sig.fs_hz)

    f_hi = min(1000.0, 0.45 * sig.fs_hz)
    mask = (freqs >= 10.0) & (freqs <= f_hi)
    vel_spec = np.zeros_like(spec)
    vel_spec[mask] = spec[mask] / (1j * 2.0 * np.pi * freqs[mask])
    vel_ms = np.fft.irfft(vel_spec, n=n)
    v_rms_mm_s = 1000.0 * float(np.sqrt(np.mean(vel_ms**2)))

    lim_ab, lim_bc, lim_cd = ISO_20816_3_LIMITS[machine_class]
    zone: Literal["A", "B", "C", "D"]
    if v_rms_mm_s < lim_ab:
        zone = "A"
    elif v_rms_mm_s < lim_bc:
        zone = "B"
    elif v_rms_mm_s < lim_cd:
        zone = "C"
    else:
        zone = "D"

    return IsoSeverity(
        velocity_rms_mm_s=round(v_rms_mm_s, 3),
        zone=zone,
        zone_description=ISO_ZONE_LABELS[zone],
        machine_class=machine_class,
        limits_mm_s=(lim_ab, lim_bc, lim_cd),
    )


def _select_envelope_band(
    x: NDArray[np.float64], fs_hz: float
) -> tuple[tuple[float, float], NDArray[np.float64]]:
    nyq = 0.5 * fs_hz
    candidates: list[tuple[float, float]] = []
    if fs_hz >= 8000.0:
        candidates.append((500.0, min(2500.0, 0.45 * fs_hz)))
        candidates.append((1500.0, min(5500.0, 0.45 * fs_hz)))
        if fs_hz >= 24_000.0:
            candidates.append((4000.0, min(12_000.0, 0.45 * fs_hz)))
    elif fs_hz >= 400.0:
        candidates.append((0.15 * fs_hz, 0.45 * fs_hz))
    else:
        candidates.append((max(5.0, 0.10 * fs_hz), 0.45 * fs_hz))

    best_band = candidates[0]
    best_sig = x
    best_kurt = -1.0
    for lo, hi in candidates:
        if lo >= hi or hi >= nyq:
            continue
        sos = butter(4, [lo, hi], btype="bandpass", fs=fs_hz, output="sos")
        filt = np.asarray(sosfiltfilt(sos, x), dtype=np.float64)
        rms = float(np.sqrt(np.mean(filt**2))) + EPS
        kurt = float(np.mean(filt**4)) / (rms**4)
        if kurt > best_kurt:
            best_kurt = kurt
            best_band = (lo, hi)
            best_sig = filt
    return best_band, best_sig


def _peak_snr(
    freqs: NDArray[np.float64],
    mag: NDArray[np.float64],
    target_hz: float,
    *,
    tol_pct: float = 0.025,
    min_tol_hz: float = 1.5,
) -> float:
    if target_hz <= 0.0 or len(freqs) == 0 or target_hz > float(freqs[-1]) * 0.96:
        return 0.0
    tol = max(min_tol_hz, target_hz * tol_pct)
    ring = max(6.0, tol * 4.0)
    core_m = (freqs >= target_hz - tol) & (freqs <= target_hz + tol)
    ring_m = ((freqs >= max(2.0, target_hz - tol - ring)) & (freqs < target_hz - tol)) | (
        (freqs > target_hz + tol) & (freqs <= target_hz + tol + ring)
    )
    if not np.any(core_m) or not np.any(ring_m):
        return 0.0
    core = float(np.max(mag[core_m]))
    bg = float(np.median(mag[ring_m])) + EPS
    return core / bg


def analyze(
    sig: SignalData,
    *,
    rpm: float | None = None,
    bearing: str | BearingSpec | None = None,
    iso_class: IsoClass = "group2_rigid",
    baseline: BaselineProfile | None = None,
) -> DiagnosisResult:
    """Run full signal audit, ISO severity, envelope diagnostics, and plain-English advice."""
    quality = audit_signal(sig)
    x = sig.values - float(np.mean(sig.values))
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x))) if len(x) > 0 else 0.0
    crest = peak / (rms + EPS)
    kurtosis = float(np.mean(x**4)) / ((rms + EPS) ** 4) if len(x) > 0 else 0.0

    iso = compute_iso_velocity(sig, machine_class=iso_class)
    drift = score_against_baseline(sig, baseline) if baseline is not None else None

    # Welch PSD / amplitude spectrum for low-frequency shaft harmonics
    nperseg = min(len(x), max(256, min(4096, int(sig.fs_hz))))
    f_psd, pxx = welch(x, fs=sig.fs_hz, nperseg=nperseg)
    fft_max_hz = min(1500.0, 0.45 * sig.fs_hz)
    f_mask = f_psd <= fft_max_hz
    fft_freqs = np.asarray(f_psd[f_mask], dtype=np.float64)
    fft_mag = np.asarray(np.sqrt(np.maximum(pxx[f_mask], 0.0)), dtype=np.float64)

    # Adaptive Hilbert squared-envelope spectrum for repeating bearing impacts
    env_band, x_bp = _select_envelope_band(x, sig.fs_hz)
    nfft = next_fast_len(len(x_bp))
    env = np.abs(hilbert(x_bp, N=nfft)[: len(x_bp)]) ** 2
    env = env - float(np.mean(env))
    hann = np.hanning(len(env))
    e_spec = np.abs(np.fft.rfft(env * hann)) / (float(np.sum(hann)) + EPS)
    e_freqs = np.fft.rfftfreq(len(env), d=1.0 / sig.fs_hz)
    env_max_hz = min(600.0, 0.45 * sig.fs_hz)
    e_mask = e_freqs <= env_max_hz
    env_freqs = np.asarray(e_freqs[e_mask], dtype=np.float64)
    env_mag = np.asarray(e_spec[e_mask], dtype=np.float64)

    bearing_spec: BearingSpec | None = get_bearing(bearing) if isinstance(bearing, str) else bearing

    matches: list[FaultMatch] = []
    findings: list[str] = []
    next_steps: list[str] = []

    if not quality.ok:
        findings.extend(quality.issues)
        next_steps.append("Check sensor mounting, cable connection, and ADC range, then re-record.")

    if iso is not None:
        findings.append(
            f"ISO 20816-3 vibration velocity is {iso.velocity_rms_mm_s:.2f} mm/s RMS "
            f"(Zone {iso.zone} — {iso.zone_description})."
        )
    elif sig.units == "audio":
        findings.append(
            "Uncalibrated audio input: skipping ISO mm/s severity and using spectral/shock checks."
        )

    if kurtosis >= 8.0:
        findings.append(
            f"Strong sharp impacts detected (kurtosis {kurtosis:.1f}; smooth machines sit ~3.0)."
        )
    elif kurtosis >= 5.0:
        findings.append(
            f"Mild impulsive spikes detected (kurtosis {kurtosis:.1f} vs ~3.0 normal baseline)."
        )

    if drift is not None:
        findings.append(drift.plain_reason)

    # Shaft-speed and bearing kinematic checks when RPM is known
    detected_bearing_faults: list[FaultMatch] = []
    if rpm is not None and rpm > 0.0:
        shaft_hz = rpm / 60.0
        snr_1x = _peak_snr(fft_freqs, fft_mag, shaft_hz)
        snr_2x = _peak_snr(fft_freqs, fft_mag, 2.0 * shaft_hz)
        if snr_1x >= 6.0:
            m = FaultMatch(
                "Shaft unbalance (1x RPM)",
                shaft_hz,
                round(snr_1x, 2),
                True,
                f"Strong 1x shaft peak at {shaft_hz:.1f} Hz ({snr_1x:.1f}x local floor).",
            )
            matches.append(m)
            findings.append(m.detail)
            next_steps.append("Check rotor/fan balance, bent shaft, or buildup on blades.")
        if snr_2x >= 6.0:
            m = FaultMatch(
                "Misalignment (2x RPM)",
                2.0 * shaft_hz,
                round(snr_2x, 2),
                True,
                f"Strong 2x shaft peak at {2.0 * shaft_hz:.1f} Hz ({snr_2x:.1f}x local floor).",
            )
            matches.append(m)
            findings.append(m.detail)
            next_steps.append("Check shaft coupling alignment and mounting bolts.")

        if bearing_spec is not None:
            freqs_map = bearing_spec.frequencies_hz(rpm)
            if freqs_map["bpfi"] > 0.45 * sig.fs_hz:
                findings.append(
                    f"Sample rate ({sig.fs_hz:.0f} Hz) is too low for {bearing_spec.code} "
                    f"inner-race impacts ({freqs_map['bpfi']:.1f} Hz); record at >= 2 kHz."
                )
            else:
                checks = [
                    ("Inner race wear (BPFI)", freqs_map["bpfi"]),
                    ("Outer race wear (BPFO)", freqs_map["bpfo"]),
                    ("Rolling element / ball wear (2xBSF)", freqs_map["bsf_2x"]),
                ]
                for label, f_hz in checks:
                    s1 = _peak_snr(env_freqs, env_mag, f_hz)
                    s2 = _peak_snr(env_freqs, env_mag, 2.0 * f_hz)
                    score = max(s1, 0.85 * s2)
                    hit = score >= 4.5
                    detail = (
                        f"{label} matches a repeating knock at {f_hz:.1f} Hz "
                        f"({score:.1f}x above local noise floor)."
                        if hit
                        else f"No clear {label.lower()} peak at {f_hz:.1f} Hz."
                    )
                    fm = FaultMatch(label, round(f_hz, 2), round(score, 2), hit, detail)
                    matches.append(fm)
                    if hit:
                        detected_bearing_faults.append(fm)
                        findings.append(detail)

                if detected_bearing_faults:
                    next_steps.append(
                        f"Inspect {bearing_spec.code} bearing lubrication, check for roughness by "
                        "hand-spinning, and plan a replacement."
                    )

    # Combine evidence into one plain-English overall status
    status: Status = "HEALTHY"
    if not quality.ok:
        status = "WATCH"
    if iso is not None and iso.zone == "D":
        status = "CRITICAL"
    elif iso is not None and iso.zone == "C":
        status = "INSPECT SOON"
    elif iso is not None and iso.zone == "B" and status == "HEALTHY":
        status = "HEALTHY"

    if drift is not None:
        if drift.drift_score >= 75.0:
            status = "CRITICAL" if status == "CRITICAL" else "INSPECT SOON"
        elif drift.drift_score >= 40.0 and status == "HEALTHY":
            status = "WATCH"

    if detected_bearing_faults:
        strongest = max(m.snr_ratio for m in detected_bearing_faults)
        if strongest >= 8.0 or kurtosis >= 7.0 or (iso is not None and iso.zone in {"C", "D"}):
            status = "CRITICAL" if (iso is not None and iso.zone == "D") else "INSPECT SOON"
        elif status == "HEALTHY":
            status = "WATCH"
    elif kurtosis >= 8.0 and status == "HEALTHY":
        status = "INSPECT SOON"
    elif kurtosis >= 5.0 and status == "HEALTHY":
        status = "WATCH"

    if status == "HEALTHY":
        summary = (
            "Machine looks healthy — vibration energy and impulse levels are within normal bounds."
        )
        if not next_steps:
            next_steps.append("No action needed. Save a baseline or re-check periodically.")
    elif status == "WATCH":
        summary = (
            "Early change noticed — monitor more frequently and compare against a healthy baseline."
        )
        if not next_steps:
            next_steps.append(
                "Re-test under the same load/RPM in a few days to see if vibration grows."
            )
    elif status == "INSPECT SOON":
        summary = (
            "Fault signature or elevated vibration detected — schedule a physical inspection soon."
        )
        if not next_steps:
            next_steps.append("Check bearing lubrication, mounting tightness, and shaft alignment.")
    else:
        summary = (
            "Severe vibration detected — stop or reduce load and inspect before damage worsens."
        )
        if not next_steps:
            next_steps.append(
                "Inspect bearings, coupling, and mounts immediately before continued operation."
            )

    # Keep a short preview slice of the waveform (up to 4000 points) for clean plotting
    preview_n = min(len(x), 4000)
    return DiagnosisResult(
        source_name=sig.source_name,
        channel=sig.channel,
        fs_hz=sig.fs_hz,
        duration_s=sig.duration_s,
        units=sig.units,
        rpm=rpm,
        bearing=bearing_spec,
        status=status,
        summary=summary,
        rms=round(rms, 6),
        peak=round(peak, 6),
        crest_factor=round(crest, 3),
        kurtosis=round(kurtosis, 3),
        quality=quality,
        iso=iso,
        baseline_drift=drift,
        fault_matches=tuple(matches),
        findings=tuple(findings),
        next_steps=tuple(dict.fromkeys(next_steps)),
        envelope_band_hz=env_band,
        waveform=x[:preview_n].copy(),
        fft_freqs=fft_freqs,
        fft_mag=fft_mag,
        env_freqs=env_freqs,
        env_mag=env_mag,
    )
