"""Incremental encoder decoding and sensor health auditing.

Decodes 1024-PPR TTL pulse trains using a Schmitt trigger with sub-sample linear
edge interpolation, producing cumulative shaft revolutions theta(t) and
instantaneous shaft speed RPM(t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import medfilt

from .data import ENCODER_PPR


@dataclass(frozen=True)
class EncoderTrack:
    """Decoded angular trajectory from an incremental rotary encoder."""

    edge_times_s: NDArray[np.float64]
    edge_revs: NDArray[np.float64]
    rpm_times_s: NDArray[np.float64]
    rpm_values: NDArray[np.float64]
    ppr: int
    v_low: float
    v_high: float

    @property
    def rpm_min(self) -> float:
        return float(np.min(self.rpm_values)) if len(self.rpm_values) > 0 else 0.0

    @property
    def rpm_max(self) -> float:
        return float(np.max(self.rpm_values)) if len(self.rpm_values) > 0 else 0.0

    @property
    def rpm_mean(self) -> float:
        return float(np.mean(self.rpm_values)) if len(self.rpm_values) > 0 else 0.0

    def rpm_at(self, times_s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate smoothed instantaneous RPM at arbitrary timestamps."""
        if len(self.rpm_times_s) == 0:
            return np.zeros_like(times_s, dtype=np.float64)
        return np.asarray(
            np.interp(times_s, self.rpm_times_s, self.rpm_values),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class SensorAudit:
    """Pre-inference instrumentation sanity check across vibration & encoder."""

    ok: bool
    clipped_ratio: float
    dropout_ratio: float
    dc_offset: float
    rms: float
    encoder_pulses: int
    rpm_min: float
    rpm_max: float
    issues: tuple[str, ...]


def decode_encoder(
    signal: NDArray[np.float64],
    fs_hz: float,
    *,
    ppr: int = ENCODER_PPR,
) -> EncoderTrack:
    """Decode TTL rising edges into shaft revolutions and smoothed RPM.

    Uses hysteresis thresholds at 35% and 65% of the [5th, 95th] percentile span
    and linearly interpolates the exact midpoint crossing to remove sample-clock
    quantization jitter.
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    if len(sig) < 4 or not np.all(np.isfinite(sig)):
        empty = np.array([], dtype=np.float64)
        return EncoderTrack(empty, empty, empty, empty, ppr=ppr, v_low=0.0, v_high=0.0)

    p5, p95 = np.percentile(sig, [5.0, 95.0])
    span = float(p95 - p5)
    if span < 0.5:
        empty = np.array([], dtype=np.float64)
        return EncoderTrack(empty, empty, empty, empty, ppr=ppr, v_low=float(p5), v_high=float(p95))

    v_low = float(p5) + 0.35 * span
    v_high = float(p5) + 0.65 * span
    v_mid = float(p5) + 0.50 * span

    state = np.zeros(len(sig), dtype=np.int8)
    state[sig >= v_high] = 1
    state[sig <= v_low] = -1
    nz = np.flatnonzero(state)
    if len(nz) < 2:
        empty = np.array([], dtype=np.float64)
        return EncoderTrack(empty, empty, empty, empty, ppr=ppr, v_low=v_low, v_high=v_high)

    # Propagate the last decisive state forward through transition samples.
    idx_map = np.searchsorted(nz, np.arange(len(sig)), side="right") - 1
    idx_map = np.maximum(idx_map, 0)
    schmitt = state[nz[idx_map]]
    rise_idx = np.flatnonzero((schmitt[:-1] < 0) & (schmitt[1:] > 0))
    if len(rise_idx) < 4:
        empty = np.array([], dtype=np.float64)
        return EncoderTrack(empty, empty, empty, empty, ppr=ppr, v_low=v_low, v_high=v_high)

    # Step back at most a few samples if the crossing of v_mid occurred just before rise_idx.
    cross_idx = rise_idx.copy()
    for _ in range(4):
        need_back = (cross_idx > 0) & (sig[cross_idx] > v_mid)
        if not np.any(need_back):
            break
        cross_idx[need_back] -= 1

    v0 = sig[cross_idx]
    v1 = sig[cross_idx + 1]
    denom = np.where(np.abs(v1 - v0) > 1e-9, v1 - v0, 1.0)
    frac = np.clip((v_mid - v0) / denom, 0.0, 1.0)
    edge_times_s = (cross_idx.astype(np.float64) + frac) / fs_hz
    edge_revs = np.arange(len(edge_times_s), dtype=np.float64) / float(ppr)

    # Estimate instantaneous RPM over multi-pulse strides to suppress single-tooth jitter.
    stride = min(128, max(1, len(edge_times_s) // 8))
    dt = edge_times_s[stride:] - edge_times_s[:-stride]
    drev = float(stride) / float(ppr)
    raw_rpm = 60.0 * drev / np.maximum(dt, 1e-9)
    rpm_times_s = 0.5 * (edge_times_s[stride:] + edge_times_s[:-stride])

    if len(raw_rpm) >= 7:
        k_med = min(15, len(raw_rpm) if len(raw_rpm) % 2 == 1 else len(raw_rpm) - 1)
        rpm_values = np.asarray(medfilt(raw_rpm, kernel_size=max(3, k_med)), dtype=np.float64)
    else:
        rpm_values = raw_rpm

    return EncoderTrack(
        edge_times_s=edge_times_s,
        edge_revs=edge_revs,
        rpm_times_s=rpm_times_s,
        rpm_values=rpm_values,
        ppr=ppr,
        v_low=v_low,
        v_high=v_high,
    )


def audit_sensors(
    vibration: NDArray[np.float64],
    track: EncoderTrack,
    *,
    rail_limit: float | None = None,
    min_rpm: float = 60.0,
    max_rpm: float = 10_000.0,
) -> SensorAudit:
    """Validate accelerometer and encoder integrity before running diagnostics."""
    vib = np.asarray(vibration, dtype=np.float64).ravel()
    issues: list[str] = []

    if len(vib) == 0 or not np.all(np.isfinite(vib)):
        return SensorAudit(
            ok=False,
            clipped_ratio=1.0,
            dropout_ratio=1.0,
            dc_offset=0.0,
            rms=0.0,
            encoder_pulses=len(track.edge_times_s),
            rpm_min=0.0,
            rpm_max=0.0,
            issues=("non_finite_vibration",),
        )

    dc_offset = float(vib.mean())
    centered = vib - dc_offset
    rms = float(np.sqrt(np.mean(centered**2)))

    # Detect sustained flatline dropouts via short sample-to-sample differences.
    diffs = np.abs(np.diff(vib))
    dropout_ratio = float(np.mean(diffs < 1e-8))
    if rms < 1e-7 or dropout_ratio > 0.25:
        issues.append("vibration_dropout_or_dead_channel")

    if rail_limit is not None and rail_limit > 0.0:
        clipped_ratio = float(np.mean(np.abs(vib) >= 0.995 * rail_limit))
    else:
        peak = float(np.max(np.abs(vib)))
        clipped_ratio = float(np.mean(np.abs(vib) >= 0.999 * peak)) if peak > 0.0 else 0.0
    if clipped_ratio > 0.01:
        issues.append("vibration_rail_clipping")

    n_pulses = len(track.edge_times_s)
    if n_pulses < 16 or len(track.rpm_values) == 0:
        issues.append("encoder_signal_lost")
    else:
        if track.rpm_min < min_rpm or track.rpm_max > max_rpm:
            issues.append("encoder_rpm_out_of_range")

    return SensorAudit(
        ok=len(issues) == 0,
        clipped_ratio=round(clipped_ratio, 6),
        dropout_ratio=round(dropout_ratio, 6),
        dc_offset=round(dc_offset, 6),
        rms=round(rms, 6),
        encoder_pulses=n_pulses,
        rpm_min=round(track.rpm_min, 2),
        rpm_max=round(track.rpm_max, 2),
        issues=tuple(issues),
    )
