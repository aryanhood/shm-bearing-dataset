"""Closed-loop hysteresis supervisory controller tests."""

from __future__ import annotations

from orderguard.encoder import SensorAudit
from orderguard.supervisor import SupervisoryController

GOOD_AUDIT = SensorAudit(
    ok=True,
    clipped_ratio=0.0,
    dropout_ratio=0.0,
    dc_offset=0.0,
    rms=0.05,
    encoder_pulses=200_000,
    rpm_min=800.0,
    rpm_max=1600.0,
    issues=(),
)

BAD_AUDIT = SensorAudit(
    ok=False,
    clipped_ratio=0.0,
    dropout_ratio=1.0,
    dc_offset=0.0,
    rms=0.0,
    encoder_pulses=0,
    rpm_min=0.0,
    rpm_max=0.0,
    issues=("encoder_signal_lost",),
)


def _probs(healthy: float = 0.95, inner: float = 0.05, combined: float = 0.0) -> list[float]:
    rem = max(0.0, 1.0 - healthy - inner - combined)
    return [healthy, inner, rem * 0.5, rem * 0.5, combined]


def test_flicker_rejection_and_hysteresis_transitions() -> None:
    ctrl = SupervisoryController(
        window_k=3, window_n=4, recovery_clean_steps=3, max_scale_slew=0.25
    )

    # Single noisy fault spike must NOT trigger DERATE
    c1 = ctrl.update(_probs(healthy=0.1, inner=0.9), GOOD_AUDIT, time_s=0.5, rpm=1200.0)
    assert c1.mode == "NOMINAL"
    assert c1.velocity_scale == 1.0

    # Followed by clean windows -> stays NOMINAL
    c2 = ctrl.update(_probs(healthy=0.95, inner=0.05), GOOD_AUDIT, time_s=1.0, rpm=1250.0)
    c3 = ctrl.update(_probs(healthy=0.95, inner=0.05), GOOD_AUDIT, time_s=1.5, rpm=1300.0)
    assert c2.mode == "NOMINAL"
    assert c3.mode == "NOMINAL"

    # 3-of-4 inner race hits -> DERATE with slew-limited scale reduction
    for step_idx in range(3):
        cmd = ctrl.update(
            _probs(healthy=0.1, inner=0.9), GOOD_AUDIT, time_s=2.0 + step_idx, rpm=1350.0
        )
    assert cmd.mode == "DERATE"
    assert not cmd.brake_engage
    assert cmd.velocity_scale < 1.0

    # High-confidence single inner fault stays DERATE (does not false E-stop)
    cmd_inner = ctrl.update(_probs(healthy=0.02, inner=0.98), GOOD_AUDIT, time_s=5.0, rpm=1380.0)
    assert cmd_inner.mode == "DERATE"
    assert not cmd_inner.brake_engage

    # Sustained severe combined fault -> SAFE_STOP
    for step_idx in range(3):
        cmd = ctrl.update(
            _probs(healthy=0.02, combined=0.98), GOOD_AUDIT, time_s=6.0 + step_idx, rpm=1400.0
        )
    assert cmd.mode == "SAFE_STOP"
    assert cmd.brake_engage


def test_recovery_and_sensor_failsafe() -> None:
    ctrl = SupervisoryController(window_k=2, window_n=3, recovery_clean_steps=2, max_scale_slew=0.5)
    ctrl.update(_probs(healthy=0.2, inner=0.8), GOOD_AUDIT, time_s=1.0, rpm=1100.0)
    cmd = ctrl.update(_probs(healthy=0.2, inner=0.8), GOOD_AUDIT, time_s=2.0, rpm=1100.0)
    assert cmd.mode == "DERATE"

    # Clean recovery dwell restores NOMINAL
    ctrl.update(_probs(healthy=0.96, inner=0.04), GOOD_AUDIT, time_s=3.0, rpm=1100.0)
    cmd = ctrl.update(_probs(healthy=0.96, inner=0.04), GOOD_AUDIT, time_s=4.0, rpm=1100.0)
    assert cmd.mode == "NOMINAL"

    # Bad sensor audit immediately derates, and 2nd consecutive bad frame triggers SAFE_STOP
    c_bad1 = ctrl.update(_probs(healthy=0.99, inner=0.01), BAD_AUDIT, time_s=5.0, rpm=0.0)
    assert c_bad1.mode == "DERATE"
    c_bad2 = ctrl.update(_probs(healthy=0.99, inner=0.01), BAD_AUDIT, time_s=6.0, rpm=0.0)
    assert c_bad2.mode == "SAFE_STOP"
    assert c_bad2.brake_engage
