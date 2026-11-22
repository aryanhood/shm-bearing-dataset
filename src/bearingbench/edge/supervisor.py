"""Closed-loop hysteresis supervisory controller with conformal ambiguity guard."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

from ..data.encoder import SensorAudit
from ..data.uottawa import UOTTAWA_CLASSES

Mode = Literal["NOMINAL", "DERATE", "SAFE_STOP"]


@dataclass(frozen=True)
class ActuatorCommand:
    """Rate-limited supervisory command emitted for each analysis window."""

    step: int
    time_s: float
    rpm: float
    mode: Mode
    predicted_class: str
    fault_prob: float
    conformal_set_size: int
    velocity_scale: float
    torque_limit: float
    brake_engage: bool
    reason: str


def _slew_toward(current: float, target: float, max_step: float) -> float:
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + (max_step if delta > 0.0 else -max_step)


class SupervisoryController:
    """Three-state hysteresis FSM with K-of-N confirmation, conformal guard & slew limits."""

    def __init__(
        self,
        *,
        window_k: int = 3,
        window_n: int = 4,
        derate_prob: float = 0.55,
        stop_prob: float = 0.85,
        recovery_clean_steps: int = 5,
        max_scale_slew: float = 0.20,
        severe_classes: tuple[str, ...] = ("combined",),
        classes: tuple[str, ...] = UOTTAWA_CLASSES,
    ) -> None:
        if window_k <= 0 or window_n < window_k:
            raise ValueError("require 0 < window_k <= window_n")
        self.window_k = window_k
        self.window_n = window_n
        self.derate_prob = derate_prob
        self.stop_prob = stop_prob
        self.recovery_clean_steps = recovery_clean_steps
        self.max_scale_slew = max_scale_slew
        self.severe_classes = set(severe_classes)
        self.classes = classes

        self.mode: Mode = "NOMINAL"
        self.velocity_scale: float = 1.0
        self.torque_limit: float = 1.0
        self._step: int = 0
        self._derate_hist: deque[bool] = deque(maxlen=window_n)
        self._stop_hist: deque[bool] = deque(maxlen=window_n)
        self._clean_streak: int = 0
        self._bad_sensor_streak: int = 0

    def reset(self) -> None:
        self.mode = "NOMINAL"
        self.velocity_scale = 1.0
        self.torque_limit = 1.0
        self._step = 0
        self._derate_hist.clear()
        self._stop_hist.clear()
        self._clean_streak = 0
        self._bad_sensor_streak = 0

    def update(
        self,
        probs: tuple[float, ...] | list[float],
        audit: SensorAudit,
        *,
        time_s: float,
        rpm: float,
        conformal_set: tuple[int, ...] | None = None,
    ) -> ActuatorCommand:
        if len(probs) != len(self.classes):
            raise ValueError(f"expected {len(self.classes)} probabilities, got {len(probs)}")

        healthy_idx = self.classes.index("healthy")
        fault_prob = float(max(0.0, min(1.0, 1.0 - float(probs[healthy_idx]))))
        pred_idx = max(range(len(probs)), key=lambda i: float(probs[i]))
        pred_class = self.classes[pred_idx]
        c_size = len(conformal_set) if conformal_set is not None else 1
        conformal_fault_included = conformal_set is not None and any(
            idx != healthy_idx for idx in conformal_set
        )

        if not audit.ok:
            self._bad_sensor_streak += 1
            self._clean_streak = 0
            if self._bad_sensor_streak >= 2:
                self.mode = "SAFE_STOP"
                reason = f"sensor_fault_persist: {','.join(audit.issues)}"
            else:
                if self.mode == "NOMINAL":
                    self.mode = "DERATE"
                reason = f"sensor_fault: {','.join(audit.issues)}"
        else:
            self._bad_sensor_streak = 0
            is_derate_hit = (fault_prob >= self.derate_prob) or (
                c_size > 1 and conformal_fault_included
            )
            is_stop_hit = fault_prob >= self.stop_prob and pred_class in self.severe_classes
            self._derate_hist.append(is_derate_hit)
            self._stop_hist.append(is_stop_hit)

            if not is_derate_hit:
                self._clean_streak += 1
            else:
                self._clean_streak = 0

            stop_confirmed = sum(self._stop_hist) >= self.window_k
            derate_confirmed = sum(self._derate_hist) >= self.window_k

            if self.mode == "SAFE_STOP":
                reason = "latched_safe_stop"
            elif stop_confirmed:
                self.mode = "SAFE_STOP"
                reason = f"confirmed_severe_fault ({pred_class}, p={fault_prob:.2f})"
            elif self.mode == "NOMINAL":
                if derate_confirmed:
                    self.mode = "DERATE"
                    reason = (
                        f"conformal_ambiguity_derate (|C|={c_size}, p={fault_prob:.2f})"
                        if fault_prob < self.derate_prob
                        else f"confirmed_fault_derate ({pred_class}, p={fault_prob:.2f})"
                    )
                else:
                    reason = "nominal"
            else:
                if self._clean_streak >= self.recovery_clean_steps:
                    self.mode = "NOMINAL"
                    self._derate_hist.clear()
                    self._stop_hist.clear()
                    reason = "recovered_after_clean_dwell"
                else:
                    reason = f"derated_monitoring ({pred_class}, p={fault_prob:.2f})"

        if self.mode == "SAFE_STOP":
            target_vel, target_trq, brake = 0.0, 0.0, True
        elif self.mode == "DERATE":
            target_vel, target_trq, brake = 0.60, 0.70, False
        else:
            target_vel, target_trq, brake = 1.0, 1.0, False

        self.velocity_scale = _slew_toward(self.velocity_scale, target_vel, self.max_scale_slew)
        self.torque_limit = _slew_toward(self.torque_limit, target_trq, self.max_scale_slew)

        cmd = ActuatorCommand(
            step=self._step,
            time_s=round(float(time_s), 4),
            rpm=round(float(rpm), 2),
            mode=self.mode,
            predicted_class=pred_class,
            fault_prob=round(fault_prob, 4),
            conformal_set_size=c_size,
            velocity_scale=round(self.velocity_scale, 4),
            torque_limit=round(self.torque_limit, 4),
            brake_engage=brake,
            reason=reason,
        )
        self._step += 1
        return cmd
