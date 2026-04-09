"""
Decision agent for SHM maintenance actions.

The agent is deliberately simple and auditable: it consumes the canonical
model-layer output and converts it into a decision-layer output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..contracts import AgentOutput, DecisionStatus, DecisionUrgency, ModelOutput
from ..utils.config import CFG

STATUS_SAFE = DecisionStatus.SAFE.value
STATUS_WARNING = DecisionStatus.WARNING.value
STATUS_CRITICAL = DecisionStatus.CRITICAL.value

_ACTIONS: Mapping[DecisionStatus, str] = {
    DecisionStatus.SAFE: "Continue scheduled monitoring.",
    DecisionStatus.WARNING: "Increase monitoring frequency and schedule inspection within 48 hours.",
    DecisionStatus.CRITICAL: "Halt operation and initiate immediate inspection and repair.",
}


@dataclass(slots=True)
class DecisionThresholds:
    safe_threshold: float
    warning_threshold: float
    anomaly_threshold: float


class DecisionAgent:
    """Rule-based SHM decision module."""

    def __init__(
        self,
        safe_threshold: float = 0.80,
        warning_threshold: float = 0.55,
        anomaly_threshold: float = -0.25,
    ) -> None:
        self.thresholds = DecisionThresholds(
            safe_threshold=safe_threshold,
            warning_threshold=warning_threshold,
            anomaly_threshold=anomaly_threshold,
        )

    @classmethod
    def from_config(cls, cfg: Mapping | None = None) -> "DecisionAgent":
        config = cfg or CFG
        return cls(
            safe_threshold=float(config["agent"]["safe_threshold"]),
            warning_threshold=float(config["agent"]["warning_threshold"]),
            anomaly_threshold=float(config["agent"]["anomaly_threshold"]),
        )

    def decide(self, model_output: ModelOutput) -> AgentOutput:
        """Convert a model-layer result into an operational decision."""
        anomaly_score = model_output.anomaly_score
        status = self._assign_status(
            predicted_class=model_output.predicted_class,
            confidence=model_output.confidence,
            anomaly_score=anomaly_score,
        )
        return AgentOutput(
            status=status,
            action=_ACTIONS[status],
            rationale=self._build_rationale(model_output, status),
            urgency=self._to_urgency(status),
        )

    def _assign_status(
        self,
        *,
        predicted_class: int,
        confidence: float,
        anomaly_score: float | None,
    ) -> DecisionStatus:
        anomaly_flag = (
            anomaly_score is not None
            and anomaly_score < self.thresholds.anomaly_threshold
        )

        if predicted_class == 0 and confidence >= self.thresholds.safe_threshold and not anomaly_flag:
            return DecisionStatus.SAFE

        if predicted_class > 0 and confidence >= self.thresholds.safe_threshold:
            return DecisionStatus.CRITICAL

        if predicted_class > 0 and confidence >= self.thresholds.warning_threshold:
            return DecisionStatus.WARNING

        if anomaly_flag:
            return DecisionStatus.WARNING

        return DecisionStatus.WARNING

    def _build_rationale(
        self,
        model_output: ModelOutput,
        status: DecisionStatus,
    ) -> str:
        anomaly_score = (
            f"{model_output.anomaly_score:.3f}"
            if model_output.anomaly_score is not None
            else "unavailable"
        )
        lines = [
            f"Classifier predicts '{model_output.predicted_label}' with {model_output.confidence * 100:.1f}% confidence.",
            f"Anomaly score: {anomaly_score}.",
            f"Health index: {model_output.health_index:.3f}.",
        ]
        if status is DecisionStatus.SAFE:
            lines.append("No actionable fault indicators were detected for this window.")
        elif status is DecisionStatus.WARNING:
            lines.append("Early degradation indicators are present; inspection is recommended before continued operation.")
        else:
            lines.append("A high-confidence fault condition was detected and requires immediate intervention.")
        return " ".join(lines)

    @staticmethod
    def _to_urgency(status: DecisionStatus) -> DecisionUrgency:
        if status is DecisionStatus.SAFE:
            return DecisionUrgency.LOW
        if status is DecisionStatus.WARNING:
            return DecisionUrgency.MEDIUM
        return DecisionUrgency.HIGH
