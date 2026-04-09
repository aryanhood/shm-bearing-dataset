"""Tests for the rule-based decision agent."""
from __future__ import annotations

import pytest

from src.agent.decision_agent import DecisionAgent, STATUS_CRITICAL, STATUS_SAFE, STATUS_WARNING
from src.contracts import DecisionUrgency, ModelOutput


@pytest.fixture
def agent() -> DecisionAgent:
    return DecisionAgent(
        safe_threshold=0.80,
        warning_threshold=0.55,
        anomaly_threshold=-0.25,
    )


def make_model_output(
    *,
    predicted_class: int,
    confidence: float,
    anomaly_score: float | None,
    health_index: float,
    predicted_label: str = "Normal",
) -> ModelOutput:
    probabilities = {
        "Normal": 1.0 - confidence if predicted_class != 0 else confidence,
        "Inner Race Fault": confidence if predicted_class == 1 else 0.0,
        "Outer Race Fault": confidence if predicted_class == 2 else 0.0,
        "Ball Fault": confidence if predicted_class == 3 else 0.0,
    }
    if predicted_class != 0:
        probabilities["Normal"] = max(0.0, 1.0 - confidence)
    else:
        residual = max(0.0, 1.0 - confidence)
        probabilities["Inner Race Fault"] = residual / 3.0
        probabilities["Outer Race Fault"] = residual / 3.0
        probabilities["Ball Fault"] = residual / 3.0

    labels = {
        0: "Normal",
        1: "Inner Race Fault",
        2: "Outer Race Fault",
        3: "Ball Fault",
    }
    return ModelOutput(
        predicted_class=predicted_class,
        predicted_label=predicted_label if predicted_label != "Normal" else labels[predicted_class],
        confidence=confidence,
        class_probs=probabilities,
        anomaly_score=anomaly_score,
        health_index=health_index,
    )


def test_safe_when_normal_high_confidence(agent: DecisionAgent) -> None:
    output = make_model_output(predicted_class=0, confidence=0.92, anomaly_score=-0.10, health_index=0.87)
    decision = agent.decide(output)
    assert decision.status.value == STATUS_SAFE
    assert decision.urgency is DecisionUrgency.LOW


def test_critical_when_fault_high_confidence(agent: DecisionAgent) -> None:
    output = make_model_output(
        predicted_class=1,
        predicted_label="Inner Race Fault",
        confidence=0.88,
        anomaly_score=-0.10,
        health_index=0.24,
    )
    decision = agent.decide(output)
    assert decision.status.value == STATUS_CRITICAL
    assert decision.urgency is DecisionUrgency.HIGH


def test_warning_when_anomalous(agent: DecisionAgent) -> None:
    output = make_model_output(predicted_class=0, confidence=0.81, anomaly_score=-0.40, health_index=0.41)
    decision = agent.decide(output)
    assert decision.status.value == STATUS_WARNING
    assert decision.urgency is DecisionUrgency.MEDIUM


def test_warning_when_fault_confidence_is_moderate(agent: DecisionAgent) -> None:
    output = make_model_output(
        predicted_class=2,
        predicted_label="Outer Race Fault",
        confidence=0.61,
        anomaly_score=-0.12,
        health_index=0.46,
    )
    decision = agent.decide(output)
    assert decision.status.value == STATUS_WARNING


def test_agent_output_fields(agent: DecisionAgent) -> None:
    output = make_model_output(predicted_class=0, confidence=0.85, anomaly_score=None, health_index=0.82)
    decision = agent.decide(output)
    assert decision.action
    assert decision.rationale
    assert decision.status.value in {STATUS_SAFE, STATUS_WARNING, STATUS_CRITICAL}
