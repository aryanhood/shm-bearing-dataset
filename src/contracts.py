"""
Canonical system contracts.

These schemas define the shared boundary between the UI/API, inference
pipeline, and decision agent.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .utils.config import get_config


def configured_window_size() -> int:
    return int(get_config()["data"]["window_size"])


def configured_class_names() -> Dict[int, str]:
    classes = get_config()["data"]["classes"]
    return {int(k): str(v) for k, v in classes.items()}


class ModelChoice(str, Enum):
    RF = "rf"
    CNN = "cnn"


class DecisionStatus(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class DecisionUrgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SignalRequest(BaseModel):
    """Canonical input contract for one vibration window."""

    model_config = ConfigDict(extra="forbid")

    signal: List[float] = Field(
        ...,
        description="Raw vibration window samples before preprocessing.",
    )

    @field_validator("signal")
    @classmethod
    def validate_signal(cls, values: List[float]) -> List[float]:
        expected = configured_window_size()
        if len(values) != expected:
            raise ValueError(f"signal must contain exactly {expected} samples")
        if any(not math.isfinite(float(x)) for x in values):
            raise ValueError("signal must contain only finite numeric values")
        return [float(x) for x in values]


class PredictRequest(SignalRequest):
    model: ModelChoice = Field(default=ModelChoice.CNN)


class ExplainRequest(SignalRequest):
    pass


class ModelOutput(BaseModel):
    """Canonical model-layer output."""

    model_config = ConfigDict(extra="forbid")

    predicted_class: int = Field(..., ge=0)
    predicted_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_probs: Dict[str, float]
    anomaly_score: float | None = None
    health_index: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_probabilities(self) -> "ModelOutput":
        total = sum(float(value) for value in self.class_probs.values())
        if self.class_probs and not math.isclose(total, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError("class_probs must sum to approximately 1.0")
        return self


class AgentOutput(BaseModel):
    """Canonical decision-layer output."""

    model_config = ConfigDict(extra="forbid")

    status: DecisionStatus
    action: str = Field(..., min_length=3)
    rationale: str = Field(..., min_length=3)
    urgency: DecisionUrgency


class PredictResponse(BaseModel):
    """Structured end-to-end inference response."""

    model_config = ConfigDict(extra="forbid")

    model_used: ModelChoice
    window_size: int = Field(..., ge=1)
    latency_ms: float = Field(..., ge=0.0)
    model_output: ModelOutput
    agent_output: AgentOutput


class FeatureImportanceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    importance: float


class ExplainResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_features: List[FeatureImportanceItem]
    signal_stats: Dict[str, float]
    fft_peak_hz: float
    note: str


class ArtifactStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rf: bool
    cnn: bool
    anomaly: bool
    preprocessor: bool
    manifest: bool
    compatible: bool
    issues: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_status: str = "ok"
    artifacts: ArtifactStatus
    config_version: str


class LatencyStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    p50: float = Field(..., ge=0.0)
    p95: float = Field(..., ge=0.0)
    max: float = Field(..., ge=0.0)


class CacheStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    entries: int = Field(..., ge=0)
    max_entries: int = Field(..., ge=0)
    ttl_seconds: float = Field(..., ge=0.0)
    hits: int = Field(..., ge=0)
    misses: int = Field(..., ge=0)
    evictions: int = Field(..., ge=0)
    hit_rate: float = Field(..., ge=0.0, le=1.0)


class PipelineMetricsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests_total: int = Field(..., ge=0)
    failures_total: int = Field(..., ge=0)
    failure_rate: float = Field(..., ge=0.0, le=1.0)
    max_batch_size: int = Field(..., ge=1)
    latency_ms: LatencyStats
    cache: CacheStats


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str
    status_code: int
