"""API contract and integration tests."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.contracts import (
    AgentOutput,
    ArtifactStatus,
    DecisionStatus,
    DecisionUrgency,
    ExplainResponse,
    FeatureImportanceItem,
    LatencyStats,
    ModelChoice,
    ModelOutput,
    PipelineMetricsResponse,
    PredictResponse,
    CacheStats,
)
from src.inference.pipeline import ArtifactLoadError
from src.utils.config import CFG


def make_signal(n: int | None = None) -> list[float]:
    length = n or int(CFG["data"]["window_size"])
    rng = np.random.default_rng(1)
    return rng.standard_normal(length).astype(float).tolist()


class FakePipeline:
    def status(self) -> ArtifactStatus:
        return ArtifactStatus(
            rf=True,
            cnn=True,
            anomaly=True,
            preprocessor=True,
            manifest=True,
            compatible=True,
            issues=[],
        )

    def predict(self, _) -> PredictResponse:
        return PredictResponse(
            model_used=ModelChoice.RF,
            window_size=int(CFG["data"]["window_size"]),
            latency_ms=3.2,
            model_output=ModelOutput(
                predicted_class=0,
                predicted_label="Normal",
                confidence=0.93,
                class_probs={
                    "Normal": 0.93,
                    "Inner Race Fault": 0.03,
                    "Outer Race Fault": 0.02,
                    "Ball Fault": 0.02,
                },
                anomaly_score=-0.08,
                health_index=0.88,
            ),
            agent_output=AgentOutput(
                status=DecisionStatus.SAFE,
                action="Continue scheduled monitoring.",
                rationale="No actionable fault indicators were detected.",
                urgency=DecisionUrgency.LOW,
            ),
        )

    def metrics(self) -> PipelineMetricsResponse:
        return PipelineMetricsResponse(
            requests_total=10,
            failures_total=1,
            failure_rate=0.1,
            max_batch_size=256,
            latency_ms=LatencyStats(p50=3.0, p95=6.0, max=9.0),
            cache=CacheStats(
                enabled=True,
                entries=4,
                max_entries=2048,
                ttl_seconds=300.0,
                hits=6,
                misses=4,
                evictions=0,
                hit_rate=0.6,
            ),
        )

    def explain(self, _) -> ExplainResponse:
        return ExplainResponse(
            top_features=[FeatureImportanceItem(feature="rms", importance=0.42)],
            signal_stats={"rms": 1.0, "kurtosis": 0.1},
            fft_peak_hz=120.0,
            note="test",
        )


class BrokenPipeline(FakePipeline):
    def predict(self, _):
        raise ArtifactLoadError("Preprocessor artifact not loaded. Run training first.")


@pytest.fixture
def client(monkeypatch) -> TestClient:
    fake = FakePipeline()
    monkeypatch.setattr("src.api.main.get_pipeline", lambda: fake)
    monkeypatch.setattr("src.api.main.load_pipeline", lambda: fake)
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        yield test_client


def test_health_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["api_status"] == "ok"
    assert payload["artifacts"]["rf"] is True
    assert payload["artifacts"]["compatible"] is True


def test_predict_returns_nested_contract(client: TestClient) -> None:
    response = client.post("/predict", json={"signal": make_signal(), "model": "rf"})
    assert response.status_code == 200
    assert "x-request-id" in response.headers
    assert "x-process-time-ms" in response.headers
    payload = response.json()
    assert payload["model_used"] == "rf"
    assert payload["agent_output"]["status"] == "SAFE"
    assert payload["model_output"]["predicted_label"] == "Normal"


def test_predict_invalid_model_returns_validation_error(client: TestClient) -> None:
    response = client.post("/predict", json={"signal": make_signal(), "model": "gpt"})
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "validation_error"


def test_predict_invalid_window_size_returns_validation_error(client: TestClient) -> None:
    response = client.post("/predict", json={"signal": make_signal(8), "model": "rf"})
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "validation_error"
    assert "exactly" in payload["detail"]


def test_explain_returns_structured_payload(client: TestClient) -> None:
    response = client.post("/explain", json={"signal": make_signal()})
    assert response.status_code == 200
    payload = response.json()
    assert payload["top_features"][0]["feature"] == "rms"
    assert payload["fft_peak_hz"] == 120.0


def test_metrics_returns_pipeline_telemetry(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["requests_total"] == 10
    assert payload["cache"]["enabled"] is True
    assert payload["latency_ms"]["p95"] == 6.0


def test_predict_surfaces_structured_pipeline_errors(monkeypatch) -> None:
    broken = BrokenPipeline()
    monkeypatch.setattr("src.api.main.get_pipeline", lambda: broken)
    monkeypatch.setattr("src.api.main.load_pipeline", lambda: broken)
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.post("/predict", json={"signal": make_signal(), "model": "rf"})

    assert response.status_code == 503
    payload = response.json()
    assert payload["error"] == "artifact_load_error"
