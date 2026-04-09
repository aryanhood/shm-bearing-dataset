"""Tests for the centralized inference pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from src.contracts import ModelChoice, PredictRequest
from src.inference.pipeline import ArtifactCompatibilityError, InferencePipeline, InferencePipelineError
from src.utils.config import CFG


class FakePreprocessor:
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32)


class FakeModel:
    def __init__(self, probabilities: np.ndarray) -> None:
        self._probabilities = probabilities.astype(np.float32)
        self.calls = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.calls += 1
        if len(X) == len(self._probabilities):
            return self._probabilities
        return np.repeat(self._probabilities[:1], len(X), axis=0)


class FakeAnomaly:
    def __init__(self, scores: np.ndarray) -> None:
        self._scores = scores.astype(np.float32)

    def score(self, X: np.ndarray) -> np.ndarray:
        if len(X) == len(self._scores):
            return self._scores
        return np.repeat(self._scores[:1], len(X), axis=0)


def make_pipeline(probabilities: np.ndarray, scores: np.ndarray) -> InferencePipeline:
    pipeline = InferencePipeline(CFG)
    pipeline.artifacts.preprocessor = FakePreprocessor()
    pipeline.artifacts.rf = FakeModel(probabilities)
    pipeline.artifacts.cnn = FakeModel(probabilities)
    pipeline.artifacts.anomaly = FakeAnomaly(scores)
    pipeline.artifacts.manifest = {}
    pipeline.compatibility_issues = []
    return pipeline


def test_predict_returns_structured_contract() -> None:
    window = int(CFG["data"]["window_size"])
    pipeline = make_pipeline(
        probabilities=np.array([[0.91, 0.04, 0.03, 0.02]], dtype=np.float32),
        scores=np.array([-0.08], dtype=np.float32),
    )
    request = PredictRequest(signal=np.zeros(window, dtype=np.float32).tolist(), model=ModelChoice.RF)
    result = pipeline.predict(request)

    assert result.model_used is ModelChoice.RF
    assert result.window_size == window
    assert 0.0 <= result.model_output.confidence <= 1.0
    assert 0.0 <= result.model_output.health_index <= 1.0
    assert result.agent_output.status.value == "SAFE"


def test_predict_batch_uses_shared_pipeline_logic() -> None:
    window = int(CFG["data"]["window_size"])
    probabilities = np.array(
        [
            [0.87, 0.05, 0.04, 0.04],
            [0.12, 0.70, 0.10, 0.08],
        ],
        dtype=np.float32,
    )
    scores = np.array([-0.10, -0.05], dtype=np.float32)
    pipeline = make_pipeline(probabilities=probabilities, scores=scores)
    signals = np.zeros((2, window), dtype=np.float32)
    results = pipeline.predict_batch(signals, model=ModelChoice.CNN)

    assert len(results) == 2
    assert results[0].agent_output.status.value == "SAFE"
    assert results[1].model_output.predicted_label == "Inner Race Fault"


def test_pipeline_rejects_incompatible_artifacts() -> None:
    window = int(CFG["data"]["window_size"])
    pipeline = make_pipeline(
        probabilities=np.array([[0.91, 0.04, 0.03, 0.02]], dtype=np.float32),
        scores=np.array([-0.08], dtype=np.float32),
    )
    pipeline.compatibility_issues = ["manifest mismatch"]

    with pytest.raises(ArtifactCompatibilityError):
        pipeline.predict(
            PredictRequest(signal=np.zeros(window, dtype=np.float32).tolist(), model=ModelChoice.RF)
        )


def test_predict_uses_cache_for_repeated_requests() -> None:
    window = int(CFG["data"]["window_size"])
    signal = np.zeros(window, dtype=np.float32).tolist()
    pipeline = make_pipeline(
        probabilities=np.array([[0.91, 0.04, 0.03, 0.02]], dtype=np.float32),
        scores=np.array([-0.08], dtype=np.float32),
    )

    first = pipeline.predict(PredictRequest(signal=signal, model=ModelChoice.RF))
    second = pipeline.predict(PredictRequest(signal=signal, model=ModelChoice.RF))

    assert first.model_output.predicted_class == second.model_output.predicted_class
    assert pipeline.artifacts.rf.calls == 1
    assert pipeline.metrics().cache.hit_rate >= 0.5


def test_predict_batch_enforces_max_batch_size() -> None:
    window = int(CFG["data"]["window_size"])
    pipeline = make_pipeline(
        probabilities=np.array([[0.91, 0.04, 0.03, 0.02]], dtype=np.float32),
        scores=np.array([-0.08], dtype=np.float32),
    )
    pipeline.max_batch_size = 1

    with pytest.raises(InferencePipelineError) as exc:
        pipeline.predict_batch(np.zeros((2, window), dtype=np.float32), model=ModelChoice.RF)

    assert exc.value.error == "batch_size_exceeded"
    assert exc.value.status_code == 413
