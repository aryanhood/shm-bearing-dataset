"""Centralized inference pipeline for SHM predictions."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis as kurt_fn, skew as skew_fn

from ..agent.decision_agent import DecisionAgent
from ..contracts import (
    ArtifactStatus,
    CacheStats,
    ExplainRequest,
    ExplainResponse,
    FeatureImportanceItem,
    LatencyStats,
    ModelChoice,
    ModelOutput,
    PipelineMetricsResponse,
    PredictRequest,
    PredictResponse,
    configured_class_names,
    configured_window_size,
)
from ..data.preprocessor import Preprocessor
from ..models.anomaly import AnomalyDetector
from ..models.cnn1d import CNN1DModel
from ..models.random_forest import RandomForestModel
from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.paths import ArtifactPaths, get_artifact_paths
from .cache import TTLCache

log = get_logger("inference.pipeline")


class InferencePipelineError(RuntimeError):
    """Base error for predictable inference failures."""

    def __init__(self, detail: str, *, error: str = "inference_error", status_code: int = 500) -> None:
        super().__init__(detail)
        self.detail = detail
        self.error = error
        self.status_code = status_code


class ArtifactLoadError(InferencePipelineError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail, error="artifact_load_error", status_code=503)


class ArtifactCompatibilityError(InferencePipelineError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail, error="artifact_compatibility_error", status_code=503)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_artifact_manifest(
    cfg: Mapping[str, Any],
    *,
    available_models: Mapping[str, bool],
    artifact_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a compatibility manifest for saved artifacts."""
    feature_cfg = cfg["features"]
    freq_cfg = feature_cfg["freq_domain"]
    return {
        "project_version": cfg.get("project", {}).get("version", "unknown"),
        "window_size": int(cfg["data"]["window_size"]),
        "sampling_rate": int(cfg["data"]["sampling_rate"]),
        "classes": {str(k): str(v) for k, v in cfg["data"]["classes"].items()},
        "feature_flags": {
            "include_time": bool(feature_cfg.get("time_domain", [])),
            "include_freq": bool(freq_cfg.get("enabled", True)),
            "n_fft_bins": int(freq_cfg.get("n_fft_bins", 64)),
        },
        "available_models": dict(available_models),
        "artifact_hashes": dict(artifact_hashes or {}),
    }


@dataclass
class LoadedArtifacts:
    preprocessor: Preprocessor | None = None
    rf: RandomForestModel | None = None
    cnn: CNN1DModel | None = None
    anomaly: AnomalyDetector | None = None
    manifest: dict[str, Any] | None = None


class InferencePipeline:
    """Single source of truth for deployed inference."""

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.cfg = cfg or get_config()
        self.paths: ArtifactPaths = get_artifact_paths(self.cfg).ensure()
        self.agent = DecisionAgent.from_config(self.cfg)
        self.artifacts = LoadedArtifacts()
        self.compatibility_issues: list[str] = []

        serving_cfg = self.cfg.get("serving", {})
        self.max_batch_size = int(serving_cfg.get("max_batch_size", 256))
        if self.max_batch_size <= 0:
            raise ValueError("serving.max_batch_size must be > 0")

        cache_cfg = serving_cfg.get("prediction_cache", {})
        cache_enabled = bool(cache_cfg.get("enabled", False))
        if cache_enabled:
            self._cache: TTLCache[str, PredictResponse] | None = TTLCache(
                max_entries=int(cache_cfg.get("max_entries", 2048)),
                ttl_seconds=float(cache_cfg.get("ttl_seconds", 300)),
            )
        else:
            self._cache = None

        self._metrics_lock = threading.Lock()
        self._requests_total = 0
        self._failures_total = 0
        self._latencies_ms: deque[float] = deque(maxlen=1024)

    def load_artifacts(self) -> "InferencePipeline":
        """Load artifacts from disk and refresh compatibility state."""
        self.artifacts = LoadedArtifacts()
        self.compatibility_issues = []

        if self.paths.manifest_path.exists():
            self.artifacts.manifest = json.loads(self.paths.manifest_path.read_text(encoding="utf-8"))
            self.compatibility_issues = self._check_compatibility(self.artifacts.manifest)
        else:
            self.compatibility_issues.append(
                f"Missing artifact manifest: {self.paths.manifest_path}"
            )

        self.artifacts.preprocessor = self._load_preprocessor()
        self.artifacts.rf = self._load_rf()
        self.artifacts.cnn = self._load_cnn()
        self.artifacts.anomaly = self._load_anomaly()
        return self

    def status(self) -> ArtifactStatus:
        return ArtifactStatus(
            rf=self.artifacts.rf is not None,
            cnn=self.artifacts.cnn is not None,
            anomaly=self.artifacts.anomaly is not None,
            preprocessor=self.artifacts.preprocessor is not None,
            manifest=self.artifacts.manifest is not None,
            compatible=len(self.compatibility_issues) == 0,
            issues=list(self.compatibility_issues),
        )

    def metrics(self) -> PipelineMetricsResponse:
        with self._metrics_lock:
            requests_total = self._requests_total
            failures_total = self._failures_total
            latencies = np.array(self._latencies_ms, dtype=np.float64)

        if len(latencies) == 0:
            latency_stats = LatencyStats(p50=0.0, p95=0.0, max=0.0)
        else:
            latency_stats = LatencyStats(
                p50=float(np.percentile(latencies, 50)),
                p95=float(np.percentile(latencies, 95)),
                max=float(np.max(latencies)),
            )

        cache_snapshot = self._cache.snapshot() if self._cache is not None else None
        if cache_snapshot is None:
            cache_stats = CacheStats(
                enabled=False,
                entries=0,
                max_entries=0,
                ttl_seconds=0.0,
                hits=0,
                misses=0,
                evictions=0,
                hit_rate=0.0,
            )
        else:
            lookups = cache_snapshot.hits + cache_snapshot.misses
            hit_rate = float(cache_snapshot.hits / lookups) if lookups > 0 else 0.0
            cache_stats = CacheStats(
                enabled=True,
                entries=cache_snapshot.entries,
                max_entries=cache_snapshot.max_entries,
                ttl_seconds=cache_snapshot.ttl_seconds,
                hits=cache_snapshot.hits,
                misses=cache_snapshot.misses,
                evictions=cache_snapshot.evictions,
                hit_rate=hit_rate,
            )

        failure_rate = float(failures_total / requests_total) if requests_total > 0 else 0.0
        return PipelineMetricsResponse(
            requests_total=requests_total,
            failures_total=failures_total,
            failure_rate=failure_rate,
            max_batch_size=self.max_batch_size,
            latency_ms=latency_stats,
            cache=cache_stats,
        )

    def predict(self, request: PredictRequest) -> PredictResponse:
        start = time.perf_counter()
        try:
            self._ensure_compatible()
            model_choice = request.model
            model = self._select_model(model_choice)
            preprocessor = self._require_preprocessor()

            signal = np.asarray(request.signal, dtype=np.float32).reshape(1, -1)
            cache_key = self._cache_key(signal[0], model_choice)
            cached = self._cache.get(cache_key) if self._cache is not None else None
            if cached is not None:
                latency_ms = (time.perf_counter() - start) * 1000.0
                self._record_success(latency_ms)
                return cached.model_copy(deep=True)

            scaled = preprocessor.transform(signal)
            probabilities = model.predict_proba(scaled)
            anomaly_scores = self.artifacts.anomaly.score(scaled) if self.artifacts.anomaly else None
            responses = self._assemble_predictions(
                probabilities=probabilities,
                anomaly_scores=anomaly_scores,
                model_choice=model_choice,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
            response = responses[0]
            if self._cache is not None:
                self._cache.put(cache_key, response.model_copy(deep=True))

            self._record_success(response.latency_ms)
            return response
        except Exception:
            self._record_failure((time.perf_counter() - start) * 1000.0)
            raise

    def predict_batch(
        self,
        signals: np.ndarray,
        *,
        model: ModelChoice,
    ) -> list[PredictResponse]:
        start = time.perf_counter()
        try:
            self._ensure_compatible()
            if signals.ndim != 2:
                raise InferencePipelineError("signals must have shape (n_samples, window_size)", status_code=400)
            if signals.shape[0] > self.max_batch_size:
                raise InferencePipelineError(
                    f"Batch size {signals.shape[0]} exceeds max_batch_size={self.max_batch_size}",
                    error="batch_size_exceeded",
                    status_code=413,
                )

            expected = configured_window_size()
            if signals.shape[1] != expected:
                raise InferencePipelineError(
                    f"Expected window_size={expected}, received {signals.shape[1]}",
                    status_code=400,
                )

            selected_model = self._select_model(model)
            preprocessor = self._require_preprocessor()
            scaled = preprocessor.transform(signals.astype(np.float32))
            probabilities = selected_model.predict_proba(scaled)
            anomaly_scores = self.artifacts.anomaly.score(scaled) if self.artifacts.anomaly else None
            latency_ms = (time.perf_counter() - start) * 1000.0
            per_item_latency_ms = float(latency_ms / max(1, len(signals)))
            responses = self._assemble_predictions(
                probabilities=probabilities,
                anomaly_scores=anomaly_scores,
                model_choice=model,
                latency_ms=per_item_latency_ms,
            )
            self._record_success(latency_ms)
            return responses
        except Exception:
            self._record_failure((time.perf_counter() - start) * 1000.0)
            raise

    def explain(self, request: ExplainRequest) -> ExplainResponse:
        self._ensure_compatible()
        rf = self.artifacts.rf
        if rf is None:
            raise ArtifactLoadError("Random Forest artifact is not available for /explain.")

        signal = np.asarray(request.signal, dtype=np.float32)
        importances = rf.feature_importances()
        feature_names = rf.extractor.feature_names
        top_indices = np.argsort(importances)[::-1][:10]
        top_features = [
            FeatureImportanceItem(
                feature=feature_names[index],
                importance=float(importances[index]),
            )
            for index in top_indices
        ]

        rms = float(np.sqrt(np.mean(signal ** 2)))
        stats = {
            "rms": rms,
            "kurtosis": float(kurt_fn(signal, fisher=True)),
            "skewness": float(skew_fn(signal)),
            "peak_to_peak": float(np.ptp(signal)),
            "crest_factor": float(np.max(np.abs(signal)) / (rms + 1e-12)),
        }

        sampling_rate = int(self.cfg["data"]["sampling_rate"])
        magnitudes = np.abs(rfft(signal))
        frequencies = rfftfreq(len(signal), d=1.0 / sampling_rate)
        fft_peak = float(frequencies[np.argmax(magnitudes)])

        return ExplainResponse(
            top_features=top_features,
            signal_stats=stats,
            fft_peak_hz=fft_peak,
            note="Global Random Forest feature importances ranked by mean decrease impurity.",
        )

    def _assemble_predictions(
        self,
        *,
        probabilities: np.ndarray,
        anomaly_scores: np.ndarray | None,
        model_choice: ModelChoice,
        latency_ms: float,
    ) -> list[PredictResponse]:
        class_names = self._resolve_class_names(probabilities.shape[1])
        results: list[PredictResponse] = []

        for index, proba in enumerate(probabilities):
            anomaly_score = float(anomaly_scores[index]) if anomaly_scores is not None else None
            model_output = self._build_model_output(proba, anomaly_score, class_names)
            agent_output = self.agent.decide(model_output)
            results.append(
                PredictResponse(
                    model_used=model_choice,
                    window_size=configured_window_size(),
                    latency_ms=latency_ms,
                    model_output=model_output,
                    agent_output=agent_output,
                )
            )
        return results

    def _resolve_class_names(self, n_classes: int) -> dict[int, str]:
        class_names = configured_class_names()
        resolved = {index: class_names.get(index, f"class_{index}") for index in range(n_classes)}
        return resolved

    def _build_model_output(
        self,
        probabilities: np.ndarray,
        anomaly_score: float | None,
        class_names: Mapping[int, str],
    ) -> ModelOutput:
        predicted_class = int(np.argmax(probabilities))
        predicted_label = class_names[predicted_class]
        class_probs = {
            class_names[index]: float(value)
            for index, value in enumerate(probabilities)
        }
        return ModelOutput(
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            confidence=float(probabilities[predicted_class]),
            class_probs=class_probs,
            anomaly_score=anomaly_score,
            health_index=self._compute_health_index(probabilities, anomaly_score),
        )

    @staticmethod
    def _compute_health_index(probabilities: np.ndarray, anomaly_score: float | None) -> float:
        normal_probability = float(probabilities[0])
        if anomaly_score is None:
            anomaly_component = normal_probability
        else:
            anomaly_component = float(np.clip((anomaly_score + 0.5) / 0.5, 0.0, 1.0))
        return float(np.clip(0.6 * normal_probability + 0.4 * anomaly_component, 0.0, 1.0))

    def _cache_key(self, signal: np.ndarray, model_choice: ModelChoice) -> str:
        digest = hashlib.sha256(signal.astype(np.float32, copy=False).tobytes()).hexdigest()
        manifest_fingerprint = self._manifest_fingerprint()
        return f"{model_choice.value}:{manifest_fingerprint}:{digest}"

    def _manifest_fingerprint(self) -> str:
        if not self.artifacts.manifest:
            return "missing-manifest"
        encoded = json.dumps(self.artifacts.manifest, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _select_model(self, choice: ModelChoice) -> RandomForestModel | CNN1DModel:
        if choice is ModelChoice.RF:
            if self.artifacts.rf is None:
                raise ArtifactLoadError("Random Forest artifact not loaded. Run training first.")
            return self.artifacts.rf
        if self.artifacts.cnn is None:
            raise ArtifactLoadError("CNN artifact not loaded. Run training first.")
        return self.artifacts.cnn

    def _require_preprocessor(self) -> Preprocessor:
        if self.artifacts.preprocessor is None:
            raise ArtifactLoadError("Preprocessor artifact not loaded. Run training first.")
        return self.artifacts.preprocessor

    def _ensure_compatible(self) -> None:
        if self.compatibility_issues:
            joined = " ".join(self.compatibility_issues)
            raise ArtifactCompatibilityError(joined)

    def _check_compatibility(self, manifest: Mapping[str, Any]) -> list[str]:
        cfg = self.cfg
        expected = build_artifact_manifest(
            cfg,
            available_models=manifest.get("available_models", {}),
            artifact_hashes=manifest.get("artifact_hashes", {}),
        )
        issues: list[str] = []

        keys_to_compare = ["project_version", "window_size", "sampling_rate", "classes", "feature_flags"]
        for key in keys_to_compare:
            if manifest.get(key) != expected.get(key):
                issues.append(f"Artifact manifest mismatch for '{key}'.")

        self._append_model_availability_issues(manifest, issues)
        self._append_hash_issues(manifest, issues)
        return issues

    def _append_model_availability_issues(self, manifest: Mapping[str, Any], issues: list[str]) -> None:
        declared = manifest.get("available_models", {})
        presence_checks = {
            "rf": self.paths.rf_model_path.exists(),
            "cnn": self.paths.cnn_model_path.exists(),
            "anomaly": self.paths.anomaly_model_path.exists(),
            "preprocessor": self.paths.preprocessor_path.exists(),
        }
        for model_name, declared_available in declared.items():
            actual_available = presence_checks.get(model_name)
            if actual_available is None:
                continue
            if bool(declared_available) != bool(actual_available):
                issues.append(
                    f"Artifact availability mismatch for '{model_name}' "
                    f"(manifest={declared_available}, disk={actual_available})."
                )

    def _append_hash_issues(self, manifest: Mapping[str, Any], issues: list[str]) -> None:
        hashes = manifest.get("artifact_hashes", {})
        if not isinstance(hashes, Mapping):
            issues.append("Manifest field 'artifact_hashes' is malformed.")
            return

        artifacts = self._manifest_artifacts()
        for artifact_name, artifact_path in artifacts.items():
            if not artifact_path.exists():
                continue
            expected_hash = hashes.get(artifact_name)
            if expected_hash is None:
                issues.append(f"Missing checksum in manifest for artifact '{artifact_name}'.")
                continue
            actual_hash = _sha256_file(artifact_path)
            if actual_hash != expected_hash:
                issues.append(f"Artifact checksum mismatch for '{artifact_name}'.")

    def _manifest_artifacts(self) -> dict[str, Path]:
        return {
            "preprocessor": self.paths.preprocessor_path,
            "rf_model": self.paths.rf_model_path,
            "cnn_model": self.paths.cnn_model_path,
            "anomaly_model": self.paths.anomaly_model_path,
            "config_snapshot": self.paths.config_snapshot_path,
        }

    def _record_success(self, latency_ms: float) -> None:
        with self._metrics_lock:
            self._requests_total += 1
            self._latencies_ms.append(float(latency_ms))

    def _record_failure(self, latency_ms: float) -> None:
        with self._metrics_lock:
            self._requests_total += 1
            self._failures_total += 1
            self._latencies_ms.append(float(latency_ms))

    def _load_preprocessor(self) -> Preprocessor | None:
        path = self.paths.preprocessor_path
        if not path.exists():
            log.warning("Preprocessor artifact missing at %s", path)
            return None
        try:
            return Preprocessor().load(path)
        except Exception as exc:
            self.compatibility_issues.append(f"Failed to load preprocessor: {exc}")
            return None

    def _load_rf(self) -> RandomForestModel | None:
        path = self.paths.rf_model_path
        if not path.exists():
            return None
        try:
            return RandomForestModel(
                cfg=self.cfg,
                sampling_rate=int(self.cfg["data"]["sampling_rate"]),
                seed=int(self.cfg.get("project.seed") or 42),
            ).load(path)
        except Exception as exc:
            self.compatibility_issues.append(f"Failed to load Random Forest artifact: {exc}")
            return None

    def _load_cnn(self) -> CNN1DModel | None:
        path = self.paths.cnn_model_path
        if not path.exists():
            return None
        try:
            merged_cfg = {**self.cfg["models"], **self.cfg["training"]}
            return CNN1DModel(
                cfg=merged_cfg,
                n_classes=len(self.cfg["data"]["classes"]),
                seed=int(self.cfg.get("project.seed") or 42),
            ).load(path)
        except Exception as exc:
            self.compatibility_issues.append(f"Failed to load CNN artifact: {exc}")
            return None

    def _load_anomaly(self) -> AnomalyDetector | None:
        path = self.paths.anomaly_model_path
        if not path.exists():
            return None
        try:
            return AnomalyDetector(
                cfg=self.cfg,
                sampling_rate=int(self.cfg["data"]["sampling_rate"]),
            ).load(path)
        except Exception as exc:
            self.compatibility_issues.append(f"Failed to load anomaly artifact: {exc}")
            return None
