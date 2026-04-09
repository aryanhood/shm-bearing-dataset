"""API-facing singleton wrapper around the centralized inference pipeline."""
from __future__ import annotations

from typing import Optional

from ..inference.pipeline import InferencePipeline

_PIPELINE: Optional[InferencePipeline] = None


def load_pipeline() -> InferencePipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = InferencePipeline()
    return _PIPELINE.load_artifacts()


def get_pipeline() -> InferencePipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = InferencePipeline().load_artifacts()
    return _PIPELINE


def pipeline_status() -> dict:
    return get_pipeline().status().model_dump()


def get_preprocessor():
    return get_pipeline().artifacts.preprocessor


def get_rf():
    return get_pipeline().artifacts.rf


def get_cnn():
    return get_pipeline().artifacts.cnn


def get_anomaly():
    return get_pipeline().artifacts.anomaly
