"""FastAPI application for the SHM decision-support system."""
from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..contracts import (
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PipelineMetricsResponse,
    PredictRequest,
    PredictResponse,
)
from ..inference.pipeline import InferencePipelineError
from ..utils.config import get_config
from ..utils.logger import get_logger, setup_root_logger
from .model_store import get_pipeline, load_pipeline


def create_app() -> FastAPI:
    cfg = get_config()
    setup_root_logger(
        level=cfg.get("project.log_level") or "INFO",
        log_file=cfg.get("artifacts.log_file"),
    )
    log = get_logger("api.main")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        log.info("Loading inference artifacts...")
        load_pipeline()
        log.info("API ready.")
        yield
        log.info("API shutting down.")

    app = FastAPI(
        title="SHM Decision-Support API",
        description="Bearing fault detection and rule-based maintenance decisions",
        version=cfg.get("project.version") or "1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg["api"].get("cors_origins", ["*"]),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        request_id = uuid4().hex[:12]
        started = perf_counter()
        response = await call_next(request)
        latency_ms = (perf_counter() - started) * 1000.0
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
        log.info(
            "request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        return response

    @app.exception_handler(InferencePipelineError)
    async def handle_pipeline_error(_: Request, exc: InferencePipelineError) -> JSONResponse:
        payload = ErrorResponse(
            error=exc.error,
            detail=exc.detail,
            status_code=exc.status_code,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        payload = ErrorResponse(
            error="validation_error",
            detail=str(exc),
            status_code=422,
        )
        return JSONResponse(status_code=422, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled API error", exc_info=exc)
        payload = ErrorResponse(
            error="internal_server_error",
            detail="Unexpected internal error during request processing.",
            status_code=500,
        )
        return JSONResponse(status_code=500, content=payload.model_dump())

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health() -> HealthResponse:
        pipeline = get_pipeline()
        return HealthResponse(
            api_status="ok",
            artifacts=pipeline.status(),
            config_version=cfg.get("project.version") or "1.0.0",
        )

    @app.get("/metrics", response_model=PipelineMetricsResponse, tags=["System"])
    def metrics() -> PipelineMetricsResponse:
        pipeline = get_pipeline()
        return pipeline.metrics()

    @app.post("/predict", response_model=PredictResponse, tags=["Inference"])
    def predict(req: PredictRequest) -> PredictResponse:
        pipeline = get_pipeline()
        return pipeline.predict(req)

    @app.post("/explain", response_model=ExplainResponse, tags=["Inference"])
    def explain(req: ExplainRequest) -> ExplainResponse:
        pipeline = get_pipeline()
        return pipeline.explain(req)

    return app


app = create_app()
