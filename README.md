# SHM Decision-Support System (Production-Oriented)

An end-to-end Structural Health Monitoring (SHM) system that converts vibration windows into actionable maintenance decisions:

- Class prediction (`Normal`, `Inner Race`, `Outer Race`, `Ball Fault`)
- Health index (0 to 1)
- Operational status (`SAFE`, `WARNING`, `CRITICAL`)
- Auditable rationale for action

This repo is intentionally structured to demonstrate engineering depth, not just model training.

## Problem

Industrial rotating equipment degrades gradually, while maintenance teams often receive noisy and ambiguous vibration signals. A useful system must:

- classify fault type reliably
- detect out-of-distribution behavior
- convert model output into decision policy
- remain reproducible and debuggable across CLI/API/dashboard paths

## Approach

```text
signal window
  -> preprocessor (train-only scaling + deterministic split)
  -> classifier (RF feature model or CNN1D raw-window model)
  -> anomaly scoring (IsolationForest)
  -> decision agent (policy thresholds + rationale)
  -> API/CLI/dashboard response
```

Core modules:

- `src/data`: CWRU-style loading, synthetic generation fallback, split/scaling
- `src/features`: time/frequency feature extraction for RF + anomaly
- `src/models`: RandomForest, CNN1D, IsolationForest
- `src/inference`: centralized inference pipeline with compatibility and integrity checks
- `src/agent`: rule-based decision synthesis
- `src/api`: FastAPI surface (`/health`, `/metrics`, `/predict`, `/explain`)
- `pipelines`: train/evaluate/predict/ablation workflows

## Engineering Upgrades Added

Recent hardening changes focus on realistic production constraints:

- Serving controls:
  - `serving.max_batch_size` to prevent unbounded batch memory spikes
  - bounded TTL prediction cache (`serving.prediction_cache`) for repeat-window latency reduction
- Artifact integrity:
  - SHA-256 checksums stored in manifest during training
  - checksum verification during pipeline load to detect corruption/drift
- Runtime observability:
  - `GET /metrics` (requests, failure rate, latency percentiles, cache hit rate)
  - per-request ID and process-time headers in API middleware
- Input and data safety:
  - overlap validation in data loader (`0 <= overlap < 1`)
  - clear failure if CWRU files exist but yield zero valid windows
  - strict preprocessor validation for shape/class-count stratification viability
- Config robustness:
  - BOM-safe YAML loading (`utf-8-sig`) to avoid silent key corruption (`project` vs `\ufeffproject`)

## Tradeoffs

- Cache vs correctness:
  - Cache keys include model choice + manifest fingerprint + signal hash.
  - This preserves correctness across artifact changes at the cost of extra key computation.
- Strict compatibility vs flexibility:
  - Startup blocks on checksum/manifest mismatch.
  - Safer for production, stricter for quick experimentation.
- Batch guardrails vs throughput:
  - Max batch size protects latency and memory stability.
  - Large offline scoring runs should use chunked processing.

## Quick Start

Use Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Train artifacts:

```bash
python -m shm train
```

Evaluate:

```bash
python -m shm evaluate
```

Serve API:

```bash
python -m shm serve
```

Run dashboard:

```bash
streamlit run app/dashboard.py
```

Run tests:

```bash
pytest -q
```

## API Surface

- `GET /health`: artifact readiness + compatibility report
- `GET /metrics`: runtime telemetry (latency p50/p95/max, failure rate, cache stats)
- `POST /predict`: one-window prediction with decision output
- `POST /explain`: RF feature-importance and signal diagnostics

Example:

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"rf\",\"signal\":[0.0,0.1,...]}"
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Engineering Audit](docs/ENGINEERING_AUDIT.md)
- [Roadmap](docs/ROADMAP.md)
- [Contributing](CONTRIBUTING.md)

## Resume-Ready Highlights

- Built a production-style SHM decision-support service with unified inference contracts across CLI, API, and dashboard.
- Added artifact compatibility and SHA-256 integrity checks to prevent serving stale or tampered model bundles.
- Implemented bounded TTL caching, batch guardrails, and runtime telemetry (`/metrics`) to improve latency and operational visibility.
- Hardened data/config pipelines by fixing UTF-8 BOM parsing issues and adding strict validation for stratified preprocessing edge cases.
