# Architecture

## System intent

The system converts vibration signals into operational maintenance guidance through a reproducible, auditable SHM pipeline.

```text
Input -> Processing -> Output -> Impact
```

- Input: raw vibration windows and project configuration
- Processing: preprocessing, model inference, anomaly detection, decision synthesis
- Output: structured diagnostic contract (model + agent outputs)
- Impact: maintenance prioritization and risk-aware engineering decisions

## High-level components

1. `src/data`
- `loader.py`: synthetic and CWRU-compatible loading
- `preprocessor.py`: stratified splits and train-only scaling

2. `src/features`
- `extractor.py`: time/frequency engineered features for RF and anomaly model

3. `src/models`
- `random_forest.py`: interpretable feature baseline
- `cnn1d.py`: raw-window deep model
- `anomaly.py`: Isolation Forest severity signal

4. `src/inference`
- `pipeline.py`: centralized serving and batch inference logic
- manifest compatibility and SHA-256 integrity checks between config and artifacts
- bounded TTL prediction cache and runtime telemetry counters

5. `src/agent`
- `decision_agent.py`: converts model output into `SAFE/WARNING/CRITICAL`

6. `src/api`
- FastAPI surface for `/health`, `/predict`, `/explain`
- Runtime telemetry endpoint `/metrics`
- shared error contracts for predictable client behavior

7. `pipelines/`
- `train.py`: train and snapshot artifacts
- `evaluate.py`: offline model evaluation with plots/reports
- `predict.py`: local inference utility
- `ablation.py`: controlled sensitivity studies

## Data and control flow

1. Load data (`synthetic` or `CWRU`).
2. Create deterministic split and fit scaler on train partition.
3. Train selected models.
4. Persist artifacts + manifest + config snapshot.
5. Load artifacts in a centralized inference pipeline.
6. Generate model output contract.
7. Apply decision-agent policy.
8. Serve through API/dashboard/CLI.

## Contracts and boundaries

- `src/contracts.py` defines canonical request/response schemas.
- API, dashboard, and pipelines share this contract to prevent drift.
- Manifest compatibility check blocks serving with mismatched config/artifacts.

## Reproducibility guarantees

- Fixed seed control (`src/utils/seed.py`)
- Versioned run snapshots in `artifacts/runs/`
- Serialized preprocessor and model artifacts
- Config snapshots for rerun parity

## Extensibility points

- Add a new model by implementing `predict_proba` contract and registering in inference selection.
- Add decision policy variants in `src/agent` without changing model internals.
- Add new datasets via `src/data/loader.py` adapters.
- Add deployment adapters around FastAPI without changing core contracts.

## Non-goals (current)

- Real-time edge streaming and continuous state estimation
- Digital-twin integration
- Full MLOps orchestration and auto-retraining

These are tracked in `docs/ROADMAP.md`.
