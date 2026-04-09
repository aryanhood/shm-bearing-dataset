# Engineering Audit and Redesign

This document captures the engineering reasoning behind the hardening pass.

## Phase 1: Deep Project Audit

### System understanding

Problem solved:

- Detect bearing faults from vibration windows.
- Convert model outputs into maintenance actions.
- Keep train/eval/inference/API behavior consistent.

Current architecture (before hardening):

- Data loading/preprocessing in `src/data`.
- RF/CNN/IsolationForest models in `src/models`.
- Centralized inference in `src/inference/pipeline.py`.
- Rule-based decision agent in `src/agent`.
- FastAPI and Streamlit interfaces.

Primary data flow:

1. Load windows (`synthetic` or CWRU files).
2. Stratified split, fit scaler on train split.
3. Train models and persist artifacts.
4. Load artifacts for inference/API.
5. Produce class probabilities + anomaly score.
6. Apply decision policy and return action.

### Critical weaknesses (ranked)

1. Config key corruption risk due UTF-8 BOM.
   - `project` key became `\ufeffproject`, silently breaking version/seed reads.
2. No artifact integrity validation.
   - Model files could be modified/corrupted without detection.
3. No serving telemetry.
   - No visibility into request failure rate, latency regressions, or cache effectiveness.
4. Unbounded serving behavior for batch predictions.
   - Potential latency and memory blow-ups under large request payloads.
5. Repeated inference recomputation.
   - No caching for identical windows (common in dashboard/replay workflows).
6. Weak data-ingestion guardrails.
   - CWRU path could fail with unclear errors when files load but produce zero windows.
7. Preprocessor accepted invalid stratification cases.
   - Small classes could fail deep inside sklearn split calls.
8. Loader allowed unsafe overlap values.
   - `overlap >= 1.0` could make segmentation invalid.
9. Runtime compatibility checks verified schema only.
   - Did not ensure bytes on disk matched manifest intent.

## Phase 2: Production Problem Discovery

### Real-world constraints introduced

- Reliability: serving must reject incompatible/tampered artifacts before prediction.
- Latency: p95 latency must be trackable and bounded.
- Capacity: API must reject oversized batches rather than degrade all traffic.
- Operability: teams need request IDs, process-time headers, and metrics visibility.
- Data quality: loader and splitter must fail early with clear remediation guidance.

### Where original design breaks

- If config file contains BOM, `project.version` and `project.seed` reads become unreliable.
- If artifact file bytes change post-training, service still starts and serves stale/wrong models.
- If one user sends huge batch input, system has no hard limit and can starve other requests.
- If repeated identical signals are queried, latency scales linearly due no response reuse.

## Phase 3: System Redesign

### Updated serving architecture

1. Train pipeline writes manifest with:
   - schema compatibility fields
   - available-model flags
   - SHA-256 hashes for each artifact
2. Inference pipeline startup:
   - validates compatibility fields
   - validates artifact availability
   - validates artifact checksums
3. Request path:
   - optional TTL cache lookup (`model + manifest fingerprint + signal hash`)
   - bounded batch size enforcement
   - telemetry recording (latency/failures)
4. API layer:
   - request context middleware (`X-Request-Id`, `X-Process-Time-Ms`)
   - `/metrics` endpoint for runtime visibility

### Why this design

- Chosen:
  - In-process cache: simplest latency optimization with low operational overhead.
  - Hash-based manifest integrity: high confidence drift detection without external registry.
  - Inference-owned telemetry: captures true model-serving latency, not just API wrapper time.
- Alternatives considered:
  - External cache (Redis): more scalable but unnecessary complexity for this stage.
  - Full model registry service: ideal long-term, but too heavy for internship portfolio scope.
  - Async queue-based inference: useful at very high concurrency, but increases architecture complexity.

### Tradeoffs

- Performance vs simplicity:
  - In-process cache is simple but not shared across replicas.
- Scalability vs cost:
  - Hard batch limits protect stability but may require client-side chunking.
- Speed vs maintainability:
  - Strict manifest/hash checks can block startup, but drastically reduce silent failure risk.

## Phase 4: Implementation Upgrades

### Code quality and structure

- Added `src/inference/cache.py` (thread-safe TTL cache with eviction stats).
- Expanded pipeline with explicit `metrics()` API and runtime accounting.
- Added stronger validation in `src/data/loader.py` and `src/data/preprocessor.py`.
- Made config loading BOM-safe in `src/utils/config.py`.

### Logging and error handling

- API middleware now logs request method/path/status/latency with request ID.
- Inference pipeline now returns explicit `batch_size_exceeded` with HTTP 413 semantics.
- Data loader now raises actionable error when CWRU files yield zero windows.

### Configuration management

- Added serving controls in `configs/config.yaml`:
  - `serving.max_batch_size`
  - `serving.prediction_cache.enabled`
  - `serving.prediction_cache.max_entries`
  - `serving.prediction_cache.ttl_seconds`

### Advanced concept introduced

- Bounded TTL response cache in serving path:
  - Reduces repeat inference latency.
  - Includes cache hit/miss/eviction observability.
  - Invalidates automatically via TTL and manifest fingerprint changes.

## Phase 5: Debugging Mindset (Failure Scenarios)

1. Artifact tampering/corruption.
   - Detection: checksum mismatch during pipeline load.
   - Root cause: artifact bytes differ from training manifest.
   - Fix: retrain or restore artifacts to manifest-matching bundle.

2. Config silently not applying (`project.version` unknown).
   - Detection: manifest showed `project_version: unknown`.
   - Root cause: UTF-8 BOM prefixed first YAML key.
   - Fix: load YAML with `utf-8-sig`.

3. Latency spikes from repeated identical inference.
   - Detection: `/metrics` p95 latency high with low cache hit rate.
   - Root cause: recomputation path on repeated windows.
   - Fix: enable TTL cache and verify hit rate in `/metrics`.

4. Request storms with oversized batch payloads.
   - Detection: 413 responses tagged `batch_size_exceeded`.
   - Root cause: client sending unbounded batch size.
   - Fix: enforce `serving.max_batch_size`, chunk at client side.

5. Stratified split failures on small class counts.
   - Detection: explicit preprocessor validation error with per-class counts.
   - Root cause: classes with <3 samples cannot be split train/val/test safely.
   - Fix: collect more data or rebalance class distribution.

## Phase 6: Engineering Thinking Proof (Decision Log)

1. Problem identified:
   - Silent config corruption from BOM.
   - Decision:
   - BOM-safe file loading (`utf-8-sig`).
   - Ambiguity:
   - Could sanitize keys after parse vs fix file decoding.
   - Resolution:
   - Decoding fix chosen as lowest-risk and globally correct.
   - Tradeoff:
   - Slightly less explicit than key-sanitization pipeline, but cleaner.

2. Problem identified:
   - Artifact compatibility without integrity is incomplete.
   - Decision:
   - Add SHA-256 checksums to manifest and validate at load.
   - Ambiguity:
   - Per-file hash only vs signed manifests.
   - Resolution:
   - Per-file hashes chosen for pragmatic reliability.
   - Tradeoff:
   - No cryptographic signing yet, but strong accidental-corruption detection.

3. Problem identified:
   - No latency/capacity controls in serving path.
   - Decision:
   - Add TTL cache and max batch guardrails.
   - Ambiguity:
   - External cache vs local in-process cache.
   - Resolution:
   - Local cache chosen for simplicity and portfolio clarity.
   - Tradeoff:
   - Works per-process only; horizontal scaling needs shared cache later.

4. Problem identified:
   - No runtime visibility for operational debugging.
   - Decision:
   - Add `/metrics` and request-level context headers.
   - Ambiguity:
   - Prometheus exporter vs typed JSON endpoint.
   - Resolution:
   - Typed JSON endpoint chosen for low-friction integration.
   - Tradeoff:
   - Less standard than Prometheus, easier for this scope.

## Phase 7: Recruiter Story Upgrade

### Resume bullets

- Designed and implemented a production-style SHM inference service unifying model contracts across training, API, and dashboard clients.
- Added artifact integrity guarantees with SHA-256 manifest checks, preventing stale/tampered model serving.
- Built latency and reliability controls (TTL response cache, max batch guardrails, failure-rate and p95 telemetry endpoint).
- Diagnosed and fixed a hidden UTF-8 BOM configuration bug that silently broke version/seed propagation in pipeline metadata.
- Added validation-driven failure handling for ingestion and preprocessing edge cases, improving debuggability and operator feedback.

### GitHub positioning

This project is positioned as:

- A fault-diagnostics ML system with explicit operational policy outputs.
- A systems-engineered service with compatibility guarantees, runtime telemetry, and failure-safe guardrails.
- A portfolio proof that demonstrates finding latent flaws, resolving ambiguity, and making explicit tradeoff-driven decisions.
