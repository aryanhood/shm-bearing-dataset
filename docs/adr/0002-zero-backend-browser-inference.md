# ADR 0002: Zero-Backend Browser Inference via 14 KB JSON MLP

## Context

Portfolio ML apps that depend on a sleeping free-tier Python API break or stall for 30+ seconds when a recruiter opens the link.

## Decision

Train a compact `StandardScaler + MLP(32, ReLU)` in scikit-learn and serialize its mean, scale, weights, and biases into a 14 KB JSON file (`web/public/data/model.json`). Port the FFT, Hilbert envelope, feature extractor, and MLP forward pass directly to TypeScript (`web/src/lib/`), and enforce `< 1e-6` numerical parity against Python in CI.

## Tradeoffs

- **Pros:** Instant load on GitHub Pages or Vercel, zero hosting cost, works offline, and lets visitors upload their own 12 kHz CSV without data leaving their browser.
- **Cons:** Maintaining two implementations of the feature extractor requires strict parity tests (`pipeline.test.ts` + `test_pipeline.py`) so Python and TypeScript never drift.
