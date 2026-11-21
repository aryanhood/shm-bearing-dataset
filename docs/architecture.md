# Architecture

BearingScope turns 12 kHz drive-end vibration windows into an explainable maintenance verdict (`SAFE`, `WATCH`, `CRITICAL`) and runs identically in Python and in the browser.

```text
raw 12 kHz signal (2048 samples ≈ 171 ms)
  │
  ├─► Time-domain statistics (5): log RMS, kurtosis, crest factor, skewness, shape factor
  ├─► Spectral band energy   (8): log energy share across 8 equal FFT bands (0–6 kHz)
  └─► Hilbert envelope       (3): 1.5–5.5 kHz bandpass -> analytic envelope -> FFT
                                  local peak-to-ring SNR at 1×/2×/3× BPFI, BPFO, BSF
                │
                ▼
     16-element feature vector
                │
                ▼
     StandardScaler -> 16×32×4 ReLU MLP -> Softmax  (14 KB JSON artifact)
                │
                ▼
     Decision policy:
       normal ≥ 0.80  -> SAFE
       fault  ≥ 0.80  -> CRITICAL  (identifies specific defect)
       otherwise      -> WATCH
```

## Two runtimes, one numerical contract

- **Offline / CLI (`bearingscope/`)**: downloads CWRU `.mat` recordings, extracts features, runs grouped cross-validation (`unseen_load` and `unseen_fault_size`), trains the 0–2 HP model, and exports `model.json`, `demo.i16`, `demo.json`, and `golden.json`.
- **Browser (`web/src/lib/`)**: implements the exact same radix-2 FFT, Hilbert envelope demodulation, 16-feature extractor, and JSON MLP forward pass in TypeScript with zero backend calls.
- **Parity gate (`web/src/lib/pipeline.test.ts` + `tests/test_pipeline.py`)**: both suites assert that TypeScript and Python produce identical features, envelope spectra, and class probabilities within `1e-6` relative tolerance on real held-out 3 HP recordings.

## Architecture Decision Records

- [ADR 0001: Hilbert envelope demodulation + 16-feature vector](adr/0001-envelope-demodulation-features.md)
- [ADR 0002: Zero-backend browser inference via 14 KB JSON MLP](adr/0002-zero-backend-browser-inference.md)
- [ADR 0003: Grouped evaluation protocols (unseen load vs. unseen fault size)](adr/0003-grouped-evaluation-protocols.md)
