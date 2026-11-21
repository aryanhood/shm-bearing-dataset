# ADR 0003: Grouped Evaluation Protocols (Unseen Load vs. Unseen Fault Size)

## Context

Randomly splitting overlapping vibration windows from the same CWRU recording produces near-100% accuracy even with weak models because adjacent windows are nearly identical.

## Decision

Cut non-overlapping 2048-sample windows and report two strictly recording-grouped cross-validation protocols:
1. **Unseen load (4-fold):** train on 3 motor loads, test on the held-out 4th load (`99.91%` accuracy, `0.999` macro F1).
2. **Unseen fault size (3-fold):** train on 2 fault diameters, test on the unseen 3rd diameter (`59.15%` accuracy, `0.669` macro F1).

Furthermore, the live browser demo model is trained strictly on loads 0–2 HP and streams only the held-out 3 HP recordings.

## Tradeoffs

- **Pros:** No data leakage; surfaces the real failure mode of bearing classifiers (healthy vs. faulty stays 100% separable, while cross-diameter fault-type classification degrades because spall geometry shifts the resonance spectrum).
- **Cons:** Headline numbers look less flattering than a leaky random split, which is intentional.
