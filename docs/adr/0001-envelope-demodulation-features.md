# ADR 0001: Hilbert Envelope Demodulation + 16-Feature Vector

## Context

A localized bearing defect excites a high-frequency structural resonance (~1.5–5.5 kHz) every time a rolling element strikes the spall. In the raw FFT, these low-energy impacts are buried under shaft harmonics and motor noise.

## Decision

Extract 16 interpretable features per 2048-sample (171 ms) window:
- 5 time-domain shape statistics (`log_rms`, `kurtosis`, `crest_factor`, `skewness`, `shape_factor`)
- 8 log spectral band energies spanning 0–6 kHz
- 3 Hilbert envelope peak-to-ring SNR scores summed over the 1×, 2×, and 3× harmonics of the SKF 6205-2RS characteristic defect frequencies (`BPFI = 5.4152×`, `BPFO = 3.5848×`, `BSF = 2.3570×` shaft frequency)

## Tradeoffs

- **Pros:** Every prediction can be audited visually on the envelope spectrum; the feature vector is small enough for a 14 KB classifier; no GPU or PyTorch runtime is required.
- **Cons:** Characteristic frequencies depend on knowing the bearing geometry and shaft RPM. Severe 0.014"/0.021" spalls can alter the resonance band or smear impact timing, which is why unseen-fault-size accuracy drops to 59.2%.
