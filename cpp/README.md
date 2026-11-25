# BearingBench Native C++20 Edge Runner

Zero-Python, single-threaded **C++20** inference and verification binary for the quantized **35.77 KB** `order_mrl_int8.onnx` model using the ONNX Runtime C++ API (`Ort::Session` wrapped in RAII `bearingbench::EdgeRunner` with `std::span` and `std::chrono::steady_clock`).

## Build & Run

From the repository root:

```bash
# 1. Fetch official Microsoft.ML.OnnxRuntime 1.32.0 C++ headers & shared libraries into .cache/
python scripts/fetch_onnxruntime.py

# 2. Configure & build C++20 release binary
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel

# 3. Verify 5/5 prediction parity against Python EdgeClassifier and benchmark latency
./cpp/build/bearingbench-edge --model artifacts/order_mrl_int8.onnx --verify-ref cpp/test_data/reference_windows.csv --runs 500 --warmup 50
```

*(On Windows without Visual Studio `cl.exe` in `PATH`, pass `-G Ninja`, `-DCMAKE_CXX_COMPILER=.../.cache/zig-cxx.cmd`, `-DCMAKE_AR=.../.cache/zig-ar.cmd`, and `-DCMAKE_RANLIB=.../.cache/zig-ranlib.cmd`.)*

## Measured Single-Thread CPU Latency (`500` runs, `50` warmup)

Measured on the same machine against `artifacts/order_mrl_int8.onnx` (`35.77 KB`) across 5 held-out uOttawa Trial-3 Profile-C windows:

| Runtime | Parity vs. Python | Mean | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| **Python `EdgeClassifier`** (`onnxruntime 1.32.0`) | reference | `50.3 µs` | `48.1 µs` | `57.9 µs` | `79.7 µs` |
| **C++20 `bearingbench-edge`** (`Ort::Session`) | **`5/5` (`4.43e-9` worst diff)** | **`49.6 µs`** | **`46.6 µs`** | **`47.4 µs`** | **`66.3 µs`** |

Both runtimes execute the same underlying ONNX Runtime C++ CPU kernel at median latency (`~47–48 µs`), while removing the Python wrapper tightens `p95` and `p99` tail latency by **17–18%** with zero interpreter or runtime dependencies.
