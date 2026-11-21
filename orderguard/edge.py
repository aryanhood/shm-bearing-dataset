"""Zero-PyTorch ONNX Runtime edge inference and latency profiler."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from .data import CLASSES
from .order_tracking import CARRIER_BANDS_HZ, N_ORDER_BINS


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / exp.sum(axis=-1, keepdims=True), dtype=np.float64)


class EdgeClassifier:
    """CPU edge classifier backed purely by onnxruntime + NumPy."""

    def __init__(self, onnx_path: Path, meta_path: Path | None = None) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.output_name = self.session.get_outputs()[0].name
        self.classes: tuple[str, ...] = CLASSES
        self.in_channels: int = (
            int(inp.shape[1])
            if len(inp.shape) == 3 and isinstance(inp.shape[1], int)
            else len(CARRIER_BANDS_HZ)
        )
        self.n_bins: int = (
            int(inp.shape[2])
            if len(inp.shape) == 3 and isinstance(inp.shape[2], int)
            else N_ORDER_BINS
        )
        if meta_path is not None and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.classes = tuple(meta.get("classes", CLASSES))
            self.in_channels = int(meta.get("in_channels", self.in_channels))
            self.n_bins = int(meta.get("n_bins", self.n_bins))

    def predict_logits(self, order_ses: NDArray[np.float32]) -> NDArray[np.float64]:
        """Run inference on shape (B, C, n_bins) or (C, n_bins) returning logits."""
        arr = np.asarray(order_ses, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3 or arr.shape[1] != self.in_channels or arr.shape[2] != self.n_bins:
            raise ValueError(
                f"expected shape (B, {self.in_channels}, {self.n_bins}), got {arr.shape}"
            )
        out = self.session.run([self.output_name], {self.input_name: arr})[0]
        return np.asarray(out, dtype=np.float64)

    def predict_proba(self, order_ses: NDArray[np.float32]) -> NDArray[np.float64]:
        """Run inference and return class probabilities."""
        return _softmax(self.predict_logits(order_ses))

    def benchmark_latency(self, n_runs: int = 500, warmup: int = 50) -> dict[str, float]:
        """Measure single-window CPU inference latency in milliseconds."""
        sample = np.zeros((1, self.in_channels, self.n_bins), dtype=np.float32)
        for _ in range(warmup):
            self.session.run([self.output_name], {self.input_name: sample})
        times_ms: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.session.run([self.output_name], {self.input_name: sample})
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(times_ms, dtype=np.float64)
        return {
            "mean_ms": round(float(arr.mean()), 4),
            "p50_ms": round(float(np.percentile(arr, 50)), 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
            "p99_ms": round(float(np.percentile(arr, 99)), 4),
        }
