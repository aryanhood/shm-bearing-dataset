"""Zero-PyTorch ONNX Runtime edge inference, static INT8 QDQ quantization & latency profiler."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

DEFAULT_CLASSES: tuple[str, ...] = ("healthy", "inner", "outer", "ball", "combined")

if TYPE_CHECKING:
    from torch import nn


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / exp.sum(axis=-1, keepdims=True), dtype=np.float64)


def export_onnx(
    model: nn.Module,
    onnx_path: Path,
    meta_path: Path | None = None,
    *,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
) -> Path:
    """Export a PyTorch 1D-CNN (OrderCNN or SlicedMatryoshkaCNN) to single-file ONNX."""
    import onnx
    import torch

    model.eval()
    in_channels = int(getattr(model, "in_channels", 4))
    n_bins = int(getattr(model, "n_bins", 400))
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, in_channels, n_bins, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        export_params=True,
        opset_version=18,
        input_names=["order_ses"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}},
        dynamo=True,
        external_data=False,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    if meta_path is not None:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "classes": list(classes),
            "in_channels": in_channels,
            "n_bins": n_bins,
            "onnx_bytes": onnx_path.stat().st_size,
            "params": sum(p.numel() for p in model.parameters()),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return onnx_path


def quantize_onnx_int8(
    fp32_path: Path,
    int8_path: Path,
    calib_x: NDArray[np.float32],
    *,
    max_calib_samples: int = 256,
) -> Path:
    """Static QDQ INT8 quantization calibrated strictly on proper-training windows."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    int8_path.parent.mkdir(parents=True, exist_ok=True)
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    sub_x = np.asarray(calib_x[:max_calib_samples], dtype=np.float32)

    class _ArrayCalibrationReader(CalibrationDataReader):  # type: ignore[misc]
        def __init__(self, data: NDArray[np.float32], name: str) -> None:
            self._data = data
            self._name = name
            self._idx = 0

        def get_next(self) -> dict[str, NDArray[np.float32]] | None:
            if self._idx >= len(self._data):
                return None
            batch = self._data[self._idx : self._idx + 1]
            self._idx += 1
            return {self._name: batch}

        def rewind(self) -> None:
            self._idx = 0

    prep_path = int8_path.with_suffix(".prep.onnx")
    try:
        quant_pre_process(str(fp32_path), str(prep_path), skip_symbolic_shape=False)
        model_to_quant = prep_path if prep_path.exists() else fp32_path
    except Exception:
        model_to_quant = fp32_path

    quantize_static(
        model_input=str(model_to_quant),
        model_output=str(int8_path),
        calibration_data_reader=_ArrayCalibrationReader(sub_x, input_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
    )
    if prep_path.exists():
        prep_path.unlink(missing_ok=True)
    return int8_path


class EdgeClassifier:
    """Single-thread CPU ONNX Runtime classifier (zero PyTorch dependency)."""

    def __init__(self, onnx_path: Path, meta_path: Path | None = None) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        self.onnx_path = onnx_path
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
        self.classes: tuple[str, ...] = DEFAULT_CLASSES
        self.in_channels: int = (
            int(inp.shape[1]) if len(inp.shape) == 3 and isinstance(inp.shape[1], int) else 4
        )
        self.n_bins: int = (
            int(inp.shape[2]) if len(inp.shape) == 3 and isinstance(inp.shape[2], int) else 400
        )
        if meta_path is not None and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.classes = tuple(cast(list[str], meta.get("classes", list(DEFAULT_CLASSES))))
            self.in_channels = int(meta.get("in_channels", self.in_channels))
            self.n_bins = int(meta.get("n_bins", self.n_bins))

    def predict_logits(self, order_ses: NDArray[np.float32]) -> NDArray[np.float64]:
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
        return _softmax(self.predict_logits(order_ses))

    def benchmark_latency(self, n_runs: int = 500, warmup: int = 50) -> dict[str, float]:
        """Measure single-window single-thread CPU latency in milliseconds."""
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
            "size_kb": round(self.onnx_path.stat().st_size / 1024.0, 2),
            "mean_ms": round(float(arr.mean()), 4),
            "p50_ms": round(float(np.percentile(arr, 50)), 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
            "p99_ms": round(float(np.percentile(arr, 99)), 4),
        }
