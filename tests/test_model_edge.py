"""PyTorch OrderCNN <-> ONNX Runtime numerical parity and latency tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orderguard.data import CLASSES
from orderguard.edge import EdgeClassifier
from orderguard.model import TrainConfig, export_onnx, predict_proba_torch, train_order_cnn
from orderguard.order_tracking import CARRIER_BANDS_HZ, N_ORDER_BINS


def test_pytorch_onnx_parity_and_latency(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    n = 40
    c = len(CARRIER_BANDS_HZ)
    x = rng.normal(0.0, 1.0, size=(n, c, N_ORDER_BINS)).astype(np.float32)
    y = np.arange(n, dtype=np.int64) % len(CLASSES)

    model = train_order_cnn(x, y, TrainConfig(epochs=3, batch_size=16))
    onnx_path = tmp_path / "model.onnx"
    meta_path = tmp_path / "meta.json"
    export_onnx(model, onnx_path, meta_path)

    edge = EdgeClassifier(onnx_path, meta_path)
    pt_probs = predict_proba_torch(model, x)
    ort_probs = edge.predict_proba(x)

    assert pt_probs.shape == (n, len(CLASSES))
    assert np.max(np.abs(pt_probs - ort_probs)) < 1e-5
    assert np.max(np.abs(pt_probs[:1] - edge.predict_proba(x[0]))) < 1e-5

    lat = edge.benchmark_latency(n_runs=50, warmup=5)
    assert lat["p50_ms"] > 0.0
    assert lat["p99_ms"] >= lat["p50_ms"]
