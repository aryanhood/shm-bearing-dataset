"""Tests for exported web studio JSON and C++ reference verification CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from bearingbench.edge.onnx_edge import EdgeClassifier


def test_exported_cpp_reference_windows_match_onnx() -> None:
    root = Path(__file__).resolve().parent.parent
    csv_path = root / "cpp" / "test_data" / "reference_windows.csv"
    assert csv_path.exists()

    clf = EdgeClassifier(
        root / "artifacts" / "order_mrl_int8.onnx",
        root / "artifacts" / "order_mrl_meta.json",
    )
    with csv_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 5
    for row in rows:
        tensor = np.asarray([float(row[f"f{i}"]) for i in range(1600)], dtype=np.float32).reshape(
            1, 4, 400
        )
        probs = clf.predict_proba(tensor)[0]
        expected = np.asarray([float(row[f"p{k}"]) for k in range(5)], dtype=np.float64)
        assert clf.classes[int(np.argmax(probs))] == row["true_class"]
        assert float(np.max(np.abs(probs - expected))) < 1e-6


def test_exported_web_samples_bundle_is_consistent() -> None:
    root = Path(__file__).resolve().parent.parent
    bundle_path = root / "web" / "public" / "data" / "motor_samples.json"
    assert bundle_path.exists()

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert payload["model"]["classes"] == ["healthy", "inner", "outer", "ball", "combined"]
    assert (root / "web" / "public" / "models" / "order_mrl_int8.onnx").exists()
    assert len(payload["onnx_samples"]) == 5

    clf = EdgeClassifier(
        root / "web" / "public" / "models" / "order_mrl_int8.onnx",
        root / "web" / "public" / "models" / "order_mrl_meta.json",
    )
    for sample in payload["onnx_samples"]:
        tensor = np.asarray(sample["tensor_4x400"], dtype=np.float32).reshape(1, 4, 400)
        probs = clf.predict_proba(tensor)[0]
        assert clf.classes[int(np.argmax(probs))] == sample["true_class"]
        assert float(np.max(np.abs(probs - np.asarray(sample["python_probs"])))) < 1e-5

    healthy_ref = payload["reference_examples"]["healthy"]
    fault_ref = payload["reference_examples"]["inner_fault"]
    assert healthy_ref["iso_zone"] == "A"
    assert fault_ref["iso_zone"] == "A"
    assert fault_ref["drift_score"] > healthy_ref["drift_score"] + 30.0
