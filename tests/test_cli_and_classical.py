"""Tests for CLI inspection on bundled examples and classical tabular baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bearingbench.cli import main
from bearingbench.models.classical import predict_proba_sklearn, train_mlp, train_random_forest


def test_cli_inspect_on_bundled_examples(capsys: object) -> None:
    root = Path(__file__).resolve().parent.parent
    rc = main(
        [
            "inspect",
            str(root / "examples" / "motor_inner_fault.csv"),
            "--rpm",
            "1750",
            "--bearing",
            "6205",
        ]
    )
    assert rc == 0


def test_classical_tabular_pipelines_handle_missing_classes() -> None:
    rng = np.random.default_rng(7)
    x_tr = rng.normal(size=(30, 10))
    y_tr = np.repeat(np.array([0, 1, 2], dtype=np.int64), 10)
    rf = train_random_forest(x_tr, y_tr, seed=7, n_estimators=20)
    mlp = train_mlp(x_tr, y_tr, seed=7, hidden_layer_sizes=(8,))
    p_rf = predict_proba_sklearn(rf, x_tr[:5], n_classes=4)
    p_mlp = predict_proba_sklearn(mlp, x_tr[:5], n_classes=4)
    assert p_rf.shape == (5, 4)
    assert p_mlp.shape == (5, 4)
    assert np.allclose(p_rf[:, 3], 0.0)
