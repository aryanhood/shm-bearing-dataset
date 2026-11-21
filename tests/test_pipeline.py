"""Unit and integration tests for BearingScope (runs without downloading CWRU .mat files)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bearingscope.cli import main as cli_main
from bearingscope.data import Windows, window
from bearingscope.decision import decide
from bearingscope.evaluate import evaluate, fold_groups
from bearingscope.features import FEATURE_NAMES, envelope_spectrum, extract, extract_many
from bearingscope.model import Model, train
from bearingscope.physics import CLASSES, SAMPLE_RATE_HZ, WINDOW, fault_frequencies


def _make_impulses(freq_hz: float, *, amp: float = 1.0, seed: int = 0) -> np.ndarray:
    """Synthetic bearing signal: damped 3 kHz resonance excited at `freq_hz`."""
    rng = np.random.default_rng(seed)
    t = np.arange(WINDOW) / SAMPLE_RATE_HZ
    x = rng.normal(0.0, 0.05, size=WINDOW)
    if freq_hz > 0:
        period = 1.0 / freq_hz
        impact_times = np.arange(0.0, t[-1], period)
        for t0 in impact_times:
            dt = t - t0
            mask = (dt >= 0) & (dt < 0.004)
            x[mask] += amp * np.exp(-dt[mask] / 0.0007) * np.sin(2 * np.pi * 3200.0 * dt[mask])
    return x


def test_fault_frequencies_match_skf_6205() -> None:
    ff = fault_frequencies(1797)
    assert ff.shaft == pytest.approx(29.95, abs=1e-2)
    assert ff.bpfi == pytest.approx(162.19, abs=0.05)
    assert ff.bpfo == pytest.approx(107.36, abs=0.05)
    assert ff.bsf == pytest.approx(70.59, abs=0.05)


def test_envelope_detects_characteristic_fault_frequency() -> None:
    rpm = 1750.0
    ff = fault_frequencies(rpm)
    inner = _make_impulses(ff.bpfi, amp=2.0, seed=1)
    outer = _make_impulses(ff.bpfo, amp=2.0, seed=2)
    normal = _make_impulses(0.0, seed=3)

    f_inner = extract(inner, rpm)
    f_outer = extract(outer, rpm)
    f_normal = extract(normal, rpm)

    assert len(f_inner) == len(FEATURE_NAMES) == 16
    assert f_inner[13] > f_inner[14]  # BPFI score dominates for inner race fault
    assert f_outer[14] > f_outer[13]  # BPFO score dominates for outer race fault
    assert f_inner[13] > f_normal[13] + 0.5
    assert envelope_spectrum(inner).freqs.shape == (WINDOW // 2,)


def test_extract_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="expected a window"):
        extract(np.zeros(100), 1750.0)
    bad = np.zeros(WINDOW)
    bad[5] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        extract(bad, 1750.0)


def test_windowing_drops_trailing_remainder() -> None:
    sig = np.arange(WINDOW * 3 + 123, dtype=np.float64)
    w = window(sig)
    assert w.shape == (3, WINDOW)
    assert w[1, 0] == WINDOW


def test_decision_thresholds() -> None:
    assert decide(np.array([0.92, 0.03, 0.03, 0.02])).status == "SAFE"
    d_crit = decide(np.array([0.01, 0.94, 0.03, 0.02]))
    assert d_crit.status == "CRITICAL"
    assert d_crit.label == "inner"
    assert decide(np.array([0.45, 0.35, 0.15, 0.05])).status == "WATCH"


def test_train_save_load_and_evaluate(tmp_path: Path) -> None:
    rpm = 1750.0
    ff = fault_frequencies(rpm)
    freqs = (0.0, ff.bpfi, ff.bpfo, ff.bsf)
    xs, ys, loads, dias, fids = [], [], [], [], []
    seed = 0
    for load in (0, 1, 2, 3):
        for cls_idx, f in enumerate(freqs):
            for dia in (0,) if cls_idx == 0 else (7, 14, 21):
                for _ in range(3):
                    xs.append(_make_impulses(f, amp=1.8, seed=seed))
                    ys.append(cls_idx)
                    loads.append(load)
                    dias.append(dia)
                    fids.append(cls_idx * 100 + load)
                    seed += 1
    data = Windows(
        x=np.stack(xs),
        y=np.array(ys, dtype=np.int64),
        load=np.array(loads, dtype=np.int64),
        diameter=np.array(dias, dtype=np.int64),
        rpm=np.full(len(ys), rpm, dtype=np.float64),
        file_id=np.array(fids, dtype=np.int64),
    )
    feats = extract_many(data.x, data.rpm)
    model = train(feats, data.y)
    path = tmp_path / "model.json"
    model.save(path)
    loaded = Model.load(path)
    np.testing.assert_allclose(model.predict_proba(feats[:5]), loaded.predict_proba(feats[:5]))

    rep_load = evaluate(data, feats, "unseen_load")
    assert rep_load.accuracy > 0.90
    assert rep_load.confusion.shape == (len(CLASSES), len(CLASSES))
    assert rep_load.to_json()["protocol"] == "unseen_load"

    rep_size = evaluate(data, feats, "unseen_fault_size")
    assert len(rep_size.folds) == 3

    with pytest.raises(ValueError, match="unknown protocol"):
        fold_groups(data, "invalid")


def test_exported_artifact_matches_golden_fixture() -> None:
    model_path = Path("artifacts/model.json")
    golden_path = Path("web/src/lib/__fixtures__/golden.json")
    assert model_path.exists() and golden_path.exists()
    model = Model.load(model_path)
    cases = json.loads(golden_path.read_text())
    for c in cases:
        f = extract(np.array(c["window"], dtype=np.float64), float(c["rpm"]))
        np.testing.assert_allclose(f, c["features"], rtol=1e-6, atol=1e-6)
        p = model.predict_proba(f)[0]
        np.testing.assert_allclose(p, c["proba"], rtol=1e-6, atol=1e-6)


def test_cli_predict_on_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv_path = tmp_path / "sample.csv"
    sig = _make_impulses(fault_frequencies(1730.0).bpfi, amp=2.0, seed=42)
    np.savetxt(csv_path, sig, fmt="%.6f")
    cli_main(["predict", str(csv_path), "--rpm", "1730"])
    out = json.loads(capsys.readouterr().out.strip())
    assert out["status"] in {"SAFE", "WATCH", "CRITICAL"}
    assert out["label"] in CLASSES
