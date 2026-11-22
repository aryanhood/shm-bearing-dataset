"""Tests for Matryoshka 1D-CNN slicing, calibration, Split APS, Online ACI, and edge ONNX."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from bearingbench.baselines.unsupervised import (
    SignalData,
    audit_signal,
    compute_iso_velocity,
    learn_baseline,
    score_against_baseline,
)
from bearingbench.data.encoder import SensorAudit
from bearingbench.edge.onnx_edge import EdgeClassifier, export_onnx, quantize_onnx_int8
from bearingbench.edge.supervisor import SupervisoryController
from bearingbench.evaluation.stats import mcnemar_exact, mean_std, paired_bootstrap_acc_diff
from bearingbench.models.cnn import (
    NESTED_DIMS,
    SlicedMatryoshkaCNN,
    TrainConfig,
    predict_logits_torch,
    train_matryoshka_cnn,
)
from bearingbench.uncertainty.calibration import (
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
    selective_risk_coverage,
    softmax,
)
from bearingbench.uncertainty.conformal import (
    SplitAPSConformal,
    run_adaptive_conformal_inference,
)


def test_matryoshka_slice_and_onnx_int8_parity(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, 4, 400)).astype(np.float32)
    y = (np.arange(40) % 5).astype(np.int64)
    model = train_matryoshka_cnn(
        x, y, n_classes=5, cfg=TrainConfig(epochs=2, batch_size=16, seed=7, n_threads=1)
    )
    for dim in (8, 32, 128):
        sliced = SlicedMatryoshkaCNN(model, dim)
        with torch.no_grad():
            expected = model.logits_at_dim(model.encode(torch.from_numpy(x[:6])), dim).numpy()
        actual = predict_logits_torch(sliced, x[:6])
        assert float(np.max(np.abs(expected - actual))) < 1e-5

    fp32 = tmp_path / "m_fp32.onnx"
    int8 = tmp_path / "m_int8.onnx"
    meta = tmp_path / "meta.json"
    export_onnx(SlicedMatryoshkaCNN(model, 64), fp32, meta)
    quantize_onnx_int8(fp32, int8, x)

    e32 = EdgeClassifier(fp32, meta)
    e8 = EdgeClassifier(int8, meta)
    p32 = e32.predict_proba(x[:8])
    p8 = e8.predict_proba(x[:8])
    assert p32.shape == (8, 5)
    assert p8.shape == (8, 5)
    assert float(np.max(np.abs(p32 - p8))) < 0.05
    lat = e8.benchmark_latency(n_runs=5, warmup=2)
    assert lat["p50_ms"] >= 0.0
    assert NESTED_DIMS == (8, 16, 32, 64, 128)


def test_calibration_and_conformal_invariants() -> None:
    rng = np.random.default_rng(19)
    y = np.repeat(np.arange(4), 50).astype(np.int64)
    logits = rng.normal(0.0, 0.7, size=(200, 4))
    logits[np.arange(200), y] += 2.5

    probs = softmax(logits)
    assert 0.0 <= expected_calibration_error(probs, y) <= 1.0
    assert 0.0 <= brier_score(probs, y) <= 2.0
    rc = selective_risk_coverage(probs, y)
    assert 0.0 <= rc.aurc <= 1.0
    assert rc.acc_at_80_cov >= rc.full_acc - 0.05

    scaler = TemperatureScaler.fit(logits[:100], y[:100])
    assert scaler.temperature > 0.0
    ts_probs = scaler.predict_proba(logits[100:])

    aps = SplitAPSConformal.fit(ts_probs[:50], y[100:150], alpha=0.10, seed=42)
    m_aps = aps.evaluate(ts_probs[50:], y[150:], seed=42)
    assert 0.6 <= m_aps.coverage <= 1.0
    assert m_aps.avg_set_size >= 1.0

    aci = run_adaptive_conformal_inference(
        aps.cal_scores, ts_probs[50:], y[150:], alpha_target=0.10, gamma=0.05
    )
    assert 0.6 <= aci.metrics.coverage <= 1.0
    assert len(aci.rolling_coverage) == 50


def test_supervisor_conformal_guard_and_unsupervised_baseline() -> None:
    ctrl = SupervisoryController(window_k=2, window_n=3, derate_prob=0.60, recovery_clean_steps=2)
    ok_audit = SensorAudit(True, 0.0, 0.0, 0.0, 0.5, 500, 1200.0, 1400.0, ())
    # Low argmax fault_prob (0.45), but ambiguous conformal set includes a fault -> DERATE
    probs_ambig = [0.55, 0.35, 0.05, 0.03, 0.02]
    ctrl.update(probs_ambig, ok_audit, time_s=0.5, rpm=1300.0, conformal_set=(0, 1))
    cmd = ctrl.update(probs_ambig, ok_audit, time_s=1.0, rpm=1310.0, conformal_set=(0, 1))
    assert cmd.mode == "DERATE"
    assert "conformal_ambiguity_derate" in cmd.reason

    # Verify analytical ISO 20816-3 velocity integration on 50 Hz sine wave + light noise
    rng = np.random.default_rng(11)
    fs = 12_000.0
    t = np.arange(12_000) / fs
    v_target_rms_mm_s = 3.0
    a_amp_g = (v_target_rms_mm_s / 1000.0) * np.sqrt(2.0) * (2.0 * np.pi * 50.0) / 9.80665
    wave = a_amp_g * np.sin(2.0 * np.pi * 50.0 * t) + 0.002 * a_amp_g * rng.normal(size=len(t))
    sig = SignalData(wave, fs, "g", "sine.csv", "DE")
    assert audit_signal(sig).ok
    iso = compute_iso_velocity(sig)
    assert iso is not None
    assert abs(iso.velocity_rms_mm_s - v_target_rms_mm_s) < 0.05
    assert iso.zone == "C"

    base = learn_baseline(sig)
    drift_same = score_against_baseline(sig, base)
    sig_noisy = SignalData(sig.values * 4.0, fs, "g", "fault.csv", "DE")
    drift_high = score_against_baseline(sig_noisy, base)
    assert drift_high.drift_score > drift_same.drift_score


def test_stats_helpers() -> None:
    y = np.array([0, 1, 2, 0, 1, 2] * 10, dtype=np.int64)
    a = y.copy()
    b = y.copy()
    b[:12] = 0
    boot = paired_bootstrap_acc_diff(y, a, b, n_resamples=200)
    assert boot["diff_mean"] > 0.0
    mc = mcnemar_exact(y, a, b)
    assert float(mc["p_value"]) < 0.05
    st = mean_std([0.90, 0.92, 0.91])
    assert abs(st.mean - 0.91) < 1e-4
