"""
Tests — Models (RF + CNN1D + Anomaly)
"""
import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.random_forest import RandomForestModel
from src.models.anomaly       import AnomalyDetector
from src.utils.config         import CFG

N_CLASSES   = 4
WINDOW_SIZE = 256
N_TRAIN     = 80
N_TEST      = 20


@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(42)
    X   = rng.standard_normal((N_TRAIN + N_TEST, WINDOW_SIZE)).astype(np.float32)
    y   = rng.integers(0, N_CLASSES, N_TRAIN + N_TEST)
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


# ── Random Forest ─────────────────────────────────────────────────────────────

class TestRandomForest:

    def test_fit_predict_shape(self, tiny_data):
        X_tr, y_tr, X_te, _ = tiny_data
        m = RandomForestModel(CFG["models"], sampling_rate=12000, seed=42)
        m.clf.set_params(n_estimators=10)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        assert preds.shape == (N_TEST,)

    def test_proba_sums_to_one(self, tiny_data):
        X_tr, y_tr, X_te, _ = tiny_data
        m = RandomForestModel(CFG["models"], sampling_rate=12000, seed=42)
        m.clf.set_params(n_estimators=10)
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(N_TEST), atol=1e-5)

    def test_save_load(self, tiny_data, tmp_path):
        X_tr, y_tr, X_te, _ = tiny_data
        m = RandomForestModel(CFG["models"], sampling_rate=12000, seed=42)
        m.clf.set_params(n_estimators=10)
        m.fit(X_tr, y_tr)
        p1 = m.predict(X_te)

        path = tmp_path / "rf.pkl"
        m.save(path)

        m2 = RandomForestModel(CFG["models"], sampling_rate=12000, seed=42)
        m2.load(path)
        p2 = m2.predict(X_te)
        np.testing.assert_array_equal(p1, p2)

    def test_not_fitted_raises(self):
        m = RandomForestModel(CFG["models"], sampling_rate=12000, seed=42)
        with pytest.raises(Exception):
            m.predict(np.zeros((2, WINDOW_SIZE), dtype=np.float32))


# ── Anomaly Detector ──────────────────────────────────────────────────────────

class TestAnomalyDetector:

    def test_score_shape(self, tiny_data):
        X_tr, y_tr, X_te, _ = tiny_data
        X_normal = X_tr[y_tr == 0]
        if len(X_normal) == 0:
            X_normal = X_tr[:10]
        det = AnomalyDetector(CFG["models"], sampling_rate=12000)
        det.fit(X_normal)
        scores = det.score(X_te)
        assert scores.shape == (N_TEST,)

    def test_scores_are_negative(self, tiny_data):
        X_tr, y_tr, X_te, _ = tiny_data
        det = AnomalyDetector(CFG["models"], sampling_rate=12000)
        det.fit(X_tr[:10])
        scores = det.score(X_te)
        assert np.all(scores <= 0), "IF scores should be in (−1, 0]"

    def test_save_load(self, tiny_data, tmp_path):
        X_tr, _, X_te, _ = tiny_data
        det = AnomalyDetector(CFG["models"], sampling_rate=12000)
        det.fit(X_tr[:10])
        s1 = det.score(X_te)
        path = tmp_path / "anm.pkl"
        det.save(path)
        det2 = AnomalyDetector(CFG["models"], sampling_rate=12000)
        det2.load(path)
        s2 = det2.score(X_te)
        np.testing.assert_allclose(s1, s2, atol=1e-6)


# ── CNN1D (PyTorch required) ──────────────────────────────────────────────────

pytest.importorskip("torch", reason="PyTorch not installed")

from src.models.cnn1d import CNN1DModel  # noqa: E402


class TestCNN1D:

    @pytest.fixture
    def tiny_cfg(self):
        return {
            **CFG["models"],
            **CFG["training"],
            "epochs": 2,
            "batch_size": 16,
        }

    def test_fit_predict(self, tiny_data, tiny_cfg):
        X_tr, y_tr, X_te, _ = tiny_data
        m = CNN1DModel(tiny_cfg, n_classes=N_CLASSES, seed=42)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        assert preds.shape == (N_TEST,)
        assert set(preds).issubset(set(range(N_CLASSES)))

    def test_proba_sums(self, tiny_data, tiny_cfg):
        X_tr, y_tr, X_te, _ = tiny_data
        m = CNN1DModel(tiny_cfg, n_classes=N_CLASSES, seed=42)
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(N_TEST), atol=1e-5)

    def test_save_load(self, tiny_data, tiny_cfg, tmp_path):
        X_tr, y_tr, X_te, _ = tiny_data
        m = CNN1DModel(tiny_cfg, n_classes=N_CLASSES, seed=42)
        m.fit(X_tr, y_tr)
        p1 = m.predict(X_te)

        path = tmp_path / "cnn.pt"
        m.save(path)
        m2 = CNN1DModel(tiny_cfg, n_classes=N_CLASSES, seed=42)
        m2.load(path)
        p2 = m2.predict(X_te)
        np.testing.assert_array_equal(p1, p2)
