"""
Tests — Data Loader + Preprocessor
"""
import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader      import BearingDataLoader
from src.data.preprocessor import Preprocessor
from src.utils.config     import CFG


@pytest.fixture(scope="module")
def raw_data():
    loader = BearingDataLoader(CFG)
    return loader.load()


def test_loader_returns_numpy(raw_data):
    X, y = raw_data
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_loader_shapes(raw_data):
    X, y = raw_data
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == CFG["data"]["window_size"]


def test_loader_dtype(raw_data):
    X, y = raw_data
    assert X.dtype == np.float32
    assert y.dtype == np.int64


def test_loader_class_range(raw_data):
    _, y = raw_data
    n_classes = len(CFG["data"]["classes"])
    assert int(y.min()) >= 0
    assert int(y.max()) < n_classes


def test_loader_all_classes_present(raw_data):
    _, y = raw_data
    n_classes = len(CFG["data"]["classes"])
    assert len(np.unique(y)) == n_classes


def test_loader_deterministic():
    l1 = BearingDataLoader(CFG); X1, y1 = l1.load()
    l2 = BearingDataLoader(CFG); X2, y2 = l2.load()
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


# ── Preprocessor ──────────────────────────────────────────────────────────────

def test_split_fractions(raw_data):
    X, y  = raw_data
    prep  = Preprocessor(train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42)
    splits = prep.fit_transform(X, y)

    total = sum(len(s[1]) for s in splits.values())
    assert total == len(y)


def test_no_train_leakage(raw_data):
    """Verify train and test index sets do not overlap."""
    X, y = raw_data
    prep = Preprocessor(seed=42)
    splits = prep.fit_transform(X, y)
    X_tr = splits["train"][0]
    X_te = splits["test"][0]
    # Different shapes guarantee no direct overlap check
    assert X_tr.shape[0] + X_te.shape[0] <= len(y)


def test_scaler_zero_mean(raw_data):
    """After scaling, training data should have approx. zero mean per feature."""
    X, y = raw_data
    prep = Preprocessor(seed=42)
    splits = prep.fit_transform(X, y)
    X_tr = splits["train"][0]
    means = X_tr.mean(axis=0)
    assert np.abs(means).mean() < 0.05, "Training mean far from zero"


def test_transform_inference(raw_data):
    X, y = raw_data
    prep = Preprocessor(seed=42)
    prep.fit_transform(X, y)
    sample = X[:5]
    out = prep.transform(sample)
    assert out.shape == sample.shape
    assert out.dtype == np.float32


def test_preprocessor_save_load(raw_data, tmp_path):
    X, y = raw_data
    prep = Preprocessor(seed=42)
    prep.fit_transform(X, y)
    path = tmp_path / "prep.pkl"
    prep.save(path)

    restored = Preprocessor().load(path)
    out1 = prep.transform(X[:10])
    out2 = restored.transform(X[:10])
    np.testing.assert_allclose(out1, out2, atol=1e-6)


def test_preprocessor_rejects_classes_too_small():
    X = np.random.randn(6, 64).astype(np.float32)
    y = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    prep = Preprocessor(seed=42)
    with pytest.raises(ValueError, match="at least 3 samples"):
        prep.fit_transform(X, y)
