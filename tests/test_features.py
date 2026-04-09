"""
Tests — Feature Extractor
"""
import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.extractor import FeatureExtractor


@pytest.fixture
def extractor():
    return FeatureExtractor(sampling_rate=12000, n_fft_bins=64)


@pytest.fixture
def batch():
    rng = np.random.default_rng(0)
    return rng.standard_normal((8, 1024)).astype(np.float32)


def test_output_shape(extractor, batch):
    F = extractor.transform(batch)
    assert F.ndim == 2
    assert F.shape[0] == 8
    assert F.shape[1] > 0


def test_output_dtype(extractor, batch):
    F = extractor.transform(batch)
    assert F.dtype == np.float32


def test_feature_names_length(extractor, batch):
    F = extractor.transform(batch)
    names = extractor.feature_names
    assert len(names) == F.shape[1], f"names={len(names)} != features={F.shape[1]}"


def test_deterministic(extractor, batch):
    F1 = extractor.transform(batch)
    F2 = extractor.transform(batch)
    np.testing.assert_array_equal(F1, F2)


def test_no_fft(batch):
    ex = FeatureExtractor(sampling_rate=12000, n_fft_bins=64, use_fft=False)
    F  = ex.transform(batch)
    # Only 8 time-domain features per sample
    assert F.shape[1] == 8


def test_freq_only(batch):
    ex = FeatureExtractor(
        sampling_rate=12000,
        n_fft_bins=64,
        include_time=False,
        include_freq=True,
    )
    F = ex.transform(batch)
    assert F.shape[1] == 67


def test_single_sample(extractor):
    sig = np.sin(2 * np.pi * 50 * np.linspace(0, 0.1, 1024)).astype(np.float32)
    F   = extractor.transform(sig[None, :])
    assert F.shape == (1, extractor.n_features(1024))


def test_rms_feature(extractor):
    """RMS of a unit-amplitude sine should be ≈ 1/√2."""
    sig = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 12000)).astype(np.float32)
    F   = extractor.transform(sig[None, :])
    rms_idx = extractor.feature_names.index("rms")
    assert abs(F[0, rms_idx] - 1 / np.sqrt(2)) < 0.05
