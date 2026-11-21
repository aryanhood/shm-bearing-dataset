"""Verify zero trial/profile leakage across grouped evaluation splits."""

from __future__ import annotations

import numpy as np

from orderguard.benchmark import PreparedDataset, fold_splits
from orderguard.data import CLASSES, SPEED_PROFILES, TRIALS
from orderguard.order_tracking import CARRIER_BANDS_HZ, N_ORDER_BINS


def _mock_dataset() -> PreparedDataset:
    stems: list[str] = []
    labels: list[int] = []
    trials: list[int] = []
    profiles: list[str] = []
    for c_idx, c_code in enumerate(("H", "I", "O", "B", "C")):
        for p_code in SPEED_PROFILES:
            for t in TRIALS:
                stems.append(f"{c_code}-{p_code}-{t}")
                # 4 windows per recording
                labels.extend([c_idx] * 4)
                trials.extend([t] * 4)
                profiles.extend([p_code] * 4)

    n = len(labels)
    assert len(CLASSES) == 5
    return PreparedDataset(
        stems=tuple(stems),
        time_x=np.zeros((n, 96), dtype=np.float64),
        order_x=np.zeros((n, 33 * len(CARRIER_BANDS_HZ)), dtype=np.float64),
        cnn_x=np.zeros((n, len(CARRIER_BANDS_HZ), N_ORDER_BINS), dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        trial=np.asarray(trials, dtype=np.int64),
        profile_code=np.asarray(profiles),
        rpm=np.full(n, 1200.0, dtype=np.float64),
    )


def test_grouped_splits_have_zero_leakage() -> None:
    ds = _mock_dataset()

    for _, tr, te in fold_splits(ds, "unseen_trial"):
        assert not np.any(tr & te)
        assert set(ds.trial[tr]).isdisjoint(set(ds.trial[te]))
        assert set(ds.y[te]) == set(range(len(CLASSES)))

    for _, tr, te in fold_splits(ds, "unseen_speed_profile"):
        assert not np.any(tr & te)
        assert set(ds.profile_code[tr]).isdisjoint(set(ds.profile_code[te]))
        assert set(ds.y[te]) == set(range(len(CLASSES)))
