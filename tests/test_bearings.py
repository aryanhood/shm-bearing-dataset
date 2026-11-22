"""Unit tests for built-in bearing catalog and kinematic frequency formulas."""

from __future__ import annotations

import pytest

from vibcheck.bearings import BEARINGS, get_bearing


def test_builtin_bearings_kinematics() -> None:
    b6205 = get_bearing("6205")
    orders = b6205.orders()
    assert orders.bpfo == pytest.approx(3.585, abs=1e-2)
    assert orders.bpfi == pytest.approx(5.415, abs=1e-2)
    assert orders.bpfo + orders.bpfi == pytest.approx(float(b6205.n_balls), abs=1e-9)

    freqs = b6205.frequencies_hz(1750.0)
    assert freqs["shaft_1x"] == pytest.approx(1750.0 / 60.0, abs=1e-6)
    assert freqs["bpfi"] == pytest.approx(orders.bpfi * 1750.0 / 60.0, abs=1e-4)

    assert get_bearing("er16k").code == "ER16K"
    assert "608" in BEARINGS
    with pytest.raises(KeyError):
        get_bearing("unknown-999")
