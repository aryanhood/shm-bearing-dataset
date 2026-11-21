"""First-principles unit tests for rolling-element bearing kinematics."""

from __future__ import annotations

import math

import pytest

from orderguard.kinematics import ER16K, BearingGeometry, sideband_orders


def test_er16k_orders_match_uottawa_spec() -> None:
    orders = ER16K.orders()
    assert orders.ftf == pytest.approx(0.397, abs=1e-3)
    assert orders.bpfo == pytest.approx(3.572, abs=1e-3)
    assert orders.bpfi == pytest.approx(5.428, abs=1e-3)
    assert orders.bsf == pytest.approx(2.323, abs=1e-3)
    assert orders.bsf_2x == pytest.approx(2.0 * orders.bsf, abs=1e-9)
    # Conservation identity: BPFO + BPFI == N_balls
    assert orders.bpfo + orders.bpfi == pytest.approx(float(ER16K.n_balls), abs=1e-9)


def test_contact_angle_reduces_ratio_effect() -> None:
    zero = BearingGeometry("test", n_balls=8, ball_diameter_mm=8.0, pitch_diameter_mm=40.0)
    angled = BearingGeometry(
        "test_15",
        n_balls=8,
        ball_diameter_mm=8.0,
        pitch_diameter_mm=40.0,
        contact_angle_deg=15.0,
    )
    assert angled.orders().bpfo > zero.orders().bpfo
    assert angled.orders().bpfi < zero.orders().bpfi
    expected_ratio = (8.0 / 40.0) * math.cos(math.radians(15.0))
    assert angled.orders().ftf == pytest.approx(0.5 * (1.0 - expected_ratio), abs=1e-9)


def test_sidebands_match_load_zone_modulation() -> None:
    orders = ER16K.orders()
    sb = sideband_orders(orders)
    assert sb["bpfi_sidebands"] == pytest.approx((orders.bpfi - 1.0, orders.bpfi + 1.0))
    assert sb["bsf_2x_sidebands"] == pytest.approx(
        (orders.bsf_2x - orders.ftf, orders.bsf_2x + orders.ftf)
    )
