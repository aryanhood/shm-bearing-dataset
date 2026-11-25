"""Rolling-element bearing kinematic orders from shaft and bearing geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FaultOrders:
    """Characteristic bearing defect orders (cycles per shaft revolution)."""

    ftf: float
    bpfo: float
    bpfi: float
    bsf: float
    bsf_2x: float


@dataclass(frozen=True)
class BearingSpec:
    """Physical geometry of a rolling-element ball bearing."""

    code: str
    description: str
    n_balls: int
    ball_diameter_mm: float
    pitch_diameter_mm: float
    contact_angle_deg: float = 0.0

    def orders(self) -> FaultOrders:
        """Compute characteristic defect orders relative to shaft rotation (1.0x)."""
        ratio = (self.ball_diameter_mm / self.pitch_diameter_mm) * math.cos(
            math.radians(self.contact_angle_deg)
        )
        ftf = 0.5 * (1.0 - ratio)
        bpfo = 0.5 * self.n_balls * (1.0 - ratio)
        bpfi = 0.5 * self.n_balls * (1.0 + ratio)
        bsf = (self.pitch_diameter_mm / (2.0 * self.ball_diameter_mm)) * (1.0 - ratio**2)
        return FaultOrders(
            ftf=ftf,
            bpfo=bpfo,
            bpfi=bpfi,
            bsf=bsf,
            bsf_2x=2.0 * bsf,
        )

    def frequencies_hz(self, rpm: float) -> dict[str, float]:
        """Return fault characteristic frequencies in Hz for a given shaft RPM."""
        shaft_hz = rpm / 60.0
        o = self.orders()
        return {
            "shaft_1x": shaft_hz,
            "ftf": o.ftf * shaft_hz,
            "bpfo": o.bpfo * shaft_hz,
            "bpfi": o.bpfi * shaft_hz,
            "bsf": o.bsf * shaft_hz,
            "bsf_2x": o.bsf_2x * shaft_hz,
        }


SKF_6205 = BearingSpec(
    code="6205",
    description="CWRU drive-end bearing (SKF 6205-2RS JEM)",
    n_balls=9,
    ball_diameter_mm=7.940,
    pitch_diameter_mm=39.04,
)

ER16K = BearingSpec(
    code="ER16K",
    description="uOttawa / SpectraQuest insert bearing (ER16K)",
    n_balls=9,
    ball_diameter_mm=7.940,
    pitch_diameter_mm=38.52,
)

BEARINGS: dict[str, BearingSpec] = {
    "608": BearingSpec(
        code="608",
        description="Small fans and BLDC rotors",
        n_balls=7,
        ball_diameter_mm=3.969,
        pitch_diameter_mm=15.0,
    ),
    "6203": BearingSpec(
        code="6203",
        description="Small motors, CWRU fan-end",
        n_balls=8,
        ball_diameter_mm=6.747,
        pitch_diameter_mm=28.5,
    ),
    "6204": BearingSpec(
        code="6204",
        description="Pumps and light industrial spindles",
        n_balls=8,
        ball_diameter_mm=7.938,
        pitch_diameter_mm=34.5,
    ),
    "6205": SKF_6205,
    "6206": BearingSpec(
        code="6206",
        description="Industrial fans and CNC spindles",
        n_balls=9,
        ball_diameter_mm=9.525,
        pitch_diameter_mm=46.0,
    ),
    "6305": BearingSpec(
        code="6305",
        description="Heavy-duty motors and compressors",
        n_balls=7,
        ball_diameter_mm=11.112,
        pitch_diameter_mm=43.5,
    ),
    "ER16K": ER16K,
}


def get_bearing(code: str) -> BearingSpec:
    """Look up a bearing preset by code (case-insensitive)."""
    key = code.strip().upper()
    if key in BEARINGS:
        return BEARINGS[key]
    raise KeyError(f"Unknown bearing '{code}'. Built-in options: {', '.join(BEARINGS)}")


def sideband_orders(orders: FaultOrders) -> dict[str, tuple[float, ...]]:
    """Amplitude-modulation sideband orders for inner-race and ball defects."""
    return {
        "bpfi_sidebands": (orders.bpfi - 1.0, orders.bpfi + 1.0),
        "bsf_2x_sidebands": (orders.bsf_2x - orders.ftf, orders.bsf_2x + orders.ftf),
    }
