"""Rolling-element bearing kinematic orders from shaft and bearing geometry.

All defect frequencies are returned in dimensionless *orders* (cycles per shaft
revolution), so they stay invariant when shaft speed changes over time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BearingGeometry:
    """Physical geometry of a rolling-element bearing."""

    name: str
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


@dataclass(frozen=True)
class FaultOrders:
    """Characteristic bearing fault orders (cycles per shaft revolution)."""

    ftf: float
    bpfo: float
    bpfi: float
    bsf: float
    bsf_2x: float


# Rexnord/MB ER16K test bearing used in the uOttawa time-varying speed dataset
# (Huang & Baddour, Data in Brief 2018): 9 balls, d = 7.94 mm, D = 38.52 mm.
ER16K = BearingGeometry(
    name="ER16K",
    n_balls=9,
    ball_diameter_mm=7.94,
    pitch_diameter_mm=38.52,
    contact_angle_deg=0.0,
)


def sideband_orders(orders: FaultOrders) -> dict[str, tuple[float, ...]]:
    """Return amplitude-modulation sideband orders for inner-race and ball defects.

    An inner-race defect rotates through the fixed radial load zone once per shaft
    revolution, producing sidebands at BPFI +/- 1.0x. A ball defect passes through
    the load zone at cage speed, modulating 2*BSF by +/- FTF.
    """
    return {
        "bpfi_sidebands": (orders.bpfi - 1.0, orders.bpfi + 1.0),
        "bsf_2x_sidebands": (orders.bsf_2x - orders.ftf, orders.bsf_2x + orders.ftf),
    }
