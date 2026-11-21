"""orderguard: encoder-locked bearing diagnostics and supervisory control."""

from __future__ import annotations

from .kinematics import ER16K, BearingGeometry, FaultOrders, sideband_orders
from .supervisor import ActuatorCommand, Mode, SupervisoryController

__version__ = "0.1.0"

__all__ = [
    "ER16K",
    "ActuatorCommand",
    "BearingGeometry",
    "FaultOrders",
    "Mode",
    "SupervisoryController",
    "__version__",
    "sideband_orders",
]
