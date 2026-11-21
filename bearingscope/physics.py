"""Bearing kinematics for the CWRU drive-end bearing (SKF 6205-2RS JEM).

A localized defect produces a train of impacts at a characteristic rate. Each
rate is a fixed multiple of the shaft speed, set by the bearing geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

SAMPLE_RATE_HZ = 12_000
WINDOW = 2048  # samples per analysis window (~171 ms at 12 kHz)

CLASSES = ("normal", "inner", "outer", "ball")
CLASS_TITLES = ("Normal", "Inner race fault", "Outer race fault", "Ball fault")

# Nominal motor speed for each load setting of the CWRU test rig.
RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
DEFAULT_RPM = 1772

# Defect frequencies as multiples of shaft frequency (CWRU bearing data center).
BPFI_ORDER = 5.4152  # ball pass frequency, inner race
BPFO_ORDER = 3.5848  # ball pass frequency, outer race
BSF_ORDER = 2.3570  # ball spin frequency


@dataclass(frozen=True)
class FaultFrequencies:
    shaft: float
    bpfi: float
    bpfo: float
    bsf: float


def fault_frequencies(rpm: float) -> FaultFrequencies:
    shaft = rpm / 60.0
    return FaultFrequencies(
        shaft=shaft,
        bpfi=BPFI_ORDER * shaft,
        bpfo=BPFO_ORDER * shaft,
        bsf=BSF_ORDER * shaft,
    )
