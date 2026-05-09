# -*- coding: utf-8 -*-
"""Problem catalogue for dual-bench.

Exports all concrete problem classes and the ``ALL_PROBLEMS`` list.
"""

from dual_bench.problems.base import CMOProblem, DualProblem
from dual_bench.problems.constrained.cantilever import CantileverBeam
from dual_bench.problems.constrained.pressure_vessel import PressureVessel
from dual_bench.problems.constrained.speed_reducer import SpeedReducer
from dual_bench.problems.constrained.spring import Spring
from dual_bench.problems.constrained.welded_beam import WeldedBeam

ALL_PROBLEMS = [Spring, PressureVessel, WeldedBeam, SpeedReducer, CantileverBeam]

__all__ = [
    "DualProblem",
    "CMOProblem",
    "Spring",
    "PressureVessel",
    "WeldedBeam",
    "SpeedReducer",
    "CantileverBeam",
    "ALL_PROBLEMS",
]
