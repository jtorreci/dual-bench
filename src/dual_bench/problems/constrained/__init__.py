# -*- coding: utf-8 -*-
"""Single-objective constrained benchmark problems in dual d/c form."""

from dual_bench.problems.constrained.cantilever import CantileverBeam
from dual_bench.problems.constrained.pressure_vessel import PressureVessel
from dual_bench.problems.constrained.speed_reducer import SpeedReducer
from dual_bench.problems.constrained.spring import Spring
from dual_bench.problems.constrained.welded_beam import WeldedBeam

__all__ = [
    "Spring",
    "PressureVessel",
    "WeldedBeam",
    "SpeedReducer",
    "CantileverBeam",
]
