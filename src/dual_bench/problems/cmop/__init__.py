# -*- coding: utf-8 -*-
"""Constrained multi-objective problem library.

Includes classic CMOPs (BNH, SRN, TNK, OSY, CONSTR), the LIRCMOP1-14 suite,
and a pymoo bridge for MW and DAS-CMOP problems.
"""

from dual_bench.problems.cmop.classic import BNH, CONSTR, OSY, SRN, TNK
from dual_bench.problems.cmop.lircmop import (
    LIRCMOP1,
    LIRCMOP2,
    LIRCMOP3,
    LIRCMOP4,
    LIRCMOP5,
    LIRCMOP6,
    LIRCMOP7,
    LIRCMOP8,
    LIRCMOP9,
    LIRCMOP10,
    LIRCMOP11,
    LIRCMOP12,
    LIRCMOP13,
    LIRCMOP14,
)
from dual_bench.problems.cmop.pymoo_bridge import PymooWrapper, dascmop, from_pymoo, mw

# Unified registry: name -> CMOProblem instance
CMOP_REGISTRY: dict = {}


def _register_all():
    for cls in [BNH, SRN, TNK, OSY, CONSTR]:
        CMOP_REGISTRY[cls.name] = cls()
    for i in range(1, 15):
        cls = globals()[f"LIRCMOP{i}"]
        CMOP_REGISTRY[f"LIRCMOP{i}"] = cls()


_register_all()

__all__ = [
    "BNH", "SRN", "TNK", "OSY", "CONSTR",
    "LIRCMOP1", "LIRCMOP2", "LIRCMOP3", "LIRCMOP4",
    "LIRCMOP5", "LIRCMOP6", "LIRCMOP7", "LIRCMOP8",
    "LIRCMOP9", "LIRCMOP10", "LIRCMOP11", "LIRCMOP12",
    "LIRCMOP13", "LIRCMOP14",
    "PymooWrapper", "from_pymoo", "mw", "dascmop",
    "CMOP_REGISTRY",
]
