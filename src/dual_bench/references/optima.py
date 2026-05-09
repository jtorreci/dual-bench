# -*- coding: utf-8 -*-
"""Known optima and metadata for built-in benchmark problems.

Each entry records the best-known objective value, its source, and the
decision vector (when available) for reproducibility checks.

Sources
-------
- Spring, Pressure Vessel, Welded Beam, Speed Reducer: Coello Coello (2000),
  Mezura-Montes & Coello Coello (2011), updated with Gandomi et al. (2013).
- Cantilever Beam: Chickermane & Goh (1996).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ProblemInfo:
    """Metadata for a single benchmark problem."""

    name: str
    n_vars: int
    n_ratios: int
    known_optimum: float
    known_x: Optional[List[float]] = None
    source: str = ""


# ======================================================================
# Registry
# ======================================================================

_OPTIMA: Dict[str, ProblemInfo] = {
    "Spring": ProblemInfo(
        name="Spring",
        n_vars=3,
        n_ratios=4,
        known_optimum=0.012665,
        known_x=[0.051690, 0.356750, 11.287126],
        source="Coello Coello (2000)",
    ),
    "PressureVessel": ProblemInfo(
        name="PressureVessel",
        n_vars=4,
        n_ratios=4,
        known_optimum=5885.33,
        known_x=[0.7782, 0.3846, 40.3196, 200.0],
        source="Kannan & Kramer (1994), continuous formulation",
    ),
    "WeldedBeam": ProblemInfo(
        name="WeldedBeam",
        n_vars=4,
        n_ratios=4,
        known_optimum=1.7249,
        known_x=[0.2057, 3.4705, 9.0366, 0.2057],
        source="Ragsdell & Phillips (1976), Deb (2000)",
    ),
    "SpeedReducer": ProblemInfo(
        name="SpeedReducer",
        n_vars=7,
        n_ratios=11,
        known_optimum=2994.47,
        known_x=[3.5, 0.7, 17.0, 7.3, 7.8, 3.3502, 5.2867],
        source="Golinski (1970), Mezura-Montes & Coello Coello (2011)",
    ),
    "CantileverBeam": ProblemInfo(
        name="CantileverBeam",
        n_vars=5,
        n_ratios=1,
        known_optimum=1.3400,
        known_x=[6.0168, 5.3132, 4.4958, 3.4923, 2.1483],
        source="Chickermane & Goh (1996)",
    ),
}


def optimum(problem: str) -> float:
    """Return the known optimal objective value for *problem*.

    Parameters
    ----------
    problem : str
        Problem name (case-sensitive): ``'Spring'``, ``'PressureVessel'``,
        ``'WeldedBeam'``, ``'SpeedReducer'``, or ``'CantileverBeam'``.

    Raises
    ------
    KeyError
        If the problem name is not found.
    """
    return _OPTIMA[problem].known_optimum


def info(problem: str) -> ProblemInfo:
    """Return full metadata for *problem*."""
    return _OPTIMA[problem]


def list_problems() -> List[str]:
    """Return the names of all problems with known optima."""
    return sorted(_OPTIMA.keys())
