# -*- coding: utf-8 -*-
"""Suite registry for dual-bench.

Provides :func:`suite` to retrieve a list of problem instances by name,
and :func:`list_suites` to enumerate available suites.

Built-in suites
----------------
``engineering5``
    5 single-objective engineering benchmarks (Spring, PressureVessel,
    WeldedBeam, SpeedReducer, CantileverBeam).
``classic5``
    5 classic constrained multi-objective problems (BNH, SRN, TNK, OSY, CONSTR).
``lircmop``
    14 LIRCMOP problems with large infeasible regions.
``mw``
    MW1-MW14 via pymoo (requires ``dual-bench[pymoo]``).
``dascmop``
    DAS-CMOP1-9 via pymoo (requires ``dual-bench[pymoo]``).
``zdt``
    ZDT1-6 unconstrained MO problems via pymoo.
``dtlz``
    DTLZ1-7 unconstrained MO problems via pymoo.
``wfg``
    WFG1-9 unconstrained MO problems via pymoo.
"""

from __future__ import annotations

from typing import List


def _engineering5():
    from dual_bench.problems.constrained import (
        CantileverBeam,
        PressureVessel,
        SpeedReducer,
        Spring,
        WeldedBeam,
    )
    return [Spring(), PressureVessel(), WeldedBeam(), SpeedReducer(), CantileverBeam()]


def _classic5():
    from dual_bench.problems.cmop.classic import BNH, CONSTR, OSY, SRN, TNK
    return [BNH(), SRN(), TNK(), OSY(), CONSTR()]


def _lircmop():
    from dual_bench.problems.cmop import lircmop
    return [getattr(lircmop, f"LIRCMOP{i}")() for i in range(1, 15)]


def _mw():
    from dual_bench.problems.cmop.pymoo_bridge import mw
    return [mw(i) for i in range(1, 15)]


def _dascmop():
    from dual_bench.problems.cmop.pymoo_bridge import dascmop
    problems = []
    for i in range(1, 10):
        if i <= 6:
            problems.append(dascmop(i, difficulty=1))
        else:
            # DASCMOP7-9 require difficulty_factors tuple
            problems.append(dascmop(i, difficulty=(0.25, 0.0, 0.0)))
    return problems


def _zdt():
    from dual_bench.problems.cmop.pymoo_bridge import from_pymoo
    return [from_pymoo(f"zdt{i}") for i in range(1, 7)]


def _dtlz():
    from dual_bench.problems.cmop.pymoo_bridge import from_pymoo
    return [from_pymoo(f"dtlz{i}") for i in range(1, 8)]


def _wfg():
    from dual_bench.problems.cmop.pymoo_bridge import from_pymoo
    return [from_pymoo(f"wfg{i}", n_var=10, n_obj=2) for i in range(1, 10)]


# Registry mapping suite name -> (factory, requires_pymoo)
_SUITES = {
    "engineering5": (_engineering5, False),
    "classic5": (_classic5, False),
    "lircmop": (_lircmop, False),
    "mw": (_mw, True),
    "dascmop": (_dascmop, True),
    "zdt": (_zdt, True),
    "dtlz": (_dtlz, True),
    "wfg": (_wfg, True),
}


def suite(name: str) -> List:
    """Return a list of problem instances for the named suite.

    Parameters
    ----------
    name : str
        One of the registered suite names (see :func:`list_suites`).

    Returns
    -------
    list
        Problem instances (:class:`DualProblem` or :class:`CMOProblem`).

    Raises
    ------
    ValueError
        If the suite name is unknown.
    ImportError
        If the suite requires pymoo and it is not installed.
    """
    if name not in _SUITES:
        available = ", ".join(sorted(_SUITES))
        raise ValueError(f"Unknown suite {name!r}. Available: {available}")
    factory, needs_pymoo = _SUITES[name]
    if needs_pymoo:
        try:
            import pymoo  # noqa: F401
        except ImportError:
            raise ImportError(
                f"Suite {name!r} requires pymoo. "
                f"Install with: pip install dual-bench[pymoo]"
            )
    return factory()


def list_suites() -> List[str]:
    """Return the list of available suite names.

    Returns
    -------
    list of str
    """
    return sorted(_SUITES.keys())
