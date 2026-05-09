# -*- coding: utf-8 -*-
"""Bridge to import problems from pymoo into the CMOProblem interface.

Provides :class:`PymooWrapper` to wrap any pymoo ``Problem`` as a
:class:`CMOProblem`, plus convenience factories for common suites:

- :func:`from_pymoo` -- generic loader by name.
- :func:`mw` -- MW test suite (MW1-MW14).
- :func:`dascmop` -- DAS-CMOP suite (DAS-CMOP1-9).

Requires ``pip install dual-bench[pymoo]``.
"""

import numpy as np

from dual_bench.problems.base import CMOProblem


class PymooWrapper(CMOProblem):
    """Wrap any pymoo Problem as a :class:`CMOProblem`.

    Parameters
    ----------
    pymoo_prob : pymoo.core.problem.Problem
        An instantiated pymoo problem.
    name : str, optional
        Display name. Defaults to the class name of the pymoo problem.
    """

    def __init__(self, pymoo_prob, name=None):
        self._p = pymoo_prob
        self.name = name or pymoo_prob.__class__.__name__
        self.n_var = pymoo_prob.n_var
        self.n_obj = pymoo_prob.n_obj
        self.n_constr = pymoo_prob.n_ieq_constr
        self.xl = np.array(pymoo_prob.xl, dtype=float)
        self.xu = np.array(pymoo_prob.xu, dtype=float)

    def evaluate(self, x):
        out = {}
        self._p._evaluate(np.atleast_2d(x), out)
        F = out["F"][0]
        G = out["G"][0] if "G" in out and out["G"] is not None else np.array([])
        return np.asarray(F, dtype=float), np.asarray(G, dtype=float)

    def pareto_front(self, n_points=100):
        try:
            return self._p.pareto_front(n_points=n_points)
        except Exception:
            return None


def from_pymoo(name, **kwargs):
    """Create a :class:`CMOProblem` from a pymoo problem by name.

    Examples::

        from_pymoo('mw1')
        from_pymoo('bnh')
        from_pymoo('zdt1')
        from_pymoo('dascmop1', difficulty=2)

    Parameters
    ----------
    name : str
        Problem name as recognized by ``pymoo.problems.get_problem``.
    **kwargs
        Additional keyword arguments passed to the pymoo constructor.

    Returns
    -------
    PymooWrapper
    """
    from pymoo.problems import get_problem
    p = get_problem(name, **kwargs)
    return PymooWrapper(p, name=name.upper())


def mw(number):
    """Return MW{number} problem (1-14) via pymoo.

    Parameters
    ----------
    number : int
        Problem index (1-14).
    """
    return from_pymoo(f"mw{number}")


def dascmop(number, difficulty):
    """Return DAS-CMOP{number} (1-9) with specified difficulty.

    Parameters
    ----------
    number : int
        Problem index (1-9).
    difficulty : int or tuple
        For DASCMOP1-6: integer (1-16).
        For DASCMOP7-9: tuple (eta, zeta, gamma).
    """
    if isinstance(difficulty, (list, tuple)):
        return from_pymoo(f"dascmop{number}",
                          difficulty_factors=tuple(difficulty))
    return from_pymoo(f"dascmop{number}", difficulty=difficulty)
