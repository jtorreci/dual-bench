# -*- coding: utf-8 -*-
"""Adapter to run dual-bench formulations with pymoo.

Provides :class:`ToPymoo` (problem adapter) and :class:`EpsilonCallback`
(constraint annealing hook).

Requires ``pip install dual-bench[pymoo]``.

Example::

    from dual_bench import Spring
    from dual_bench.engines.pymoo_engine import ToPymoo, EpsilonCallback
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    form = Spring().to_2ob_rest()
    result = minimize(
        ToPymoo(form), NSGA2(pop_size=100),
        ('n_eval', 30000),
        callback=EpsilonCallback(form, 30000),
    )
"""

import numpy as np
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem as PymooProblem


class ToPymoo(PymooProblem):
    """Convert a dual-bench formulation to a pymoo Problem.

    Parameters
    ----------
    formulation
        Any dual-bench formulation object with ``evaluate``,
        ``var_bounds``, ``n_objectives``, and ``n_constraints``.
    """

    def __init__(self, formulation):
        self._form = formulation
        xl = np.array([lb for lb, ub in formulation.var_bounds])
        xu = np.array([ub for lb, ub in formulation.var_bounds])
        super().__init__(
            n_var=len(formulation.var_bounds),
            n_obj=formulation.n_objectives,
            n_ieq_constr=formulation.n_constraints,
            xl=xl, xu=xu,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F, G = [], []
        for x in X:
            f, g = self._form.evaluate(x)
            F.append(f)
            G.append(g)
        out["F"] = np.array(F)
        if self._form.n_constraints > 0:
            out["G"] = np.array(G)


class EpsilonCallback(Callback):
    """pymoo callback that updates epsilon-annealing each generation.

    Parameters
    ----------
    formulation
        A formulation with ``set_progress(t, T)`` method.
    nfe_max : int
        Maximum number of function evaluations for the run.
    """

    def __init__(self, formulation, nfe_max):
        super().__init__()
        self._form = formulation
        self._nfe_max = nfe_max

    def notify(self, algorithm):
        nfe = algorithm.evaluator.n_eval
        self._form.set_progress(nfe, self._nfe_max)
