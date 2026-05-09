# -*- coding: utf-8 -*-
"""Adapter to run dual-bench formulations with Platypus.

Provides :class:`ToPlatypus` (problem adapter) and :class:`EpsilonUpdater`
(constraint annealing hook).

Requires ``pip install dual-bench[platypus]``.

Example::

    from dual_bench.problems.cmop import BNH
    from dual_bench.formulations import CMOPEnvelope
    from dual_bench.engines.platypus_engine import ToPlatypus, EpsilonUpdater
    from platypus import NSGAII

    form = CMOPEnvelope(BNH())
    algo = NSGAII(ToPlatypus(form), population_size=100)
    updater = EpsilonUpdater(form, 30000)
    for gen in range(300):
        algo.step()
        updater.update(algo)
"""

import numpy as np
from platypus import Problem as PlatypusProblem
from platypus import Real


class ToPlatypus(PlatypusProblem):
    """Convert a dual-bench formulation to a Platypus Problem.

    Parameters
    ----------
    formulation
        Any dual-bench formulation object with ``evaluate``,
        ``var_bounds``, ``n_objectives``, and ``n_constraints``.
    """

    def __init__(self, formulation):
        n_var = len(formulation.var_bounds)
        super().__init__(n_var, formulation.n_objectives,
                         formulation.n_constraints)
        for i, (lb, ub) in enumerate(formulation.var_bounds):
            self.types[i] = Real(lb, ub)
        for i in range(formulation.n_constraints):
            self.constraints[i] = "<=0"
        self._form = formulation

    def evaluate(self, solution):
        x = np.array([solution.variables[i]
                       for i in range(len(self._form.var_bounds))])
        f, g = self._form.evaluate(x)
        solution.objectives[:] = list(f)
        if self._form.n_constraints > 0:
            solution.constraints[:] = list(g)


class EpsilonUpdater:
    """Hook to update epsilon-annealing in a Platypus run loop.

    Parameters
    ----------
    formulation
        A formulation with ``set_progress(t, T)`` method.
    nfe_max : int
        Maximum number of function evaluations for the run.

    Example::

        updater = EpsilonUpdater(formulation, 30000)
        for gen in range(n_gen):
            algo.step()
            updater.update(algo)
    """

    def __init__(self, formulation, nfe_max):
        self._form = formulation
        self._nfe_max = nfe_max

    def update(self, algorithm):
        """Call after each generation step."""
        self._form.set_progress(algorithm.nfe, self._nfe_max)
