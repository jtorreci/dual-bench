# -*- coding: utf-8 -*-
"""Formulation wrappers for dual-bench problems.

Each wrapper translates a problem into a specific multi-objective /
constrained layout that an optimisation engine can consume.  All wrappers
expose a uniform interface::

    evaluate(x) -> (objectives: List[float], constraints: List[float])
    n_objectives : int
    n_constraints : int
    var_bounds    : List[Tuple[float, float]]

All formulations use the **G <= 0** constraint convention: a constraint
value <= 0 means feasible.  For SO formulations, constraints are
``d_i - 1`` (so feasible when d_i <= 1).  CMOP formulations follow the
same convention natively.

Single-objective formulations (wrap :class:`DualProblem`):

==========================  ============  =================
Class                       Objectives    Constraints
==========================  ============  =================
MOFormulation               1 (f+penalty) n_ratios (d_i-1)
BiobjFormulation            2 (f, max d)  0
ManyobjFormulation          n_ratios+1    0
BiobjRestFormulation        2 (f, max d)  n_ratios (d_i-1)
ManyobjRestFormulation      n_ratios+1    n_ratios (d_i-1)
==========================  ============  =================

CMOP formulations (wrap :class:`CMOProblem`):

==========================  ==================  ==============  =========
Class                       Objectives          Constraints     Annealing
==========================  ==================  ==============  =========
CMOPStandard                n_obj (original)    n_constr (orig) No
CMOPEnvelope                n_obj+1 (+max d_j)  n_constr        Yes
CMOPFull                    n_obj+n_constr      n_constr        Yes
==========================  ==================  ==============  =========
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from dual_bench.problems.base import CMOProblem, DualProblem
from dual_bench.schedules import DEFAULT_CP, DEFAULT_EPS_MAX, epsilon_pow


class MOFormulation:
    """Single-objective with quadratic penalty and explicit constraints.

    ``objective[0] = f + penalty * sum(max(0, d_i - 1)^2)``

    ``constraints = [d_1 - 1, ..., d_n - 1]``  (each <= 0 when feasible)
    """

    def __init__(self, problem: DualProblem, penalty: float = 1e6):
        self.problem = problem
        self.penalty = penalty
        self.n_objectives = 1
        self.n_constraints = problem.n_ratios
        self.var_bounds = list(problem.var_bounds)

    def evaluate(self, x: List[float]) -> Tuple[List[float], List[float]]:
        f, ratios = self.problem.evaluate(x)
        pen = self.penalty * sum(max(0.0, d - 1.0)**2 for d in ratios)
        return [f + pen], [d - 1.0 for d in ratios]


class BiobjFormulation:
    """Bi-objective formulation: minimise ``(f, max d_i)``, no constraints."""

    def __init__(self, problem: DualProblem):
        self.problem = problem
        self.n_objectives = 2
        self.n_constraints = 0
        self.var_bounds = list(problem.var_bounds)

    def evaluate(self, x: List[float]) -> Tuple[List[float], List[float]]:
        f, ratios = self.problem.evaluate(x)
        return [f, max(ratios)], []


class ManyobjFormulation:
    """Full many-objective formulation: minimise ``(f, d_1, ..., d_n)``.

    Each d_i is capped at *d_cap* to prevent scale disparity.
    """

    def __init__(self, problem: DualProblem, d_cap: float = 5.0):
        self.problem = problem
        self.d_cap = d_cap
        self.n_objectives = 1 + problem.n_ratios
        self.n_constraints = 0
        self.var_bounds = list(problem.var_bounds)

    def evaluate(self, x: List[float]) -> Tuple[List[float], List[float]]:
        f, ratios = self.problem.evaluate(x)
        capped = [min(d, self.d_cap) for d in ratios]
        return [f] + capped, []


class BiobjRestFormulation:
    """Bi-objective with explicit constraints (2OB-REST).

    ``objectives  = [f, max(d_i)]``

    ``constraints = [d_1 - 1, ..., d_n - 1]``  (each <= 0 when feasible)
    """

    def __init__(self, problem: DualProblem):
        self.problem = problem
        self.n_objectives = 2
        self.n_constraints = problem.n_ratios
        self.var_bounds = list(problem.var_bounds)

    def evaluate(self, x: List[float]) -> Tuple[List[float], List[float]]:
        f, ratios = self.problem.evaluate(x)
        return [f, max(ratios)], [d - 1.0 for d in ratios]


class ManyobjRestFormulation:
    """Many-objective with explicit constraints (FOB-REST).

    ``objectives  = [f, min(d_1, cap), ..., min(d_n, cap)]``

    ``constraints = [d_1 - 1, ..., d_n - 1]``  (each <= 0 when feasible)
    """

    def __init__(self, problem: DualProblem, d_cap: float = 5.0):
        self.problem = problem
        self.d_cap = d_cap
        self.n_objectives = 1 + problem.n_ratios
        self.n_constraints = problem.n_ratios
        self.var_bounds = list(problem.var_bounds)

    def evaluate(self, x: List[float]) -> Tuple[List[float], List[float]]:
        f, ratios = self.problem.evaluate(x)
        capped = [min(d, self.d_cap) for d in ratios]
        return [f] + capped, [d - 1.0 for d in ratios]


# ======================================================================
# CMOP dual-formulation wrappers (Paper D)
# ======================================================================

class CMOPEnvelope:
    """Envelope dual formulation for CMOPs.

    Converts ``min [f1,...,fk] s.t. g_j<=0`` into::

        min [f1,...,fk, max(d_j)]  s.t. d_j <= 1 + eps(t)

    where ``d_j = 1 + g_j / s_j`` and eps(t) decays from eps_max to 0.
    """

    def __init__(self, problem: CMOProblem, scales=None,
                 eps_max=DEFAULT_EPS_MAX, cp=DEFAULT_CP):
        self.problem = problem
        self.n_objectives = problem.n_obj + 1
        self.n_constraints = problem.n_constr
        self.var_bounds = list(zip(problem.xl, problem.xu))
        self.xl = problem.xl
        self.xu = problem.xu
        self.eps_max = eps_max
        self.cp = cp
        self._epsilon = eps_max
        self._scales = scales

    @property
    def epsilon(self):
        """Current epsilon value."""
        return self._epsilon

    def set_progress(self, t, T):
        """Update epsilon based on current NFE progress."""
        self._epsilon = epsilon_pow(t, T, self.eps_max, self.cp)

    def _compute_d(self, g):
        """Convert constraint values g (<=0 feasible) to ratios d (<=1 feasible)."""
        if self._scales is not None:
            return 1.0 + g / self._scales
        return 1.0 + g

    def evaluate(self, x):
        """Return (objectives [n_obj+1], constraints [n_constr])."""
        f, g = self.problem.evaluate(np.asarray(x, dtype=float))
        d = self._compute_d(g)
        max_d = float(np.max(d)) if len(d) > 0 else 0.0
        objectives = list(f) + [max_d]
        constraints = list(d - (1.0 + self._epsilon))
        return objectives, constraints


class CMOPFull:
    """Full dual formulation for CMOPs.

    Converts ``min [f1,...,fk] s.t. g_j<=0`` into::

        min [f1,...,fk, d_1,...,d_J]  s.t. d_j <= 1 + eps(t)

    where ``d_j = 1 + g_j / s_j`` and each d_j is capped at *d_cap*.
    """

    def __init__(self, problem: CMOProblem, scales=None,
                 eps_max=DEFAULT_EPS_MAX, cp=DEFAULT_CP, d_cap=5.0):
        self.problem = problem
        self.n_objectives = problem.n_obj + problem.n_constr
        self.n_constraints = problem.n_constr
        self.var_bounds = list(zip(problem.xl, problem.xu))
        self.xl = problem.xl
        self.xu = problem.xu
        self.eps_max = eps_max
        self.cp = cp
        self.d_cap = d_cap
        self._epsilon = eps_max
        self._scales = scales

    @property
    def epsilon(self):
        """Current epsilon value."""
        return self._epsilon

    def set_progress(self, t, T):
        """Update epsilon based on current NFE progress."""
        self._epsilon = epsilon_pow(t, T, self.eps_max, self.cp)

    def _compute_d(self, g):
        if self._scales is not None:
            return 1.0 + g / self._scales
        return 1.0 + g

    def evaluate(self, x):
        """Return (objectives [n_obj+n_constr], constraints [n_constr])."""
        f, g = self.problem.evaluate(np.asarray(x, dtype=float))
        d = self._compute_d(g)
        d_capped = np.minimum(d, self.d_cap)
        objectives = list(f) + list(d_capped)
        constraints = list(d - (1.0 + self._epsilon))
        return objectives, constraints


class CMOPStandard:
    """Standard CMOP formulation (baseline, no dual transformation).

    Keeps original objectives and constraints unchanged.
    """

    def __init__(self, problem: CMOProblem):
        self.problem = problem
        self.n_objectives = problem.n_obj
        self.n_constraints = problem.n_constr
        self.var_bounds = list(zip(problem.xl, problem.xu))
        self.xl = problem.xl
        self.xu = problem.xu

    def set_progress(self, t, T):
        """No-op (no annealing in standard formulation)."""

    def evaluate(self, x):
        """Return (objectives [n_obj], constraints [n_constr])."""
        f, g = self.problem.evaluate(np.asarray(x, dtype=float))
        return list(f), list(g)
