# -*- coding: utf-8 -*-
"""Abstract base classes for dual-bench problems.

Two base classes are provided:

- :class:`DualProblem` for single-objective problems with demand-to-capacity
  ratios (Papers A, B).
- :class:`CMOProblem` for constrained multi-objective problems (Paper D).
"""

from __future__ import annotations

import abc
from typing import List, Tuple

import numpy as np

# ======================================================================
# CMOProblem -- constrained multi-objective problems
# ======================================================================

class CMOProblem(abc.ABC):
    """Base class for constrained multi-objective optimisation problems.

    Subclasses must set:
        name     : str
        n_var    : int
        n_obj    : int
        n_constr : int
        xl       : np.ndarray   -- lower bounds [n_var]
        xu       : np.ndarray   -- upper bounds [n_var]

    And implement:
        evaluate(x) -> (F [n_obj], G [n_constr])

    Convention: ``G_j <= 0`` means constraint *j* is satisfied (feasible).
    For unconstrained problems, set ``n_constr = 0`` and return an empty array.
    """

    name: str = ""
    n_var: int = 0
    n_obj: int = 0
    n_constr: int = 0
    xl: np.ndarray = np.array([])
    xu: np.ndarray = np.array([])

    @abc.abstractmethod
    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (objectives [n_obj], constraints [n_constr])."""

    def is_feasible(self, x: np.ndarray) -> bool:
        """Return True when all constraints are satisfied (G_j <= 0)."""
        _, g = self.evaluate(x)
        if len(g) == 0:
            return True
        return bool(np.all(g <= 0))

    def pareto_front(self, n_points: int = 100) -> np.ndarray | None:
        """Return reference Pareto front if known, else None."""
        return None

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}('{self.name}', "
                f"{self.n_var}var, {self.n_obj}obj, {self.n_constr}con)")


# ======================================================================
# DualProblem -- single-objective d/c problems
# ======================================================================

class DualProblem(abc.ABC):
    """Base class for a constrained optimisation problem in dual d/c form.

    Every problem exposes ``evaluate(x) -> (f, [d_1, ..., d_n])`` where:
        - *f* is the scalar objective to minimise
        - *d_i* are demand/capacity ratios (``d_i <= 1`` means feasible)

    Subclasses must set:
        name          : str
        n_vars        : int
        n_ratios      : int
        var_bounds    : List[Tuple[float, float]]
        known_optimum : float | None

    And implement:
        evaluate(x) -> Tuple[float, List[float]]
    """

    name: str = ""
    n_vars: int = 0
    n_ratios: int = 0
    var_bounds: List[Tuple[float, float]] = []
    known_optimum: float | None = None

    @abc.abstractmethod
    def evaluate(self, x: List[float]) -> Tuple[float, List[float]]:
        """Evaluate the problem at point *x*.

        Parameters
        ----------
        x : list of float, length == n_vars

        Returns
        -------
        (f, ratios) where
            f      : float        -- objective value
            ratios : list[float]  -- d/c ratios [d_1, ..., d_n_ratios]
        """

    def is_feasible(self, x: List[float]) -> bool:
        """Return True when all d/c ratios are <= 1.0."""
        _, ratios = self.evaluate(x)
        return all(d <= 1.0 for d in ratios)

    def midpoint(self) -> List[float]:
        """Return the centre of the variable bounds."""
        return [(lb + ub) / 2.0 for lb, ub in self.var_bounds]

    # ------------------------------------------------------------------
    # formulation factories
    # ------------------------------------------------------------------
    def to_mo(self, penalty: float = 1e6):
        """Create a single-objective (MO) formulation wrapper."""
        from dual_bench.formulations import MOFormulation
        return MOFormulation(self, penalty=penalty)

    def to_2ob(self):
        """Create a bi-objective (2OB) formulation wrapper."""
        from dual_bench.formulations import BiobjFormulation
        return BiobjFormulation(self)

    def to_fob(self, d_cap: float = 5.0):
        """Create a full many-objective (FOB) formulation wrapper."""
        from dual_bench.formulations import ManyobjFormulation
        return ManyobjFormulation(self, d_cap=d_cap)

    def to_2ob_rest(self):
        """Create a bi-objective + constraints (2OB-REST) formulation wrapper."""
        from dual_bench.formulations import BiobjRestFormulation
        return BiobjRestFormulation(self)

    def to_fob_rest(self, d_cap: float = 5.0):
        """Create a many-objective + constraints (FOB-REST) formulation wrapper."""
        from dual_bench.formulations import ManyobjRestFormulation
        return ManyobjRestFormulation(self, d_cap=d_cap)

    def info(self) -> str:
        """Return a human-readable summary of this problem."""
        lines = [
            f"Problem : {self.name}",
            f"  n_vars   = {self.n_vars}",
            f"  n_ratios = {self.n_ratios}",
            f"  bounds   = {self.var_bounds}",
            f"  known f* = {self.known_optimum}",
        ]
        text = "\n".join(lines)
        print(text)
        return text
