# -*- coding: utf-8 -*-
"""Spring Design problem (3 vars, 4 d/c ratios).

Minimise the weight of a tension/compression spring subject to constraints
on buckling, shear stress, surge frequency, and diameter.

References
----------
Arora, J. S. (2004). *Introduction to Optimum Design*, 2nd ed.
"""

from __future__ import annotations

from typing import List, Tuple

from dual_bench.problems.base import DualProblem


class Spring(DualProblem):
    """Spring Design: 3 variables, 4 d/c ratios, f* = 0.012665."""

    name = "Spring"
    n_vars = 3
    n_ratios = 4
    var_bounds = [(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)]
    known_optimum = 0.012665

    def evaluate(self, x: List[float]) -> Tuple[float, List[float]]:
        x1, x2, x3 = x[0], x[1], x[2]

        f = (x3 + 2) * x2 * x1**2

        d1 = 71785 * x1**4 / (x2**3 * x3)

        g2 = ((4 * x2**2 - x1 * x2) /
              (12566 * (x2 * x1**3 - x1**4)) +
              1 / (5108 * x1**2) - 1)
        d2 = max(0.0, g2 + 1)

        d3 = x2**2 * x3 / (140.45 * x1)

        d4 = (x1 + x2) / 1.5

        return f, [d1, d2, d3, d4]
