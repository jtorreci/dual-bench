# -*- coding: utf-8 -*-
"""Cantilever Beam problem (5 vars, 1 d/c ratio).

Minimise the weight of a cantilever beam with five segments subject to
a single tip deflection constraint.

References
----------
Fleury, C., & Braibant, V. (1986). Structural optimization: a new dual
method using mixed variables.
"""

from __future__ import annotations

from typing import List, Tuple

from dual_bench.problems.base import DualProblem


class CantileverBeam(DualProblem):
    """Cantilever Beam: 5 variables, 1 d/c ratio, f* ~ 1.3400."""

    name = "CantileverBeam"
    n_vars = 5
    n_ratios = 1
    var_bounds = [(0.01, 100.0)] * 5
    known_optimum = 1.3400

    def evaluate(self, x: List[float]) -> Tuple[float, List[float]]:
        f = 0.0624 * sum(x)

        d1 = (61.0 / x[0]**3 + 37.0 / x[1]**3 + 19.0 / x[2]**3 +
              7.0 / x[3]**3 + 1.0 / x[4]**3)

        return f, [d1]
