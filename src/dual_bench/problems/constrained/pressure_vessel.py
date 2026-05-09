# -*- coding: utf-8 -*-
"""Pressure Vessel Design problem (4 vars, 4 d/c ratios).

Minimise the total cost of a cylindrical pressure vessel subject to
constraints on shell thickness, head thickness, volume, and length.

References
----------
Kannan, B. K., & Kramer, S. N. (1994). An augmented Lagrange multiplier
based method for mixed integer discrete continuous optimization.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from dual_bench.problems.base import DualProblem


class PressureVessel(DualProblem):
    """Pressure Vessel Design: 4 variables, 4 d/c ratios, f* = 5885.33 (continuous)."""

    name = "PressureVessel"
    n_vars = 4
    n_ratios = 4
    var_bounds = [(0.0625, 6.1875), (0.0625, 6.1875),
                  (10.0, 200.0), (10.0, 200.0)]
    known_optimum = 5885.33

    def evaluate(self, x: List[float]) -> Tuple[float, List[float]]:
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

        f = (0.6224 * x1 * x3 * x4 +
             1.7781 * x2 * x3**2 +
             3.1661 * x1**2 * x4 +
             19.84 * x1**2 * x3)

        d1 = 0.0193 * x3 / x1 if x1 > 0 else 1e6
        d2 = 0.00954 * x3 / x2 if x2 > 0 else 1e6

        volume = math.pi * x3**2 * x4 + 4 * math.pi * x3**3 / 3
        d3 = 1296000 / volume if volume > 0 else 1e6

        d4 = x4 / 240.0

        return f, [d1, d2, d3, d4]
