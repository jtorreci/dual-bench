# -*- coding: utf-8 -*-
"""Speed Reducer (Golinski) problem (7 vars, 11 d/c ratios).

Minimise the weight of a speed reducer (gear box) subject to 11 constraints
on bending stress, surface stress, shaft deflection, and dimensional limits.

References
----------
Golinski, J. (1970). Optimal synthesis problems solved by means of
nonlinear programming and random methods.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from dual_bench.problems.base import DualProblem


class SpeedReducer(DualProblem):
    """Speed Reducer (Golinski): 7 variables, 11 d/c ratios, f* ~ 2994.47."""

    name = "SpeedReducer"
    n_vars = 7
    n_ratios = 11
    var_bounds = [
        (2.6, 3.6), (0.7, 0.8), (17.0, 28.0),
        (7.3, 8.3), (7.3, 8.3), (2.9, 3.9), (5.0, 5.5),
    ]
    known_optimum = 2994.47

    def evaluate(self, x: List[float]) -> Tuple[float, List[float]]:
        x1, x2, x3, x4, x5, x6, x7 = (
            x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        )

        f = (0.7854 * x1 * x2**2 *
             (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
             - 1.508 * x1 * (x6**2 + x7**2)
             + 7.4777 * (x6**3 + x7**3)
             + 0.7854 * (x4 * x6**2 + x5 * x7**2))

        d1 = 27.0 / (x1 * x2**2 * x3) if (x1 * x2**2 * x3) > 0 else 1e6
        d2 = 397.5 / (x1 * x2**2 * x3**2) if (x1 * x2**2 * x3**2) > 0 else 1e6
        d3 = 1.93 * x4**3 / (x2 * x3 * x6**4) if (x2 * x3 * x6**4) > 0 else 1e6
        d4 = 1.93 * x5**3 / (x2 * x3 * x7**4) if (x2 * x3 * x7**4) > 0 else 1e6

        A5 = (745.0 * x4 / (x2 * x3))**2 + 16.9e6 if (x2 * x3) > 0 else 1e12
        d5 = math.sqrt(A5) / (110.0 * x6**3) if x6**3 > 0 else 1e6

        A6 = (745.0 * x5 / (x2 * x3))**2 + 157.5e6 if (x2 * x3) > 0 else 1e12
        d6 = math.sqrt(A6) / (85.0 * x7**3) if x7**3 > 0 else 1e6

        d7 = x2 * x3 / 40.0
        d8 = 5.0 * x2 / x1 if x1 > 0 else 1e6
        d9 = x1 / (12.0 * x2) if x2 > 0 else 1e6
        d10 = (1.5 * x6 + 1.9) / x4 if x4 > 0 else 1e6
        d11 = (1.1 * x7 + 1.9) / x5 if x5 > 0 else 1e6

        return f, [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11]
