# -*- coding: utf-8 -*-
"""Welded Beam Design problem (4 vars, 4 d/c ratios).

Minimise the fabrication cost of a welded beam subject to constraints
on shear stress, bending stress, deflection, and buckling.

References
----------
Ragsdell, K. M., & Phillips, D. T. (1976). Optimal design of a class of
welded structures using geometric programming.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from dual_bench.problems.base import DualProblem


class WeldedBeam(DualProblem):
    """Welded Beam Design: 4 variables, 4 d/c ratios, f* ~ 1.8496."""

    name = "WeldedBeam"
    n_vars = 4
    n_ratios = 4
    var_bounds = [(0.1, 2.0), (0.1, 10.0), (0.1, 10.0), (0.1, 2.0)]
    known_optimum = 1.8496

    def evaluate(self, x: List[float]) -> Tuple[float, List[float]]:
        h, wl, t, b = x[0], x[1], x[2], x[3]

        P = 6000.0
        L = 14.0
        E = 30e6
        G = 12e6
        tau_max = 13600.0
        sigma_max = 30000.0
        delta_max = 0.25

        f = 1.10471 * h**2 * wl + 0.04811 * t * b * (14.0 + wl)

        M = P * (L + wl / 2)
        R = math.sqrt(wl**2 / 4 + ((h + t) / 2)**2)
        J = 2 * (math.sqrt(2) * h * wl *
                 (wl**2 / 12 + ((h + t) / 2)**2))

        tau_prime = P / (math.sqrt(2) * h * wl) if (h * wl) > 0 else 1e12
        tau_double = M * R / J if J > 0 else 1e12
        tau = (math.sqrt(tau_prime**2 + 2 * tau_prime * tau_double *
                         (wl / 2) / R + tau_double**2)
               if R > 0 else 1e12)

        sigma = 6 * P * L / (b * t**2) if (b * t**2) > 0 else 1e12
        delta = 4 * P * L**3 / (E * b * t**3) if (E * b * t**3) > 0 else 1e12

        Pc = (4.013 * E * math.sqrt(t**2 * b**6 / 36) / L**2 *
              (1 - t / (2 * L) * math.sqrt(E / (4 * G))))
        if Pc <= 0:
            Pc = 1e-12

        d1 = tau / tau_max
        d2 = sigma / sigma_max
        d3 = delta / delta_max
        d4 = P / Pc

        return f, [d1, d2, d3, d4]
