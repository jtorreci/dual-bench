# -*- coding: utf-8 -*-
"""Classic constrained multi-objective benchmark problems.

All problems use the convention ``G_j <= 0`` means feasible.

Problems
--------
BNH    : Binh & Korn (1997), 2 obj, 2 constr, 2 var.
SRN    : Srinivas & Deb (1994), 2 obj, 2 constr, 2 var.
TNK    : Tanaka et al. (1995), 2 obj, 2 constr, 2 var.
OSY    : Osyczka & Kundu (1995), 2 obj, 6 constr, 6 var.
CONSTR : Deb (2002), 2 obj, 2 constr, 2 var.
"""

from math import atan2, cos, pi

import numpy as np

from dual_bench.problems.base import CMOProblem


class BNH(CMOProblem):
    """Binh & Korn (1997).  2 obj, 2 constr, 2 var."""

    name = "BNH"
    n_var, n_obj, n_constr = 2, 2, 2
    xl = np.array([0.0, 0.0])
    xu = np.array([5.0, 3.0])

    def evaluate(self, x):
        f1 = 4.0 * x[0]**2 + 4.0 * x[1]**2
        f2 = (x[0] - 5.0)**2 + (x[1] - 5.0)**2
        g1 = (x[0] - 5.0)**2 + x[1]**2 - 25.0
        g2 = -(x[0] - 8.0)**2 - (x[1] + 3.0)**2 + 7.7
        return np.array([f1, f2]), np.array([g1, g2])


class SRN(CMOProblem):
    """Srinivas & Deb (1994).  2 obj, 2 constr, 2 var."""

    name = "SRN"
    n_var, n_obj, n_constr = 2, 2, 2
    xl = np.array([-20.0, -20.0])
    xu = np.array([20.0, 20.0])

    def evaluate(self, x):
        f1 = 2.0 + (x[0] - 2.0)**2 + (x[1] - 1.0)**2
        f2 = 9.0 * x[0] - (x[1] - 1.0)**2
        g1 = x[0]**2 + x[1]**2 - 225.0
        g2 = x[0] - 3.0 * x[1] + 10.0
        return np.array([f1, f2]), np.array([g1, g2])


class TNK(CMOProblem):
    """Tanaka et al. (1995).  2 obj, 2 constr, 2 var."""

    name = "TNK"
    n_var, n_obj, n_constr = 2, 2, 2
    xl = np.array([1e-30, 1e-30])
    xu = np.array([pi, pi])

    def evaluate(self, x):
        f1, f2 = x[0], x[1]
        g1 = -(x[0]**2 + x[1]**2 - 1.0
               - 0.1 * cos(16.0 * atan2(x[0], x[1])))
        g2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.5
        return np.array([f1, f2]), np.array([g1, g2])


class OSY(CMOProblem):
    """Osyczka & Kundu (1995).  2 obj, 6 constr, 6 var."""

    name = "OSY"
    n_var, n_obj, n_constr = 6, 2, 6
    xl = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    xu = np.array([10.0, 10.0, 5.0, 6.0, 5.0, 10.0])

    def evaluate(self, x):
        f1 = -(25.0*(x[0]-2)**2 + (x[1]-2)**2 + (x[2]-1)**2
               + (x[3]-4)**2 + (x[4]-1)**2)
        f2 = x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2
        g = np.array([
            2.0 - x[0] - x[1],
            x[0] + x[1] - 6.0,
            x[1] - x[0] - 2.0,
            x[0] - 3.0*x[1] - 2.0,
            (x[2]-3.0)**2 + x[3] - 4.0,
            4.0 - (x[4]-3.0)**2 - x[5],
        ])
        return np.array([f1, f2]), g


class CONSTR(CMOProblem):
    """Constrained problem from Deb (2002).  2 obj, 2 constr, 2 var."""

    name = "CONSTR"
    n_var, n_obj, n_constr = 2, 2, 2
    xl = np.array([0.1, 0.0])
    xu = np.array([1.0, 5.0])

    def evaluate(self, x):
        f1 = x[0]
        f2 = (1.0 + x[1]) / x[0]
        g1 = 6.0 - x[1] - 9.0*x[0]
        g2 = 1.0 + x[1] - 9.0*x[0]
        return np.array([f1, f2]), np.array([g1, g2])
