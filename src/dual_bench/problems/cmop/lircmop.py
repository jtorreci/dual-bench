# -*- coding: utf-8 -*-
"""LIRCMOP test suite -- Large Infeasible Region CMOPs.

14 problems with increasingly challenging constraint landscapes:
- LIRCMOP1-4: Narrow feasibility bands on g-functions.
- LIRCMOP5-8: Elliptical infeasible regions in objective space.
- LIRCMOP9-12: Disconnected Pareto fronts.
- LIRCMOP13-14: 3-objective variants.

References
----------
Fan, Z., Li, W., Cai, X. et al. (2019). An improved epsilon-constrained
method in MOEA/D for CMOPs with large infeasible regions.
Soft Computing, 23(23), 12761-12776.

Convention: ``G_j <= 0`` means feasible.
"""

from math import cos, pi, sin, sqrt

import numpy as np

from dual_bench.problems.base import CMOProblem

# ======================================================================
# Helper functions
# ======================================================================

def _g_sum(x, indices, phase_fn):
    """Sum of (x_i - phase_fn(i, x))^2 for i in indices."""
    return sum((x[i] - phase_fn(i, x))**2 for i in indices)


def _odd_indices(n):
    """Even 0-based indices starting from 2: {2, 4, 6, ...}."""
    return list(range(2, n, 2))


def _even_indices(n):
    """Odd 0-based indices starting from 1: {1, 3, 5, ...}."""
    return list(range(1, n, 2))


def _ellipse_constraint(f1, f2, xc, yc, a, b, r, theta=-0.25*pi):
    """Rotated ellipse constraint (g <= 0 means outside ellipse = feasible)."""
    dx, dy = f1 - xc, f2 - yc
    ct, st = cos(theta), sin(theta)
    u = dx * ct - dy * st
    v = dx * st + dy * ct
    return r - (u / a)**2 - (v / b)**2


# ======================================================================
# Type I: LIRCMOP1-4  (narrow feasibility band on g-functions)
# ======================================================================

class LIRCMOP1(CMOProblem):
    """LIRCMOP1: 2 obj, 2 constr, 30 var (default). Narrow feasibility band."""

    name, n_obj, n_constr = "LIRCMOP1", 2, 2

    def __init__(self, n_var=30):
        self.n_var = n_var
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)
        self._J1 = _odd_indices(n_var)
        self._J2 = _even_indices(n_var)

    def _g1(self, x):
        return sum((x[i] - sin(0.5 * pi * x[0]))**2 for i in self._J1)

    def _g2(self, x):
        return sum((x[i] - cos(0.5 * pi * x[0]))**2 for i in self._J2)

    def _objectives(self, x):
        return x[0] + self._g1(x), 1.0 - x[0]**2 + self._g2(x)

    def evaluate(self, x):
        f1, f2 = self._objectives(x)
        g1_val, g2_val = self._g1(x), self._g2(x)
        c1 = -(0.51 - g1_val) * (g1_val - 0.5)
        c2 = -(0.51 - g2_val) * (g2_val - 0.5)
        return np.array([f1, f2]), np.array([c1, c2])


class LIRCMOP2(LIRCMOP1):
    """LIRCMOP2: variant of LIRCMOP1 with sqrt front shape."""

    name = "LIRCMOP2"

    def _objectives(self, x):
        return x[0] + self._g1(x), 1.0 - sqrt(x[0]) + self._g2(x)


class LIRCMOP3(LIRCMOP1):
    """LIRCMOP3: LIRCMOP1 + sinusoidal constraint. 3 constraints."""

    name, n_constr = "LIRCMOP3", 3

    def evaluate(self, x):
        f1, f2 = self._objectives(x)
        g1_val, g2_val = self._g1(x), self._g2(x)
        c1 = -(0.51 - g1_val) * (g1_val - 0.5)
        c2 = -(0.51 - g2_val) * (g2_val - 0.5)
        c3 = -(sin(20.0 * pi * x[0]) - 0.5)
        return np.array([f1, f2]), np.array([c1, c2, c3])


class LIRCMOP4(LIRCMOP2):
    """LIRCMOP4: LIRCMOP2 + sinusoidal constraint. 3 constraints."""

    name, n_constr = "LIRCMOP4", 3

    def evaluate(self, x):
        f1, f2 = self._objectives(x)
        g1_val, g2_val = self._g1(x), self._g2(x)
        c1 = -(0.51 - g1_val) * (g1_val - 0.5)
        c2 = -(0.51 - g2_val) * (g2_val - 0.5)
        c3 = -(sin(20.0 * pi * x[0]) - 0.5)
        return np.array([f1, f2]), np.array([c1, c2, c3])


# ======================================================================
# Type II: LIRCMOP5-8  (elliptical infeasible regions in obj space)
# ======================================================================

class LIRCMOP5(CMOProblem):
    """LIRCMOP5: 2 obj, 2 constr, 30 var. Elliptical infeasible regions."""

    name, n_obj, n_constr = "LIRCMOP5", 2, 2
    _offset = 0.7057
    _x_offsets = [1.6, 2.5]
    _y_offsets = [1.6, 2.5]
    _a = [2.0, 2.0]
    _b = [4.0, 8.0]
    _r = 0.1

    def __init__(self, n_var=30):
        self.n_var = n_var
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)
        self._J1 = _odd_indices(n_var)
        self._J2 = _even_indices(n_var)

    def _g1(self, x):
        n = len(x)
        return sum((x[i] - sin(0.5 * i / n * pi * x[0]))**2 for i in self._J1)

    def _g2(self, x):
        n = len(x)
        return sum((x[i] - cos(0.5 * i / n * pi * x[0]))**2 for i in self._J2)

    def _objectives(self, x):
        return (x[0] + 10*self._g1(x) + self._offset,
                1.0 - sqrt(x[0]) + 10*self._g2(x) + self._offset)

    def evaluate(self, x):
        f1, f2 = self._objectives(x)
        g = np.array([
            _ellipse_constraint(f1, f2, xc, yc, a, b, self._r)
            for xc, yc, a, b in zip(
                self._x_offsets, self._y_offsets, self._a, self._b)
        ])
        return np.array([f1, f2]), g


class LIRCMOP6(LIRCMOP5):
    """LIRCMOP6: variant with quadratic front shape."""

    name = "LIRCMOP6"
    _x_offsets = [1.8, 2.8]
    _y_offsets = [1.8, 2.8]
    _a = [2.0, 2.0]
    _b = [8.0, 8.0]

    def _objectives(self, x):
        return (x[0] + 10*self._g1(x) + self._offset,
                1.0 - x[0]**2 + 10*self._g2(x) + self._offset)


class LIRCMOP7(LIRCMOP5):
    """LIRCMOP7: LIRCMOP5 with 3 elliptical constraints."""

    name, n_constr = "LIRCMOP7", 3
    _x_offsets = [1.2, 2.25, 3.5]
    _y_offsets = [1.2, 2.25, 3.5]
    _a = [2.0, 2.5, 2.5]
    _b = [6.0, 12.0, 10.0]


class LIRCMOP8(LIRCMOP6):
    """LIRCMOP8: LIRCMOP6 with 3 elliptical constraints."""

    name, n_constr = "LIRCMOP8", 3
    _x_offsets = [1.2, 2.25, 3.5]
    _y_offsets = [1.2, 2.25, 3.5]
    _a = [2.0, 2.5, 2.5]
    _b = [6.0, 12.0, 10.0]


# ======================================================================
# Type III: LIRCMOP9-12  (disconnected Pareto fronts)
# ======================================================================

class LIRCMOP9(LIRCMOP5):
    """LIRCMOP9: disconnected Pareto front with sinusoidal + ellipse constraints."""

    name, n_constr = "LIRCMOP9", 2

    def _objectives(self, x):
        return (1.7057 * x[0] * (10*self._g1(x) + 1),
                1.7957 * (1 - x[0]**2) * (10*self._g2(x) + 1))

    def evaluate(self, x):
        f1, f2 = self._objectives(x)
        theta = -0.25 * pi
        ct, st = cos(theta), sin(theta)
        c1 = -(f1*st + f2*ct - sin(4*pi*(f1*ct - f2*st)) - 2.0)
        c2 = _ellipse_constraint(f1, f2, 1.40, 1.40, 1.5, 6.0, 0.1)
        return np.array([f1, f2]), np.array([c1, c2])


class LIRCMOP10(LIRCMOP5):
    """LIRCMOP10: variant with sqrt front shape."""

    name, n_constr = "LIRCMOP10", 2

    def _objectives(self, x):
        return (1.7057 * x[0] * (10*self._g1(x) + 1),
                1.7957 * (1 - sqrt(x[0])) * (10*self._g2(x) + 1))

    def evaluate(self, x):
        f1, f2 = self._objectives(x)
        theta = -0.25 * pi
        ct, st = cos(theta), sin(theta)
        c1 = -(f1*st + f2*ct - sin(4*pi*(f1*ct - f2*st)) - 1.0)
        c2 = _ellipse_constraint(f1, f2, 1.1, 1.2, 2.0, 4.0, 0.1)
        return np.array([f1, f2]), np.array([c1, c2])


class LIRCMOP11(LIRCMOP10):
    """LIRCMOP11: variant of LIRCMOP10 with different constraint parameters."""

    name = "LIRCMOP11"

    def evaluate(self, x):
        f1, f2 = self._objectives(x)
        theta = -0.25 * pi
        ct, st = cos(theta), sin(theta)
        c1 = -(f1*st + f2*ct - sin(4*pi*(f1*ct - f2*st)) - 2.1)
        c2 = _ellipse_constraint(f1, f2, 1.2, 1.2, 1.5, 5.0, 0.1)
        return np.array([f1, f2]), np.array([c1, c2])


class LIRCMOP12(LIRCMOP9):
    """LIRCMOP12: variant of LIRCMOP9 with different constraint parameters."""

    name = "LIRCMOP12"

    def evaluate(self, x):
        f1, f2 = self._objectives(x)
        theta = -0.25 * pi
        ct, st = cos(theta), sin(theta)
        c1 = -(f1*st + f2*ct - sin(4*pi*(f1*ct - f2*st)) - 2.5)
        c2 = _ellipse_constraint(f1, f2, 1.6, 1.6, 1.5, 6.0, 0.1)
        return np.array([f1, f2]), np.array([c1, c2])


# ======================================================================
# 3-objective: LIRCMOP13-14
# ======================================================================

class LIRCMOP13(CMOProblem):
    """LIRCMOP13: 3 objectives, 2 constraints, spherical Pareto front."""

    name, n_obj, n_constr = "LIRCMOP13", 3, 2

    def __init__(self, n_var=30):
        self.n_var = n_var
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)

    def _g1(self, x):
        return sum(10*(x[i] - 0.5)**2 for i in range(2, len(x), 2))

    def _objectives(self, x):
        g = self._g1(x)
        f1 = (1.7057 + g) * cos(0.5*pi*x[0]) * cos(0.5*pi*x[1])
        f2 = (1.7057 + g) * cos(0.5*pi*x[0]) * sin(0.5*pi*x[1])
        f3 = (1.7057 + g) * sin(0.5*pi*x[0])
        return f1, f2, f3

    def evaluate(self, x):
        f1, f2, f3 = self._objectives(x)
        s = f1**2 + f2**2 + f3**2
        c1 = -((s - 9.0) * (s - 4.0))
        c2 = -((s - 1.9**2) * (s - 1.8**2))
        return np.array([f1, f2, f3]), np.array([c1, c2])


class LIRCMOP14(LIRCMOP13):
    """LIRCMOP14: 3 objectives, 3 constraints."""

    name, n_constr = "LIRCMOP14", 3

    def evaluate(self, x):
        f1, f2, f3 = self._objectives(x)
        s = f1**2 + f2**2 + f3**2
        c1 = -((s - 9.0) * (s - 4.0))
        c2 = -((s - 1.9**2) * (s - 1.8**2))
        c3 = -((s - 1.75**2) * (s - 1.6**2))
        return np.array([f1, f2, f3]), np.array([c1, c2, c3])
