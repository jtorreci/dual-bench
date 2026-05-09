#!/usr/bin/env python3
"""Demonstrate how to create a custom problem and use it with dual-bench."""

from dual_bench.problems.base import DualProblem


class ThreeBarTruss(DualProblem):
    """Three-bar truss design (2 vars, 3 d/c ratios).

    Minimise the volume of a three-bar planar truss subject to
    stress constraints on each member.

    f* ~ 263.896
    """

    name = "ThreeBarTruss"
    n_vars = 2
    n_ratios = 3
    var_bounds = [(0.0, 1.0), (0.0, 1.0)]
    known_optimum = 263.896

    def evaluate(self, x):
        import math

        x1, x2 = x[0], x[1]
        L = 100.0
        P = 2.0
        sigma_max = 2.0

        # Objective: volume
        f = L * (2 * math.sqrt(2) * x1 + x2)

        # Stresses (demand)
        denom = 2 * math.sqrt(2) * x1**2 + 2 * x1 * x2
        if denom <= 0:
            return f, [1e6, 1e6, 1e6]

        s1 = P * (2 * x1 + x2) / denom
        s2 = P * x2 / denom
        s3 = P / (x1 + math.sqrt(2) * x2) if (x1 + math.sqrt(2) * x2) > 0 else 1e6

        # d/c ratios: stress / sigma_max <= 1
        d1 = abs(s1) / sigma_max
        d2 = abs(s2) / sigma_max
        d3 = abs(s3) / sigma_max

        return f, [d1, d2, d3]


if __name__ == "__main__":
    # Create and test
    truss = ThreeBarTruss()
    truss.info()

    # Test formulations
    print("\n--- Formulations ---")
    x = truss.midpoint()
    for name, form_fn in [
        ("MO", truss.to_mo),
        ("2OB", truss.to_2ob),
        ("2OB-REST", truss.to_2ob_rest),
    ]:
        form = form_fn()
        objs, cons = form.evaluate(x)
        print(f"  {name}: {form.n_objectives} obj, {form.n_constraints} con")

    print("\nCustom problem works with all dual-bench formulations!")
