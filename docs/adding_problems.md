# Adding new problems

This guide shows how to add a custom problem to `dual-bench`.

## Single-objective problem (DualProblem)

Subclass `DualProblem` and implement `evaluate`:

```python
from dual_bench.problems.base import DualProblem

class MyBeam(DualProblem):
    name = "MyBeam"
    n_vars = 3
    n_ratios = 2
    var_bounds = [(0.1, 10.0), (0.1, 10.0), (0.1, 10.0)]
    known_optimum = 42.0  # or None if unknown

    def evaluate(self, x):
        x1, x2, x3 = x[0], x[1], x[2]

        # Objective: weight
        f = x1 * x2 * x3

        # d/c ratios (d_i <= 1 means feasible)
        d1 = 100.0 / (x1 * x2)     # stress demand / capacity
        d2 = x3 / 5.0               # deflection demand / capacity

        return f, [d1, d2]
```

The key contract is:
- `evaluate(x)` returns `(f, [d_1, ..., d_n])`.
- `d_i <= 1` means constraint *i* is satisfied.
- `d_i > 1` means constraint *i* is violated.

You can then use all formulation wrappers:
```python
beam = MyBeam()
form = beam.to_2ob_rest()  # works automatically
```

## Multi-objective problem (CMOProblem)

Subclass `CMOProblem`:

```python
import numpy as np
from dual_bench.problems.base import CMOProblem

class MyMOP(CMOProblem):
    name = "MyMOP"
    n_var = 2
    n_obj = 2
    n_constr = 1
    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])

    def evaluate(self, x):
        f1 = x[0]
        f2 = 1.0 - x[0]**0.5 + x[1]
        g1 = x[0] + x[1] - 1.5   # g <= 0 means feasible
        return np.array([f1, f2]), np.array([g1])
```

Convention: `G_j <= 0` means feasible.

## Registering in a suite

To include your problem in a custom suite, extend the registry:

```python
from dual_bench.problems.suites import _SUITES

def _my_suite():
    return [MyBeam(), MyMOP()]

_SUITES["my_suite"] = (_my_suite, False)  # False = no pymoo required
```

## Running tests

After adding a problem, verify it works:

```python
# Quick check
p = MyBeam()
f, ratios = p.evaluate(p.midpoint())
assert len(ratios) == p.n_ratios

# All formulations
for form_fn in [p.to_mo, p.to_2ob, p.to_fob, p.to_2ob_rest, p.to_fob_rest]:
    form = form_fn()
    objs, cons = form.evaluate(p.midpoint())
    assert len(objs) == form.n_objectives
    assert len(cons) == form.n_constraints
```

Or add it to the test parametrisation in `tests/test_problems.py`:

```python
from dual_bench.problems import ALL_PROBLEMS

# Add to the ALL_PROBLEMS list or parametrize directly
@pytest.mark.parametrize("ProblemClass", ALL_PROBLEMS + [MyBeam])
```
