# Quick start

## Installation

```bash
pip install dual-bench[all]
```

This installs `dual-bench` with both pymoo and Platypus backends.

## Core concepts

`dual-bench` separates three concerns:

1. **Problems** define the physics (objective + constraints).
2. **Formulations** wrap problems into specific optimisation layouts
   (bi-objective, many-objective, envelope, etc.).
3. **Engine adapters** translate formulations into solver-specific representations.

## Single-objective example

```python
import dual_bench

# 1. Create a problem
spring = dual_bench.Spring()
print(spring.info())
# Problem : Spring
#   n_vars   = 3
#   n_ratios = 4

# 2. Wrap in a formulation
form = spring.to_2ob_rest()  # bi-objective + constraints
print(f"{form.n_objectives} objectives, {form.n_constraints} constraints")
# 2 objectives, 4 constraints

# 3. Evaluate at any point
x = spring.midpoint()
objectives, constraints = form.evaluate(x)
print(f"f = {objectives[0]:.4f}, max_d = {objectives[1]:.4f}")
```

## Available formulations

### For single-objective problems (`DualProblem`)

| Method | Objectives | Constraints | Description |
|--------|-----------|------------|-------------|
| `to_mo()` | 1 | n_ratios | Penalty + explicit constraints |
| `to_2ob()` | 2 | 0 | Bi-objective (f, max d) |
| `to_fob()` | n_ratios+1 | 0 | Many-objective (f, d_1, ..., d_n) |
| `to_2ob_rest()` | 2 | n_ratios | Bi-objective + constraints |
| `to_fob_rest()` | n_ratios+1 | n_ratios | Many-objective + constraints |

### For multi-objective problems (`CMOProblem`)

| Class | Objectives | Constraints | Annealing |
|-------|-----------|------------|-----------|
| `CMOPStandard` | n_obj | n_constr | No |
| `CMOPEnvelope` | n_obj+1 | n_constr | Yes |
| `CMOPFull` | n_obj+n_constr | n_constr | Yes |

## Running with pymoo

```python
from dual_bench import Spring
from dual_bench.engines.pymoo_engine import ToPymoo, EpsilonCallback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# Problem + formulation
form = Spring().to_2ob_rest()
nfe = 30000

# Run
result = minimize(
    ToPymoo(form),
    NSGA2(pop_size=100),
    ('n_eval', nfe),
    verbose=False,
)

# Extract results
print(f"Found {len(result.F)} solutions")
```

## Running with Platypus

```python
from dual_bench.problems.cmop import BNH
from dual_bench.formulations import CMOPEnvelope
from dual_bench.engines.platypus_engine import ToPlatypus, EpsilonUpdater
from platypus import NSGAII

# Problem + formulation with epsilon-annealing
form = CMOPEnvelope(BNH())
nfe = 30000
n_gen = nfe // 100

# Run
algo = NSGAII(ToPlatypus(form), population_size=100)
updater = EpsilonUpdater(form, nfe)
for gen in range(n_gen):
    algo.step()
    updater.update(algo)

print(f"Found {len(algo.result)} solutions")
```

## Using suites

```python
import dual_bench

# List available suites
print(dual_bench.list_suites())

# Load all 5 engineering benchmarks
for problem in dual_bench.suite("engineering5"):
    form = problem.to_2ob()
    x = problem.midpoint()
    obj, _ = form.evaluate(x)
    print(f"{problem.name}: f={obj[0]:.4f}, max_d={obj[1]:.4f}")
```

## Epsilon-annealing

CMOP formulations support constraint annealing: the feasibility boundary
relaxes initially (epsilon > 0) and tightens as the run progresses.

```python
from dual_bench.schedules import epsilon_pow

# At the start: epsilon is large (relaxed)
print(epsilon_pow(0, 30000))       # 0.2

# At the end: epsilon is zero (strict)
print(epsilon_pow(30000, 30000))   # 0.0
```

The `EpsilonCallback` (pymoo) and `EpsilonUpdater` (Platypus) handle this
automatically during a run.
