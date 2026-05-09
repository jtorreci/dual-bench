# API reference

## Top-level exports

```python
import dual_bench

dual_bench.__version__       # "1.1.0"
dual_bench.list_suites()     # list of suite names
dual_bench.suite(name)       # list of problem instances
```

## Problems

### `dual_bench.DualProblem` (abstract)

Single-objective problem with demand-to-capacity ratios.

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | str | Human-readable name |
| `n_vars` | int | Number of decision variables |
| `n_ratios` | int | Number of d/c constraint ratios |
| `var_bounds` | list of (float, float) | Variable bounds |
| `known_optimum` | float or None | Best known objective value |

| Method | Returns | Description |
|--------|---------|-------------|
| `evaluate(x)` | (float, list[float]) | Objective + d/c ratios |
| `is_feasible(x)` | bool | All ratios <= 1? |
| `midpoint()` | list[float] | Centre of bounds |
| `to_mo(penalty)` | MOFormulation | Single-objective wrapper |
| `to_2ob()` | BiobjFormulation | Bi-objective wrapper |
| `to_fob(d_cap)` | ManyobjFormulation | Many-objective wrapper |
| `to_2ob_rest()` | BiobjRestFormulation | Bi-obj + constraints |
| `to_fob_rest(d_cap)` | ManyobjRestFormulation | Many-obj + constraints |
| `info()` | str | Print summary |

Concrete implementations: `Spring`, `PressureVessel`, `WeldedBeam`,
`SpeedReducer`, `CantileverBeam`.

### `dual_bench.CMOProblem` (abstract)

Constrained multi-objective problem (G_j <= 0 = feasible).

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | str | Problem name |
| `n_var` | int | Decision variables |
| `n_obj` | int | Objectives |
| `n_constr` | int | Constraints |
| `xl`, `xu` | np.ndarray | Bounds |

| Method | Returns | Description |
|--------|---------|-------------|
| `evaluate(x)` | (ndarray, ndarray) | (F, G) |
| `is_feasible(x)` | bool | All G_j <= 0? |
| `pareto_front(n_points)` | ndarray or None | Reference PF |

Concrete implementations: `BNH`, `SRN`, `TNK`, `OSY`, `CONSTR`,
`LIRCMOP1`-`LIRCMOP14`, plus pymoo wrappers via `from_pymoo()`.

## Formulations

All formulations expose:

```python
form.n_objectives   # int
form.n_constraints  # int
form.var_bounds     # list of (float, float)
form.evaluate(x)    # -> (objectives, constraints)
```

### Single-objective formulations

| Class | Objectives | Constraints |
|-------|-----------|------------|
| `MOFormulation(problem, penalty=1e6)` | 1 | n_ratios |
| `BiobjFormulation(problem)` | 2 | 0 |
| `ManyobjFormulation(problem, d_cap=5.0)` | n_ratios+1 | 0 |
| `BiobjRestFormulation(problem)` | 2 | n_ratios |
| `ManyobjRestFormulation(problem, d_cap=5.0)` | n_ratios+1 | n_ratios |

### CMOP formulations

| Class | Extra params | Objectives | Constraints |
|-------|-------------|-----------|------------|
| `CMOPStandard(problem)` | - | n_obj | n_constr |
| `CMOPEnvelope(problem, scales, eps_max, cp)` | annealing | n_obj+1 | n_constr |
| `CMOPFull(problem, scales, eps_max, cp, d_cap)` | annealing | n_obj+n_constr | n_constr |

CMOP formulations with annealing support `set_progress(t, T)` and expose
an `epsilon` property.

## Schedules

```python
from dual_bench.schedules import epsilon_pow, epsilon_lin, epsilon_exp

epsilon_pow(t, T, eps_max=0.2, cp=5)   # power-law (default)
epsilon_lin(t, T, eps_max=0.2)          # linear
epsilon_exp(t, T, eps_max=0.2, ce=5)    # exponential
```

## Suites

```python
dual_bench.list_suites()    # -> ['classic5', 'dascmop', 'dtlz', ...]
dual_bench.suite("zdt")     # -> [PymooWrapper('ZDT1'), ...]
```

## Engine adapters

### pymoo (`dual_bench.engines.pymoo_engine`)

```python
ToPymoo(formulation)                  # -> pymoo Problem
EpsilonCallback(formulation, nfe_max) # -> pymoo Callback
```

### Platypus (`dual_bench.engines.platypus_engine`)

```python
ToPlatypus(formulation)                  # -> Platypus Problem
EpsilonUpdater(formulation, nfe_max)     # hook for run loop
```

### Solution dataclass (`dual_bench.engines.base`)

```python
Solution(x, objectives, constraints=[], feasible=True)
```
