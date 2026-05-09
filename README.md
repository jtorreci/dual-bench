# dual-bench

Benchmark library for constrained optimization via demand-to-capacity ratios.

`dual-bench` provides a uniform interface for benchmarking constrained
optimisation algorithms using demand-to-capacity (d/c) ratio formulations.
It includes 40+ built-in problems across 8 suites, 8 formulation wrappers,
epsilon-annealing schedules, and engine adapters for pymoo and Platypus.

## Installation

```bash
pip install dual-bench              # core (numpy only)
pip install dual-bench[pymoo]       # + pymoo engine adapter
pip install dual-bench[platypus]    # + Platypus engine adapter
pip install dual-bench[all]         # pymoo + Platypus
pip install dual-bench[dev]         # + pytest, ruff
```

## Quick start

```python
import dual_bench

# Create a problem and wrap it in a bi-objective formulation
spring = dual_bench.Spring()
form = spring.to_2ob_rest()  # 2 objectives + 4 constraints

# Evaluate at the midpoint
objectives, constraints = form.evaluate(spring.midpoint())

# List available suites
print(dual_bench.list_suites())
# ['classic5', 'dascmop', 'dtlz', 'engineering5', 'lircmop', 'mw', 'wfg', 'zdt']

# Load a full suite
problems = dual_bench.suite("engineering5")
```

### Running with pymoo

```python
from dual_bench import Spring
from dual_bench.formulations import CMOPEnvelope
from dual_bench.engines.pymoo_engine import ToPymoo, EpsilonCallback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

form = Spring().to_2ob_rest()
result = minimize(
    ToPymoo(form), NSGA2(pop_size=100),
    ('n_eval', 30000),
)
```

### Running with Platypus

```python
from dual_bench.problems.cmop import BNH
from dual_bench.formulations import CMOPEnvelope
from dual_bench.engines.platypus_engine import ToPlatypus, EpsilonUpdater
from platypus import NSGAII

form = CMOPEnvelope(BNH())
algo = NSGAII(ToPlatypus(form), population_size=100)
updater = EpsilonUpdater(form, 30000)
for gen in range(300):
    algo.step()
    updater.update(algo)
```

## Available suites

| Suite | Problems | Type | Engine required |
|-------|----------|------|-----------------|
| `engineering5` | 5 | Single-objective d/c | - |
| `classic5` | 5 | Constrained MO | - |
| `lircmop` | 14 | Constrained MO (large infeasible) | - |
| `mw` | 14 | Constrained MO | pymoo |
| `dascmop` | 9 | Constrained MO | pymoo |
| `zdt` | 6 | Unconstrained MO | pymoo |
| `dtlz` | 7 | Unconstrained MO | pymoo |
| `wfg` | 9 | Unconstrained MO | pymoo |

## Documentation

- [Quick start guide](docs/quickstart.md)
- [API reference](docs/api.md)
- [Adding new problems](docs/adding_problems.md)

## Citation

If you use `dual-bench` in your research, please cite:

```bibtex
@article{dual-bench,
  title  = {dual-bench: A Benchmark Library for Constrained Optimization
            via Demand-to-Capacity Ratios},
  author = {Torre-Cifuentes, Joaquin},
  year   = {2026},
  journal = {Journal of Open Source Software},
}
```

## License

MIT
