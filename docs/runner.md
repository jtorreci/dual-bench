# Experiment runner

The `dual_bench.runner` module provides an engine-agnostic factorial
experiment runner with automatic pymoo/Platypus detection and SQLite
persistence.

## Basic usage

```python
from dual_bench import Spring, PressureVessel
from dual_bench.runner import ExperimentConfig, run_experiment

# Define algorithm factories (receive an engine-native problem)
def nsga2_factory(problem):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    return NSGA2(pop_size=100)

config = ExperimentConfig(
    problems=[Spring(), PressureVessel()],
    formulations=["2OB", "2OB-REST"],
    algorithms=[("NSGA-II", nsga2_factory)],
    n_reps=5,
    nfe=25_000,
)

results = run_experiment(config, output_dir="results")
print(f"Completed {len(results)} runs")
```

## Single run

For ad-hoc runs without a full factorial design:

```python
from dual_bench import Spring
from dual_bench.runner import run_single

result = run_single(
    Spring(), "2OB-REST",
    algorithm_factory=nsga2_factory,
    nfe=25_000,
)
print(result.objectives.shape)  # (n_solutions, 2)
print(f"Elapsed: {result.elapsed_seconds:.1f}s")
```

## SQLite persistence

Each run is saved as a SQLite file containing a `run_meta` table (problem,
formulation, algorithm, rep, nfe, elapsed) and a `solutions` table with
objective, constraint, and variable columns.

```python
from dual_bench.runner import save_run, load_run

# Save manually
save_run(result, "my_run.db")

# Load back
loaded = load_run("my_run.db")
print(loaded.problem, loaded.formulation, loaded.objectives.shape)
```

## Available formulations

The runner resolves formulation names automatically:

- Single-objective problems: `"MO"`, `"2OB"`, `"FOB"`, `"2OB-REST"`, `"FOB-REST"`
- CMOP problems: `"Standard"`, `"Envelope"`, `"Full"`

```python
from dual_bench.runner import available_formulations
from dual_bench import Spring, BNH

available_formulations(Spring())  # ['MO', '2OB', 'FOB', '2OB-REST', 'FOB-REST']
available_formulations(BNH())     # ['Standard', 'Envelope', 'Full']
```
