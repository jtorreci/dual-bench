# Reference data

The `dual_bench.references` module provides known optimal values and summary
statistics from 3,000 optimisation runs published in Paper A
(Torre-Cifuentes, 2026).

## Known optima

```python
import dual_bench.references as ref

# List all problems with known optima
ref.list_problems()
# ['CantileverBeam', 'PressureVessel', 'SpeedReducer', 'Spring', 'WeldedBeam']

# Get the best-known objective value
ref.optimum("Spring")  # 0.012665

# Get full metadata (n_vars, n_ratios, known_x, source)
info = ref.info("Spring")
print(info.known_x)  # [0.05169, 0.35675, 11.287126]
```

## Paper A results

The experimental design covers 5 problems, 5 formulations, 6 algorithms,
and 20 repetitions per cell (3,000 runs total; NFE = 25,000; Platypus engine).

```python
# Per-formulation summary (pooled across 6 algorithms, n=120)
rows = ref.paper_a_results("Spring", "2OB-REST")
for r in rows:
    print(f"HV = {r['hv_mean']:.4f}, df* = {r['df_mean']:.4f}")

# Per-algorithm detail (n=20 per cell)
rows = ref.paper_a_results("Spring", "2OB-REST", algorithm="NSGAII")
print(rows[0]["hv_mean"])  # 0.7537

# Retrieve experimental setup metadata
setup = ref.paper_a_setup()
print(setup["total_runs"])  # 3000
```

## Comparing your results

```python
import numpy as np
from dual_bench import Spring
from dual_bench.analysis import hypervolume_2d
import dual_bench.references as ref

# Run your optimisation and collect the (f, max_d) Pareto front...
my_front = np.array([[0.013, 0.95], [0.014, 0.80]])
my_hv = hypervolume_2d(my_front, ref=np.array([1.1, 1.1]))

# Compare against the Paper A baseline
baseline = ref.paper_a_results("Spring", "2OB-REST")[0]
print(f"My HV: {my_hv:.4f}, Paper A HV: {baseline['hv_mean']:.4f}")
```
