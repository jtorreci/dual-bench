# Changelog

## 1.1.0 (2026-03-27)

New modules for reproducibility and post-processing.

- Reference data module (`dual_bench.references`): known optima for all
  5 engineering benchmarks and summary statistics from 3,000 Paper A runs
  (5 problems x 5 formulations x 6 algorithms x 20 reps)
- Experiment runner (`dual_bench.runner`): engine-agnostic factorial
  experiment runner with SQLite persistence, automatic pymoo/Platypus
  detection, and configurable repetitions
- Analysis pipeline (`dual_bench.analysis`): non-parametric hypothesis
  tests (Friedman, Nemenyi, Kruskal-Wallis), effect sizes (Cliff's delta),
  performance metrics (2-D hypervolume, IGD, delta-f*, feasibility rate,
  spread), and LaTeX booktabs table generation
- Added `scipy>=1.7` as optional `[analysis]` dependency

## 1.0.0 (2026-03-11)

Initial public release.

- 5 single-objective engineering benchmarks (Spring, PressureVessel, WeldedBeam, SpeedReducer, CantileverBeam)
- 5 classic CMOPs (BNH, SRN, TNK, OSY, CONSTR)
- 14 LIRCMOP problems
- pymoo bridge for MW (14), DAS-CMOP (9), ZDT (6), DTLZ (7), WFG (9)
- 5 single-objective formulations (MO, 2OB, FOB, 2OB-REST, FOB-REST)
- 3 CMOP formulations (Standard, Envelope, Full)
- 3 epsilon-annealing schedules (power-law, linear, exponential)
- Engine adapters for pymoo and Platypus
- Suite registry with 8 built-in suites
