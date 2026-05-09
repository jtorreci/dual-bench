#!/usr/bin/env python3
"""Basic usage of dual-bench: create problems, wrap in formulations, evaluate."""

import dual_bench

# List available suites
print("Available suites:", dual_bench.list_suites())

# Load the 5 engineering benchmarks
print("\n--- Engineering benchmarks ---")
for problem in dual_bench.suite("engineering5"):
    x = problem.midpoint()
    f, ratios = problem.evaluate(x)
    feasible = problem.is_feasible(x)
    print(f"  {problem.name:20s}  f={f:12.4f}  max_d={max(ratios):.4f}  feasible={feasible}")

# Wrap Spring in different formulations
print("\n--- Spring formulations ---")
spring = dual_bench.Spring()
x = spring.midpoint()

formulations = {
    "MO":       spring.to_mo(),
    "2OB":      spring.to_2ob(),
    "FOB":      spring.to_fob(),
    "2OB-REST": spring.to_2ob_rest(),
    "FOB-REST": spring.to_fob_rest(),
}

for name, form in formulations.items():
    objs, cons = form.evaluate(x)
    print(f"  {name:10s}  n_obj={form.n_objectives}  n_con={form.n_constraints}"
          f"  obj[0]={objs[0]:12.4f}")

# CMOP example
print("\n--- CMOP problems (classic5) ---")
for problem in dual_bench.suite("classic5"):
    import numpy as np
    x = (problem.xl + problem.xu) / 2.0
    F, G = problem.evaluate(x)
    print(f"  {problem.name:10s}  F={F}  feasible={problem.is_feasible(x)}")

print("\nDone!")
