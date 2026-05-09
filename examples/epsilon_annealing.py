#!/usr/bin/env python3
"""Demonstrate epsilon-annealing schedules and CMOP formulation progress."""

from dual_bench.schedules import epsilon_pow, epsilon_lin, epsilon_exp
from dual_bench.problems.cmop import BNH
from dual_bench.formulations import CMOPEnvelope

# Compare the three schedules
T = 30000
print("--- Epsilon schedules (T=30000) ---")
print(f"{'t':>8s}  {'pow':>8s}  {'lin':>8s}  {'exp':>8s}")
for t in [0, 3000, 6000, 10000, 15000, 20000, 25000, 28000, 30000]:
    ep = epsilon_pow(t, T)
    el = epsilon_lin(t, T)
    ee = epsilon_exp(t, T)
    print(f"{t:8d}  {ep:8.5f}  {el:8.5f}  {ee:8.5f}")

# Use with a CMOP formulation
print("\n--- CMOPEnvelope with progress updates ---")
bnh = BNH()
form = CMOPEnvelope(bnh, eps_max=0.3, cp=5)

import numpy as np
x = (bnh.xl + bnh.xu) / 2.0

for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
    t = int(progress * T)
    form.set_progress(t, T)
    objs, cons = form.evaluate(x)
    print(f"  progress={progress:.0%}  eps={form.epsilon:.5f}"
          f"  constraint_margin={max(cons):.4f}")
