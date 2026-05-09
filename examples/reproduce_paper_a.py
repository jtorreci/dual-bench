#!/usr/bin/env python3
"""Reproduce a subset of Paper A results and compare against reference data.

Runs a small validation experiment (2 problems x 3 formulations x 1 algorithm
x 3 reps = 18 runs) and checks that the measured delta-f* metric falls within
the expected range reported in Paper A (Torre-Cifuentes, 2026).

Paper A used 5 problems x 5 formulations x 6 algorithms x 20 reps = 3,000 runs
with Platypus, pop_size=100, NFE=25,000.  This script uses pymoo NSGA-II with
the same NFE and pop_size but only 3 reps, so higher variance is expected.

Metrics
-------
- **delta-f***: ``(f_best - f*) / |f*|`` among feasible solutions.
- **HV**: 2-D hypervolume on the (f, max_d) front (informational only).

Usage::

    python examples/reproduce_paper_a.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize

import dual_bench
from dual_bench import Spring, WeldedBeam
from dual_bench.analysis import delta_f_star, hypervolume_2d
from dual_bench.engines.pymoo_engine import EpsilonCallback, ToPymoo
from dual_bench.references import optimum, paper_a_results
from dual_bench.runner import RunResult, apply_formulation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROBLEMS = [Spring(), WeldedBeam()]
FORMULATIONS = ["MO", "2OB-REST", "FOB-REST"]
N_REPS = 3
NFE = 25_000
POP_SIZE = 100
HV_REF = np.array([1.1, 1.1])
SIGMA_THRESHOLD = 2  # OK if within mean +/- 2*std of Paper A


# ---------------------------------------------------------------------------
# Run function
# ---------------------------------------------------------------------------

def _run_one(problem, formulation_name: str, nfe: int, rep: int) -> RunResult:
    """Execute one pymoo NSGA-II run and return a RunResult."""
    formulation = apply_formulation(problem, formulation_name)
    pymoo_prob = ToPymoo(formulation)
    algo = NSGA2(pop_size=POP_SIZE)

    kwargs = dict(verbose=False)
    if hasattr(formulation, "set_progress"):
        kwargs["callback"] = EpsilonCallback(formulation, nfe)

    t0 = time.perf_counter()
    res = pymoo_minimize(pymoo_prob, algo, ("n_eval", nfe), **kwargs)
    elapsed = time.perf_counter() - t0

    # Prefer pymoo's result (feasible set); fall back to full population
    if res.F is not None and len(res.F) > 0:
        F = np.atleast_2d(res.F)
        G = np.atleast_2d(res.G) if res.G is not None else None
    else:
        pop = res.algorithm.pop
        F = np.atleast_2d(pop.get("F"))
        G = pop.get("G")
        if G is not None:
            G = np.atleast_2d(G)

    nfe_actual = int(res.algorithm.evaluator.n_eval) if res.algorithm else nfe

    return RunResult(
        problem=getattr(problem, "name", type(problem).__name__),
        formulation=formulation_name,
        algorithm="NSGA-II",
        rep=rep,
        objectives=F,
        constraints=G,
        variables=None,
        elapsed_seconds=elapsed,
        nfe=nfe_actual,
    )


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    """Metrics extracted from a single run."""
    problem: str
    formulation: str
    rep: int
    hv: Optional[float]
    df: Optional[float]


def _feasible_mask(result: RunResult) -> np.ndarray:
    """Boolean mask: feasible when all constraints <= 0 (G <= 0 convention)."""
    if result.constraints is not None and result.constraints.size > 0:
        return np.all(result.constraints <= 0.0, axis=1)
    return np.ones(len(result.objectives), dtype=bool)


def _build_2d_front(result: RunResult) -> np.ndarray:
    """Build an (f, max_d) array from any formulation's result."""
    n = len(result.objectives)
    if n == 0:
        return np.empty((0, 2))

    f_vals = result.objectives[:, 0]
    n_obj = result.objectives.shape[1]

    if n_obj == 2:
        # 2OB-REST: objectives = [f, max_d]
        max_d = result.objectives[:, 1]
    elif result.constraints is not None and result.constraints.size > 0:
        # MO or FOB-REST: constraints are d_i - 1, so d_i = constraint + 1
        max_d = np.max(result.constraints + 1.0, axis=1)
    else:
        max_d = np.zeros(n)

    return np.column_stack([f_vals, max_d])


def extract_metrics(result: RunResult) -> RunMetrics:
    """Compute HV and delta-f* from a RunResult."""
    f_star = optimum(result.problem)

    front_2d = _build_2d_front(result)
    hv = hypervolume_2d(front_2d, HV_REF) if len(front_2d) > 0 else None

    df = None
    feasible = _feasible_mask(result)
    if feasible.any():
        f_best = float(result.objectives[feasible, 0].min())
        df = delta_f_star(f_best, f_star)

    return RunMetrics(
        problem=result.problem,
        formulation=result.formulation,
        rep=result.rep,
        hv=hv,
        df=df,
    )


# ---------------------------------------------------------------------------
# Comparison with Paper A
# ---------------------------------------------------------------------------

@dataclass
class ComparisonRow:
    problem: str
    formulation: str
    metric: str
    run_mean: float
    run_std: float
    run_n: int
    ref_mean: float
    ref_std: float
    ref_n: int
    status: str


def _within_tolerance(run_mean, ref_mean, ref_std, k=SIGMA_THRESHOLD):
    return abs(run_mean - ref_mean) <= k * ref_std


def build_comparison(all_metrics: List[RunMetrics]) -> List[ComparisonRow]:
    """Compare aggregated metrics against Paper A per-algorithm NSGAII data."""
    rows: List[ComparisonRow] = []

    groups: dict[tuple, List[RunMetrics]] = {}
    for m in all_metrics:
        groups.setdefault((m.problem, m.formulation), []).append(m)

    for (prob, form), metrics in sorted(groups.items()):
        refs = paper_a_results(problem=prob, formulation=form, algorithm="NSGAII")
        if not refs:
            continue
        ref = refs[0]

        df_vals = [m.df for m in metrics if m.df is not None]
        if df_vals and ref.get("df_mean") is not None:
            df_mean = float(np.mean(df_vals))
            df_std = float(np.std(df_vals, ddof=1)) if len(df_vals) > 1 else 0.0
            ok = _within_tolerance(df_mean, ref["df_mean"], ref["df_std"])
            rows.append(ComparisonRow(
                problem=prob, formulation=form, metric="df*",
                run_mean=df_mean, run_std=df_std, run_n=len(df_vals),
                ref_mean=ref["df_mean"], ref_std=ref["df_std"],
                ref_n=ref["n"], status="OK" if ok else "WARN",
            ))

        hv_vals = [m.hv for m in metrics if m.hv is not None]
        if hv_vals:
            hv_mean = float(np.mean(hv_vals))
            hv_std = float(np.std(hv_vals, ddof=1)) if len(hv_vals) > 1 else 0.0
            rows.append(ComparisonRow(
                problem=prob, formulation=form, metric="HV",
                run_mean=hv_mean, run_std=hv_std, run_n=len(hv_vals),
                ref_mean=ref["hv_mean"], ref_std=ref["hv_std"],
                ref_n=ref["n"], status="info",
            ))

    return rows


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def print_table(rows: List[ComparisonRow], elapsed: float) -> None:
    print()
    print(f"=== dual-bench v{dual_bench.__version__} -- Paper A Validation ===")
    print()

    header = (f"{'Problem':<16s} | {'Form':<9s} | {'Metric':<6s} | "
              f"{'This run (mean +/- std)':<24s} | "
              f"{'Paper A (NSGAII, n=20)':<24s} | Status")
    sep = "-" * len(header)
    print(header)
    print(sep)

    for r in rows:
        run_str = f"{r.run_mean:.4f} +/- {r.run_std:.4f}"
        ref_str = f"{r.ref_mean:.4f} +/- {r.ref_std:.4f}"
        print(f"{r.problem:<16s} | {r.formulation:<9s} | {r.metric:<6s} | "
              f"{run_str:<24s} | {ref_str:<24s} | {r.status:>4s}")

    print(sep)

    df_rows = [r for r in rows if r.metric == "df*"]
    n_ok = sum(1 for r in df_rows if r.status == "OK")
    n_total = len(df_rows)
    verdict = "PASS" if n_ok == n_total else ("PARTIAL" if n_ok > 0 else "FAIL")

    print()
    print(f"Validation (df*): {n_ok}/{n_total} metrics within "
          f"{SIGMA_THRESHOLD} sigma of Paper A -- {verdict}")
    print()
    print(f"  Paper A ref:  Platypus NSGAII, n=20 reps, NFE=25,000, pop_size=100")
    print(f"  This run:     pymoo NSGA-II,   n={N_REPS} reps,  NFE={NFE:,}, pop_size={POP_SIZE}")
    print(f"  Wall time:    {elapsed:.1f}s")

    hv_rows = [r for r in rows if r.metric == "HV"]
    if hv_rows:
        print()
        print("  HV values are shown for information only.  Paper A computes HV")
        print("  with a normalisation that is not replicated here, so absolute")
        print("  values differ.  Use df* for the primary validation check.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n_total = len(PROBLEMS) * len(FORMULATIONS) * N_REPS
    print(f"dual-bench v{dual_bench.__version__} -- Paper A Validation")
    print(f"Running {len(PROBLEMS)} problems x {len(FORMULATIONS)} formulations "
          f"x {N_REPS} reps = {n_total} runs  (NFE={NFE:,}, pop_size={POP_SIZE})")
    print()

    all_metrics: List[RunMetrics] = []
    t_start = time.perf_counter()
    run_idx = 0

    for prob in PROBLEMS:
        for form_name in FORMULATIONS:
            for rep in range(1, N_REPS + 1):
                run_idx += 1
                label = f"[{run_idx}/{n_total}] {prob.name} / {form_name} / rep {rep}"
                print(f"  {label} ...", end="", flush=True)

                result = _run_one(prob, form_name, NFE, rep)
                metrics = extract_metrics(result)
                all_metrics.append(metrics)

                hv_s = f"HV={metrics.hv:.4f}" if metrics.hv is not None else "HV=n/a"
                df_s = f"df*={metrics.df:.4f}" if metrics.df is not None else "df*=n/a"
                print(f"  {hv_s}  {df_s}  ({result.elapsed_seconds:.1f}s)")

    elapsed = time.perf_counter() - t_start
    comparison = build_comparison(all_metrics)
    print_table(comparison, elapsed)


if __name__ == "__main__":
    main()
