# -*- coding: utf-8 -*-
"""dual-bench -- Benchmark library for constrained optimization via demand-to-capacity ratios.

Provides a uniform interface for benchmarking constrained optimisation
algorithms using demand-to-capacity (d/c) ratio formulations.  Includes
single-objective engineering problems, constrained multi-objective problems,
epsilon-annealing schedules, and engine adapters for pymoo and Platypus.
"""

__version__ = "1.1.0"

# --- Problems ---
# --- Engine result type ---
from dual_bench.engines.base import Solution

# --- Formulations ---
from dual_bench.formulations import (
    BiobjFormulation,
    BiobjRestFormulation,
    CMOPEnvelope,
    CMOPFull,
    CMOPStandard,
    ManyobjFormulation,
    ManyobjRestFormulation,
    MOFormulation,
)
from dual_bench.problems import (
    ALL_PROBLEMS,
    CantileverBeam,
    CMOProblem,
    DualProblem,
    PressureVessel,
    SpeedReducer,
    Spring,
    WeldedBeam,
)
from dual_bench.problems.cmop import CMOP_REGISTRY

# --- Suites ---
from dual_bench.problems.suites import list_suites, suite

# --- Schedules ---
from dual_bench.schedules import epsilon_exp, epsilon_lin, epsilon_pow

# --- References (lazy import to keep startup fast) ---
from dual_bench import references

# --- Runner ---
from dual_bench.runner import ExperimentConfig, RunResult, run_experiment, run_single

# --- Analysis (lazy -- requires scipy) ---
from dual_bench import analysis

__all__ = [
    # Version
    "__version__",
    # Base classes
    "DualProblem", "CMOProblem",
    # SO problems
    "Spring", "PressureVessel", "WeldedBeam", "SpeedReducer", "CantileverBeam",
    "ALL_PROBLEMS",
    # CMOP registry
    "CMOP_REGISTRY",
    # SO formulations
    "MOFormulation", "BiobjFormulation", "ManyobjFormulation",
    "BiobjRestFormulation", "ManyobjRestFormulation",
    # CMOP formulations
    "CMOPEnvelope", "CMOPFull", "CMOPStandard",
    # Schedules
    "epsilon_pow", "epsilon_lin", "epsilon_exp",
    # Suites
    "suite", "list_suites",
    # Engine types
    "Solution",
    # References
    "references",
    # Runner
    "ExperimentConfig", "RunResult", "run_single", "run_experiment",
    # Analysis
    "analysis",
]
