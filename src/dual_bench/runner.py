# -*- coding: utf-8 -*-
"""Engine-agnostic experiment runner for dual-bench.

Supports factorial designs (problems x formulations x algorithms x reps)
with automatic engine detection (pymoo / Platypus) and SQLite persistence.

Example::

    from dual_bench import Spring, PressureVessel
    from dual_bench.runner import ExperimentConfig, run_experiment

    config = ExperimentConfig(
        problems=[Spring(), PressureVessel()],
        formulations=["2OB", "2OB-REST"],
        algorithms=[("NSGA-II", my_nsga2_factory)],
    )
    results = run_experiment(config, output_dir="results")
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from dual_bench.formulations import (
    BiobjFormulation, BiobjRestFormulation,
    CMOPEnvelope, CMOPFull, CMOPStandard,
    ManyobjFormulation, ManyobjRestFormulation, MOFormulation,
)
from dual_bench.problems.base import CMOProblem, DualProblem

# -- Formulation registries --------------------------------------------------

_SO_FORMULATIONS: dict[str, Callable] = {
    "MO": lambda p: MOFormulation(p),
    "2OB": lambda p: BiobjFormulation(p),
    "FOB": lambda p: ManyobjFormulation(p),
    "2OB-REST": lambda p: BiobjRestFormulation(p),
    "FOB-REST": lambda p: ManyobjRestFormulation(p),
}

_CMOP_FORMULATIONS: dict[str, Callable] = {
    "Standard": lambda p: CMOPStandard(p),
    "Envelope": lambda p: CMOPEnvelope(p),
    "Full": lambda p: CMOPFull(p),
}


def available_formulations(problem: DualProblem | CMOProblem) -> list[str]:
    """Return the formulation names valid for *problem*."""
    if isinstance(problem, DualProblem):
        return list(_SO_FORMULATIONS)
    return list(_CMOP_FORMULATIONS)


def apply_formulation(problem: DualProblem | CMOProblem, name: str):
    """Wrap *problem* in the named formulation and return the wrapper."""
    if isinstance(problem, DualProblem):
        if name not in _SO_FORMULATIONS:
            raise ValueError(
                f"Unknown SO formulation {name!r}. "
                f"Choose from {list(_SO_FORMULATIONS)}")
        return _SO_FORMULATIONS[name](problem)
    if isinstance(problem, CMOProblem):
        if name not in _CMOP_FORMULATIONS:
            raise ValueError(
                f"Unknown CMOP formulation {name!r}. "
                f"Choose from {list(_CMOP_FORMULATIONS)}")
        return _CMOP_FORMULATIONS[name](problem)
    raise TypeError(f"Unsupported problem type: {type(problem)}")


# -- Data classes -------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a single optimisation run."""
    problem: str
    formulation: str
    algorithm: str
    rep: int
    objectives: np.ndarray                    # (n_solutions, n_obj)
    constraints: Optional[np.ndarray] = None  # (n_solutions, n_con) or None
    variables: Optional[np.ndarray] = None    # (n_solutions, n_var) or None
    elapsed_seconds: float = 0.0
    nfe: int = 0


@dataclass
class ExperimentConfig:
    """Factorial experimental design specification.

    *algorithms* is a list of ``(name, factory)`` tuples where *factory*
    receives an engine-native problem and returns an algorithm instance.
    """
    problems: List[Any] = field(default_factory=list)
    formulations: List[str] = field(default_factory=list)
    algorithms: List[Tuple[str, Callable]] = field(default_factory=list)
    n_reps: int = 20
    nfe: int = 25_000

    def enumerate_jobs(self) -> list[dict]:
        """Return every (problem, formulation, algorithm, rep) combination."""
        jobs = []
        for prob, form_name, (alg_name, alg_factory), rep in product(
            self.problems, self.formulations, self.algorithms,
            range(1, self.n_reps + 1),
        ):
            jobs.append(dict(
                problem=prob, formulation=form_name,
                algorithm_name=alg_name, algorithm_factory=alg_factory,
                rep=rep,
            ))
        return jobs


# -- Engine detection / dispatch ----------------------------------------------

def _detect_engine(algorithm_obj) -> str:
    """Return ``'pymoo'`` or ``'platypus'`` based on the algorithm object."""
    module = type(algorithm_obj).__module__ or ""
    if "pymoo" in module:
        return "pymoo"
    if "platypus" in module:
        return "platypus"
    raise RuntimeError(
        f"Cannot detect engine for {type(algorithm_obj).__qualname__} "
        f"(module={module!r}). Expected pymoo or Platypus.")


def _create_algorithm(formulation, algorithm_factory: Callable):
    """Wrap *formulation* in the right engine adapter and call the factory."""
    # Try pymoo first, then Platypus.
    for engine_name, import_path, cls_name in [
        ("pymoo", "dual_bench.engines.pymoo_engine", "ToPymoo"),
        ("platypus", "dual_bench.engines.platypus_engine", "ToPlatypus"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(import_path)
            adapter_cls = getattr(mod, cls_name)
            native_prob = adapter_cls(formulation)
            algo = algorithm_factory(native_prob)
            if engine_name in (type(algo).__module__ or ""):
                return algo
        except Exception:
            continue

    raise RuntimeError(
        "algorithm_factory did not return a recognised pymoo or Platypus "
        "algorithm for any engine adapter.")


def _run_pymoo(formulation, algorithm_obj, nfe: int):
    """Execute a pymoo run and return ``(F, G, X, nfe_actual)``."""
    from pymoo.optimize import minimize as pymoo_minimize
    from dual_bench.engines.pymoo_engine import EpsilonCallback, ToPymoo

    pymoo_prob = ToPymoo(formulation)
    kwargs: dict[str, Any] = dict(verbose=False)
    if hasattr(formulation, "set_progress"):
        kwargs["callback"] = EpsilonCallback(formulation, nfe)

    res = pymoo_minimize(pymoo_prob, algorithm_obj, ("n_eval", nfe), **kwargs)

    F = res.F if res.F is not None else np.empty((0, formulation.n_objectives))
    G = res.G if res.G is not None else None
    X = res.X if res.X is not None else None
    nfe_actual = res.algorithm.evaluator.n_eval if res.algorithm else nfe
    return F, G, X, int(nfe_actual)


def _run_platypus(formulation, algorithm_obj, nfe: int):
    """Execute a Platypus run and return ``(F, G, X, nfe_actual)``."""
    from dual_bench.engines.platypus_engine import EpsilonUpdater

    updater = None
    if hasattr(formulation, "set_progress"):
        updater = EpsilonUpdater(formulation, nfe)

    while algorithm_obj.nfe < nfe:
        algorithm_obj.step()
        if updater is not None:
            updater.update(algorithm_obj)

    solutions = algorithm_obj.result
    if not solutions:
        return np.empty((0, formulation.n_objectives)), None, None, algorithm_obj.nfe

    n_obj = formulation.n_objectives
    n_con = formulation.n_constraints
    n_var = len(formulation.var_bounds)

    F = np.array([s.objectives[:n_obj] for s in solutions])
    G = np.array([s.constraints[:n_con] for s in solutions]) if n_con > 0 else None
    X = np.array([[s.variables[i] for i in range(n_var)] for s in solutions])
    return F, G, X, int(algorithm_obj.nfe)


# -- Public API: single run ---------------------------------------------------

def run_single(
    problem: DualProblem | CMOProblem,
    formulation_name: str,
    algorithm_factory: Callable,
    nfe: int,
) -> RunResult:
    """Execute one optimisation run and return a :class:`RunResult`."""
    formulation = apply_formulation(problem, formulation_name)
    algo_obj = _create_algorithm(formulation, algorithm_factory)
    engine = _detect_engine(algo_obj)

    t0 = time.perf_counter()
    if engine == "pymoo":
        F, G, X, nfe_actual = _run_pymoo(formulation, algo_obj, nfe)
    else:
        F, G, X, nfe_actual = _run_platypus(formulation, algo_obj, nfe)
    elapsed = time.perf_counter() - t0

    return RunResult(
        problem=getattr(problem, "name", type(problem).__name__),
        formulation=formulation_name,
        algorithm="",   # filled by run_experiment
        rep=0,          # filled by run_experiment
        objectives=F, constraints=G, variables=X,
        elapsed_seconds=elapsed, nfe=nfe_actual,
    )


# -- Public API: full experiment ----------------------------------------------

def run_experiment(
    config: ExperimentConfig,
    output_dir: Optional[str] = None,
    callback: Optional[Callable[[RunResult], Any]] = None,
) -> list[RunResult]:
    """Run a full-factorial experiment defined by *config*.

    Optionally persists each run to *output_dir* as a SQLite ``.db`` file
    and calls *callback(result)* after each run completes.
    """
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for job in config.enumerate_jobs():
        result = run_single(
            job["problem"], job["formulation"],
            job["algorithm_factory"], config.nfe)
        result.algorithm = job["algorithm_name"]
        result.rep = job["rep"]
        results.append(result)

        if output_dir is not None:
            db_name = (f"{result.problem}_{result.formulation}_"
                       f"{result.algorithm}_{result.rep:02d}.db")
            save_run(result, str(Path(output_dir) / db_name))
        if callback is not None:
            callback(result)

    return results


# -- SQLite persistence -------------------------------------------------------

def save_run(result: RunResult, path: str) -> None:
    """Persist a :class:`RunResult` to a SQLite file at *path*."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        _write_run(conn, result)
    finally:
        conn.close()


def load_run(path: str) -> RunResult:
    """Load a :class:`RunResult` from a SQLite file at *path*."""
    conn = sqlite3.connect(path)
    try:
        return _read_run(conn)
    finally:
        conn.close()


def _write_run(conn: sqlite3.Connection, result: RunResult) -> None:
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS run_meta ("
        "problem TEXT, formulation TEXT, algorithm TEXT,"
        "rep INTEGER, nfe INTEGER, elapsed REAL)")
    cur.execute(
        "INSERT INTO run_meta VALUES (?,?,?,?,?,?)",
        (result.problem, result.formulation, result.algorithm,
         result.rep, result.nfe, result.elapsed_seconds))

    # Dynamic column layout for solutions table.
    n_sol, n_obj = result.objectives.shape if result.objectives.ndim == 2 else (0, 0)
    n_con = (result.constraints.shape[1]
             if result.constraints is not None and result.constraints.ndim == 2 else 0)
    n_var = (result.variables.shape[1]
             if result.variables is not None and result.variables.ndim == 2 else 0)

    col_defs = ["id INTEGER PRIMARY KEY"]
    col_names: list[str] = []
    for prefix, count in [("obj", n_obj), ("con", n_con), ("var", n_var)]:
        for i in range(count):
            col_defs.append(f"{prefix}_{i} REAL")
            col_names.append(f"{prefix}_{i}")

    cur.execute(f"CREATE TABLE IF NOT EXISTS solutions ({', '.join(col_defs)})")

    if n_sol > 0 and col_names:
        placeholders = ", ".join(["?"] * len(col_names))
        sql = f"INSERT INTO solutions ({', '.join(col_names)}) VALUES ({placeholders})"
        for idx in range(n_sol):
            vals = list(result.objectives[idx])
            if result.constraints is not None and n_con > 0:
                vals.extend(result.constraints[idx])
            if result.variables is not None and n_var > 0:
                vals.extend(result.variables[idx])
            cur.execute(sql, vals)

    conn.commit()


def _read_run(conn: sqlite3.Connection) -> RunResult:
    cur = conn.cursor()

    try:
        cur.execute(
            "SELECT problem, formulation, algorithm, rep, nfe, elapsed "
            "FROM run_meta")
    except sqlite3.OperationalError as exc:
        raise ValueError(f"No run_meta row found in database ({exc})") from exc
    row = cur.fetchone()
    if row is None:
        raise ValueError("No run_meta row found in database.")
    problem, formulation, algorithm, rep, nfe, elapsed = row

    # Discover column layout.
    cur.execute("PRAGMA table_info(solutions)")
    columns = [info[1] for info in cur.fetchall()]

    def _sort_cols(prefix):
        cols = [c for c in columns if c.startswith(prefix + "_")]
        cols.sort(key=lambda c: int(c.rsplit("_", 1)[1]))
        return cols

    obj_cols, con_cols, var_cols = _sort_cols("obj"), _sort_cols("con"), _sort_cols("var")
    select_cols = obj_cols + con_cols + var_cols

    if not select_cols:
        return RunResult(
            problem=problem, formulation=formulation, algorithm=algorithm,
            rep=rep, objectives=np.empty((0, 0)),
            elapsed_seconds=elapsed, nfe=nfe)

    cur.execute(f"SELECT {', '.join(select_cols)} FROM solutions")
    rows = cur.fetchall()
    n_obj, n_con, n_var = len(obj_cols), len(con_cols), len(var_cols)

    if not rows:
        return RunResult(
            problem=problem, formulation=formulation, algorithm=algorithm,
            rep=rep, objectives=np.empty((0, n_obj)),
            constraints=np.empty((0, n_con)) if n_con else None,
            variables=np.empty((0, n_var)) if n_var else None,
            elapsed_seconds=elapsed, nfe=nfe)

    data = np.array(rows, dtype=float)
    objectives = data[:, :n_obj]
    idx = n_obj
    constraints = data[:, idx:idx + n_con] if n_con else None
    idx += n_con
    variables = data[:, idx:idx + n_var] if n_var else None

    return RunResult(
        problem=problem, formulation=formulation, algorithm=algorithm,
        rep=rep, objectives=objectives, constraints=constraints,
        variables=variables, elapsed_seconds=elapsed, nfe=nfe)
