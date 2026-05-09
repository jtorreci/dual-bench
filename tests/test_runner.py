# -*- coding: utf-8 -*-
"""Tests for the experiment runner module.

Unit tests exercise configuration, formulation mapping, job enumeration,
and SQLite round-trip persistence.  No actual optimisation is run (those
would be marked ``@pytest.mark.slow``).
"""

import numpy as np
import pytest

from dual_bench.problems import Spring, PressureVessel
from dual_bench.problems.cmop import BNH
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
from dual_bench.runner import (
    ExperimentConfig,
    RunResult,
    apply_formulation,
    available_formulations,
    load_run,
    save_run,
)


# ======================================================================
# Formulation mapping
# ======================================================================

class TestApplyFormulation:
    """Test the formulation name -> wrapper mapping."""

    @pytest.mark.parametrize("name,expected_cls", [
        ("MO", MOFormulation),
        ("2OB", BiobjFormulation),
        ("FOB", ManyobjFormulation),
        ("2OB-REST", BiobjRestFormulation),
        ("FOB-REST", ManyobjRestFormulation),
    ])
    def test_so_formulations(self, name, expected_cls):
        p = Spring()
        form = apply_formulation(p, name)
        assert isinstance(form, expected_cls)

    @pytest.mark.parametrize("name,expected_cls", [
        ("Standard", CMOPStandard),
        ("Envelope", CMOPEnvelope),
        ("Full", CMOPFull),
    ])
    def test_cmop_formulations(self, name, expected_cls):
        p = BNH()
        form = apply_formulation(p, name)
        assert isinstance(form, expected_cls)

    def test_unknown_so_raises(self):
        with pytest.raises(ValueError, match="Unknown SO formulation"):
            apply_formulation(Spring(), "NoSuchForm")

    def test_unknown_cmop_raises(self):
        with pytest.raises(ValueError, match="Unknown CMOP formulation"):
            apply_formulation(BNH(), "NoSuchForm")

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported problem type"):
            apply_formulation("not_a_problem", "2OB")


class TestAvailableFormulations:

    def test_so_keys(self):
        names = available_formulations(Spring())
        assert set(names) == {"MO", "2OB", "FOB", "2OB-REST", "FOB-REST"}

    def test_cmop_keys(self):
        names = available_formulations(BNH())
        assert set(names) == {"Standard", "Envelope", "Full"}


# ======================================================================
# ExperimentConfig and job enumeration
# ======================================================================

class TestExperimentConfig:

    def test_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.n_reps == 20
        assert cfg.nfe == 25_000
        assert cfg.problems == []
        assert cfg.formulations == []
        assert cfg.algorithms == []

    def test_enumerate_empty(self):
        cfg = ExperimentConfig()
        assert cfg.enumerate_jobs() == []

    def test_enumerate_factorial_count(self):
        dummy_factory = lambda prob: None  # noqa: E731
        cfg = ExperimentConfig(
            problems=[Spring(), PressureVessel()],
            formulations=["2OB", "FOB-REST"],
            algorithms=[("A1", dummy_factory), ("A2", dummy_factory)],
            n_reps=3,
            nfe=1000,
        )
        jobs = cfg.enumerate_jobs()
        # 2 problems x 2 formulations x 2 algorithms x 3 reps = 24
        assert len(jobs) == 24

    def test_enumerate_job_fields(self):
        factory = lambda prob: None  # noqa: E731
        cfg = ExperimentConfig(
            problems=[Spring()],
            formulations=["2OB"],
            algorithms=[("MyAlgo", factory)],
            n_reps=1,
        )
        jobs = cfg.enumerate_jobs()
        assert len(jobs) == 1
        job = jobs[0]
        assert job["problem"] is cfg.problems[0]
        assert job["formulation"] == "2OB"
        assert job["algorithm_name"] == "MyAlgo"
        assert job["algorithm_factory"] is factory
        assert job["rep"] == 1

    def test_enumerate_reps_start_at_one(self):
        factory = lambda prob: None  # noqa: E731
        cfg = ExperimentConfig(
            problems=[Spring()],
            formulations=["2OB"],
            algorithms=[("A", factory)],
            n_reps=5,
        )
        reps = [j["rep"] for j in cfg.enumerate_jobs()]
        assert reps == [1, 2, 3, 4, 5]


# ======================================================================
# RunResult basics
# ======================================================================

class TestRunResult:

    def test_creation_minimal(self):
        r = RunResult(
            problem="Spring",
            formulation="2OB",
            algorithm="NSGA-II",
            rep=1,
            objectives=np.array([[1.0, 0.5], [2.0, 0.8]]),
        )
        assert r.problem == "Spring"
        assert r.constraints is None
        assert r.variables is None
        assert r.elapsed_seconds == 0.0
        assert r.nfe == 0

    def test_creation_full(self):
        r = RunResult(
            problem="BNH",
            formulation="Envelope",
            algorithm="NSGA-II",
            rep=3,
            objectives=np.ones((10, 3)),
            constraints=np.zeros((10, 2)),
            variables=np.random.rand(10, 4),
            elapsed_seconds=12.5,
            nfe=25000,
        )
        assert r.objectives.shape == (10, 3)
        assert r.constraints.shape == (10, 2)
        assert r.variables.shape == (10, 4)


# ======================================================================
# SQLite round-trip
# ======================================================================

class TestSQLitePersistence:

    def _make_result(self, n_sol=5, n_obj=2, n_con=3, n_var=4):
        return RunResult(
            problem="TestProblem",
            formulation="2OB",
            algorithm="TestAlgo",
            rep=7,
            objectives=np.random.rand(n_sol, n_obj),
            constraints=np.random.rand(n_sol, n_con) if n_con > 0 else None,
            variables=np.random.rand(n_sol, n_var) if n_var > 0 else None,
            elapsed_seconds=3.14,
            nfe=10000,
        )

    def test_roundtrip_full(self, tmp_path):
        original = self._make_result(n_sol=8, n_obj=3, n_con=2, n_var=5)
        db_path = str(tmp_path / "run.db")
        save_run(original, db_path)
        loaded = load_run(db_path)

        assert loaded.problem == original.problem
        assert loaded.formulation == original.formulation
        assert loaded.algorithm == original.algorithm
        assert loaded.rep == original.rep
        assert loaded.nfe == original.nfe
        assert loaded.elapsed_seconds == pytest.approx(original.elapsed_seconds)
        np.testing.assert_array_almost_equal(loaded.objectives, original.objectives)
        np.testing.assert_array_almost_equal(loaded.constraints, original.constraints)
        np.testing.assert_array_almost_equal(loaded.variables, original.variables)

    def test_roundtrip_no_constraints(self, tmp_path):
        original = self._make_result(n_sol=4, n_obj=2, n_con=0, n_var=3)
        db_path = str(tmp_path / "run_nc.db")
        save_run(original, db_path)
        loaded = load_run(db_path)

        assert loaded.constraints is None
        np.testing.assert_array_almost_equal(loaded.objectives, original.objectives)
        np.testing.assert_array_almost_equal(loaded.variables, original.variables)

    def test_roundtrip_no_variables(self, tmp_path):
        original = self._make_result(n_sol=3, n_obj=2, n_con=1, n_var=0)
        db_path = str(tmp_path / "run_nv.db")
        save_run(original, db_path)
        loaded = load_run(db_path)

        assert loaded.variables is None
        np.testing.assert_array_almost_equal(loaded.objectives, original.objectives)
        np.testing.assert_array_almost_equal(loaded.constraints, original.constraints)

    def test_roundtrip_objectives_only(self, tmp_path):
        original = self._make_result(n_sol=6, n_obj=4, n_con=0, n_var=0)
        db_path = str(tmp_path / "run_obj.db")
        save_run(original, db_path)
        loaded = load_run(db_path)

        assert loaded.constraints is None
        assert loaded.variables is None
        np.testing.assert_array_almost_equal(loaded.objectives, original.objectives)

    def test_roundtrip_empty_solutions(self, tmp_path):
        original = RunResult(
            problem="Empty",
            formulation="2OB",
            algorithm="Algo",
            rep=1,
            objectives=np.empty((0, 2)),
            constraints=np.empty((0, 3)),
            variables=np.empty((0, 4)),
            elapsed_seconds=0.1,
            nfe=100,
        )
        db_path = str(tmp_path / "run_empty.db")
        save_run(original, db_path)
        loaded = load_run(db_path)

        assert loaded.problem == "Empty"
        assert loaded.nfe == 100
        assert loaded.objectives.shape[0] == 0
        # With empty solutions, column counts are 0 so constraints/variables
        # come back as None since there are no columns to read.

    def test_save_creates_parent_dirs(self, tmp_path):
        result = self._make_result()
        deep_path = str(tmp_path / "a" / "b" / "c" / "run.db")
        save_run(result, deep_path)
        loaded = load_run(deep_path)
        assert loaded.problem == result.problem

    def test_load_missing_meta_raises(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "bad.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE run_meta (problem TEXT)")
        conn.commit()
        conn.close()
        with pytest.raises(ValueError, match="No run_meta row"):
            load_run(db_path)


# ======================================================================
# Integration test (marked slow -- skipped by default)
# ======================================================================

@pytest.mark.slow
class TestIntegration:
    """Run a tiny real optimisation.  Requires pymoo or platypus."""

    def test_run_single_pymoo(self):
        from pymoo.algorithms.moo.nsga2 import NSGA2

        from dual_bench.runner import run_single

        result = run_single(
            Spring(),
            "2OB",
            lambda prob: NSGA2(pop_size=20),
            nfe=200,
        )
        assert result.objectives.shape[0] > 0
        assert result.objectives.shape[1] == 2  # 2OB has 2 objectives
        assert result.nfe >= 200

    def test_run_single_platypus(self):
        from platypus import NSGAII

        from dual_bench.runner import run_single

        result = run_single(
            Spring(),
            "2OB",
            lambda prob: NSGAII(prob, population_size=20),
            nfe=200,
        )
        assert result.objectives.shape[0] > 0
        assert result.objectives.shape[1] == 2
