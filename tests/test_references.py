# -*- coding: utf-8 -*-
"""Tests for dual_bench.references module."""

import pytest

from dual_bench.references import (
    info,
    list_problems,
    optimum,
    paper_a_by_algorithm,
    paper_a_by_formulation,
    paper_a_results,
    paper_a_setup,
)
from dual_bench.references.optima import ProblemInfo
from dual_bench.references.paper_a import (
    ALGORITHMS,
    FORMULATIONS,
    PROBLEMS,
    _BY_ALGORITHM,
    _BY_FORMULATION,
)


# ======================================================================
# Optima
# ======================================================================

class TestOptima:
    def test_known_problems(self):
        expected = {
            "Spring": 0.012665,
            "PressureVessel": 5885.33,
            "WeldedBeam": 1.7249,
            "SpeedReducer": 2994.47,
            "CantileverBeam": 1.3400,
        }
        for name, f_star in expected.items():
            assert optimum(name) == f_star

    def test_unknown_problem_raises(self):
        with pytest.raises(KeyError):
            optimum("NonExistent")

    def test_info_returns_dataclass(self):
        pi = info("Spring")
        assert isinstance(pi, ProblemInfo)
        assert pi.n_vars == 3
        assert pi.n_ratios == 4
        assert pi.known_x is not None
        assert len(pi.known_x) == pi.n_vars

    def test_list_problems(self):
        names = list_problems()
        assert len(names) == 5
        assert "Spring" in names
        assert names == sorted(names)

    def test_all_problems_have_known_x(self):
        for name in list_problems():
            pi = info(name)
            assert pi.known_x is not None
            assert len(pi.known_x) == pi.n_vars

    def test_all_problems_have_source(self):
        for name in list_problems():
            pi = info(name)
            assert len(pi.source) > 0


# ======================================================================
# Paper A data integrity
# ======================================================================

class TestPaperAData:
    def test_by_algorithm_count(self):
        """5 problems × 5 formulations × 6 algorithms = 150 entries."""
        assert len(_BY_ALGORITHM) == 150

    def test_by_formulation_count(self):
        """5 problems × 5 formulations = 25 entries."""
        assert len(_BY_FORMULATION) == 25

    def test_all_combinations_present(self):
        for p in PROBLEMS:
            for f in FORMULATIONS:
                for a in ALGORITHMS:
                    assert (p, f, a) in _BY_ALGORITHM, f"Missing ({p}, {f}, {a})"

    def test_hv_values_in_range(self):
        for key, (hv_m, hv_s, df_m, df_s, n) in _BY_ALGORITHM.items():
            assert 0.0 <= hv_m <= 1.0, f"{key}: hv_mean={hv_m} out of [0,1]"
            assert hv_s >= 0.0, f"{key}: hv_std negative"

    def test_n_reps_is_20(self):
        for key, (_, _, _, _, n) in _BY_ALGORITHM.items():
            assert n == 20, f"{key}: expected n=20, got {n}"

    def test_n_pooled_is_120(self):
        for key, vals in _BY_FORMULATION.items():
            assert vals[-1] == 120, f"{key}: expected n=120, got {vals[-1]}"


# ======================================================================
# Paper A public API
# ======================================================================

class TestPaperAAPI:
    def test_by_algorithm_unfiltered(self):
        rows = paper_a_by_algorithm()
        assert len(rows) == 150
        assert all("hv_mean" in r for r in rows)

    def test_by_algorithm_filter_problem(self):
        rows = paper_a_by_algorithm(problem="Spring")
        assert len(rows) == 30  # 5 formulations × 6 algorithms
        assert all(r["problem"] == "Spring" for r in rows)

    def test_by_algorithm_filter_all(self):
        rows = paper_a_by_algorithm("Spring", "2OB-REST", "NSGAII")
        assert len(rows) == 1
        assert rows[0]["hv_mean"] == 0.7537

    def test_by_formulation_unfiltered(self):
        rows = paper_a_by_formulation()
        assert len(rows) == 25

    def test_by_formulation_filter(self):
        rows = paper_a_by_formulation("WeldedBeam", "2OB-REST")
        assert len(rows) == 1
        assert rows[0]["hv_median"] == 0.9096

    def test_paper_a_results_dispatches(self):
        # Without algorithm -> by_formulation
        rows = paper_a_results("Spring")
        assert len(rows) == 5
        # With algorithm -> by_algorithm
        rows = paper_a_results("Spring", algorithm="GDE3")
        assert len(rows) == 5  # 5 formulations

    def test_setup(self):
        s = paper_a_setup()
        assert s["total_runs"] == 3000
        assert s["nfe"] == 25000
        assert s["engine"] == "Platypus"
        assert len(s["problems"]) == 5
        assert len(s["formulations"]) == 5
        assert len(s["algorithms"]) == 6
