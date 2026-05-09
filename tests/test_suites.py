# -*- coding: utf-8 -*-
"""Tests for suite registry."""

import pytest

from dual_bench.problems.suites import suite, list_suites


class TestListSuites:

    def test_returns_list(self):
        result = list_suites()
        assert isinstance(result, list)

    def test_contains_expected_suites(self):
        names = list_suites()
        assert "engineering5" in names
        assert "classic5" in names
        assert "lircmop" in names
        assert "zdt" in names
        assert "dtlz" in names
        assert "wfg" in names
        assert "mw" in names
        assert "dascmop" in names

    def test_sorted(self):
        names = list_suites()
        assert names == sorted(names)


class TestSuiteEngineering5:

    def test_returns_5_problems(self):
        problems = suite("engineering5")
        assert len(problems) == 5

    def test_all_have_name(self):
        for p in suite("engineering5"):
            assert p.name != ""

    def test_problem_names(self):
        names = [p.name for p in suite("engineering5")]
        assert "Spring" in names
        assert "CantileverBeam" in names


class TestSuiteClassic5:

    def test_returns_5_problems(self):
        problems = suite("classic5")
        assert len(problems) == 5

    def test_all_cmop(self):
        from dual_bench.problems.base import CMOProblem
        for p in suite("classic5"):
            assert isinstance(p, CMOProblem)


class TestSuiteLircmop:

    def test_returns_14_problems(self):
        problems = suite("lircmop")
        assert len(problems) == 14


class TestSuiteUnknown:

    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown suite"):
            suite("nonexistent")


class TestPymooSuites:
    """Suites that require pymoo."""

    pymoo = pytest.importorskip("pymoo")

    def test_zdt_returns_6(self):
        problems = suite("zdt")
        assert len(problems) == 6

    def test_dtlz_returns_7(self):
        problems = suite("dtlz")
        assert len(problems) == 7

    def test_wfg_returns_9(self):
        problems = suite("wfg")
        assert len(problems) == 9

    def test_mw_returns_14(self):
        problems = suite("mw")
        assert len(problems) == 14

    def test_dascmop_returns_9(self):
        problems = suite("dascmop")
        assert len(problems) == 9

    def test_zdt_problems_are_unconstrained(self):
        for p in suite("zdt"):
            assert p.n_constr == 0

    def test_mw_problems_are_constrained(self):
        for p in suite("mw"):
            assert p.n_constr > 0
