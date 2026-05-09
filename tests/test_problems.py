# -*- coding: utf-8 -*-
"""Tests for all problems: correct shapes, metadata consistency, feasibility."""

import numpy as np
import pytest

from dual_bench.problems import ALL_PROBLEMS
from dual_bench.problems.cmop import CMOP_REGISTRY


# ======================================================================
# Single-objective DualProblem tests
# ======================================================================

@pytest.mark.parametrize("ProblemClass", ALL_PROBLEMS, ids=lambda c: c.name)
class TestDualProblems:

    def test_metadata(self, ProblemClass):
        p = ProblemClass()
        assert p.n_vars > 0
        assert p.n_ratios > 0
        assert len(p.var_bounds) == p.n_vars
        assert p.name != ""

    def test_evaluate_shapes(self, ProblemClass):
        p = ProblemClass()
        x = p.midpoint()
        f, ratios = p.evaluate(x)
        assert isinstance(f, float)
        assert len(ratios) == p.n_ratios

    def test_midpoint(self, ProblemClass):
        p = ProblemClass()
        mid = p.midpoint()
        assert len(mid) == p.n_vars
        for val, (lb, ub) in zip(mid, p.var_bounds):
            assert lb <= val <= ub

    def test_is_feasible_returns_bool(self, ProblemClass):
        p = ProblemClass()
        result = p.is_feasible(p.midpoint())
        assert isinstance(result, bool)

    def test_info_returns_string(self, ProblemClass):
        p = ProblemClass()
        text = p.info()
        assert isinstance(text, str)
        assert p.name in text


# ======================================================================
# CMOProblem tests
# ======================================================================

@pytest.mark.parametrize(
    "name,problem",
    list(CMOP_REGISTRY.items()),
    ids=list(CMOP_REGISTRY.keys()),
)
class TestCMOProblems:

    def test_metadata(self, name, problem):
        assert problem.n_var > 0
        assert problem.n_obj >= 2
        assert problem.n_constr >= 0
        assert len(problem.xl) == problem.n_var
        assert len(problem.xu) == problem.n_var

    def test_evaluate_shapes(self, name, problem):
        x = (problem.xl + problem.xu) / 2.0
        F, G = problem.evaluate(x)
        assert len(F) == problem.n_obj
        assert len(G) == problem.n_constr

    def test_bounds_order(self, name, problem):
        assert np.all(problem.xl <= problem.xu)

    def test_is_feasible_returns_bool(self, name, problem):
        x = (problem.xl + problem.xu) / 2.0
        result = problem.is_feasible(x)
        assert isinstance(result, bool)

    def test_repr(self, name, problem):
        r = repr(problem)
        assert name in r
