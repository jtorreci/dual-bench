# -*- coding: utf-8 -*-
"""Tests for formulation wrappers: shapes, constraints, progress updates."""

import pytest
import numpy as np

from dual_bench.problems import ALL_PROBLEMS, Spring
from dual_bench.problems.cmop import BNH, LIRCMOP1
from dual_bench.formulations import (
    MOFormulation,
    BiobjFormulation,
    ManyobjFormulation,
    BiobjRestFormulation,
    ManyobjRestFormulation,
    CMOPEnvelope,
    CMOPFull,
    CMOPStandard,
)


# ======================================================================
# SO formulation tests
# ======================================================================

SO_FORMULATIONS = [
    ("MO", MOFormulation),
    ("2OB", BiobjFormulation),
    ("FOB", ManyobjFormulation),
    ("2OB-REST", BiobjRestFormulation),
    ("FOB-REST", ManyobjRestFormulation),
]


@pytest.mark.parametrize("ProblemClass", ALL_PROBLEMS, ids=lambda c: c.name)
@pytest.mark.parametrize("fname,FormClass", SO_FORMULATIONS, ids=lambda x: x[0] if isinstance(x, tuple) else x)
class TestSOFormulations:

    def test_n_objectives(self, ProblemClass, fname, FormClass):
        p = ProblemClass()
        form = FormClass(p) if FormClass != MOFormulation else FormClass(p, penalty=1e6)
        if FormClass in (MOFormulation,):
            assert form.n_objectives == 1
        elif FormClass in (BiobjFormulation, BiobjRestFormulation):
            assert form.n_objectives == 2
        else:
            assert form.n_objectives == 1 + p.n_ratios

    def test_n_constraints(self, ProblemClass, fname, FormClass):
        p = ProblemClass()
        form = FormClass(p) if FormClass != MOFormulation else FormClass(p, penalty=1e6)
        if FormClass in (BiobjFormulation, ManyobjFormulation):
            assert form.n_constraints == 0
        else:
            assert form.n_constraints == p.n_ratios

    def test_evaluate_shapes(self, ProblemClass, fname, FormClass):
        p = ProblemClass()
        form = FormClass(p) if FormClass != MOFormulation else FormClass(p, penalty=1e6)
        objs, cons = form.evaluate(p.midpoint())
        assert len(objs) == form.n_objectives
        assert len(cons) == form.n_constraints

    def test_var_bounds_length(self, ProblemClass, fname, FormClass):
        p = ProblemClass()
        form = FormClass(p) if FormClass != MOFormulation else FormClass(p, penalty=1e6)
        assert len(form.var_bounds) == p.n_vars


class TestRESTConstraintViolation:
    """REST formulations should report violated constraints (G > 0) for infeasible points."""

    def test_biobj_rest_infeasible(self):
        p = Spring()
        form = BiobjRestFormulation(p)
        # Use bounds that are likely infeasible
        x_low = [lb for lb, ub in p.var_bounds]
        _, cons = form.evaluate(x_low)
        # G <= 0 convention: violated constraints are > 0
        assert any(c > 0.0 for c in cons)

    def test_manyobj_rest_infeasible(self):
        p = Spring()
        form = ManyobjRestFormulation(p)
        x_low = [lb for lb, ub in p.var_bounds]
        _, cons = form.evaluate(x_low)
        assert any(c > 0.0 for c in cons)


# ======================================================================
# CMOP formulation tests
# ======================================================================

class TestCMOPFormulations:

    @pytest.fixture
    def bnh(self):
        return BNH()

    @pytest.fixture
    def lircmop1(self):
        return LIRCMOP1()

    def test_standard_shapes(self, bnh):
        form = CMOPStandard(bnh)
        assert form.n_objectives == bnh.n_obj
        assert form.n_constraints == bnh.n_constr
        x = (bnh.xl + bnh.xu) / 2.0
        objs, cons = form.evaluate(x)
        assert len(objs) == bnh.n_obj
        assert len(cons) == bnh.n_constr

    def test_envelope_shapes(self, bnh):
        form = CMOPEnvelope(bnh)
        assert form.n_objectives == bnh.n_obj + 1
        assert form.n_constraints == bnh.n_constr
        x = (bnh.xl + bnh.xu) / 2.0
        objs, cons = form.evaluate(x)
        assert len(objs) == bnh.n_obj + 1
        assert len(cons) == bnh.n_constr

    def test_full_shapes(self, bnh):
        form = CMOPFull(bnh)
        assert form.n_objectives == bnh.n_obj + bnh.n_constr
        assert form.n_constraints == bnh.n_constr
        x = (bnh.xl + bnh.xu) / 2.0
        objs, cons = form.evaluate(x)
        assert len(objs) == bnh.n_obj + bnh.n_constr
        assert len(cons) == bnh.n_constr

    def test_set_progress_updates_epsilon(self, bnh):
        form = CMOPEnvelope(bnh, eps_max=0.5, cp=5)
        assert form.epsilon == 0.5
        form.set_progress(500, 1000)
        assert 0 < form.epsilon < 0.5
        form.set_progress(1000, 1000)
        assert form.epsilon == pytest.approx(0.0, abs=1e-10)

    def test_standard_set_progress_noop(self, bnh):
        form = CMOPStandard(bnh)
        form.set_progress(500, 1000)  # should not raise

    def test_envelope_with_scales(self, bnh):
        scales = np.array([10.0, 10.0])
        form = CMOPEnvelope(bnh, scales=scales)
        x = (bnh.xl + bnh.xu) / 2.0
        objs, cons = form.evaluate(x)
        assert len(objs) == bnh.n_obj + 1

    def test_full_d_cap(self, lircmop1):
        form = CMOPFull(lircmop1, d_cap=2.0)
        x = (lircmop1.xl + lircmop1.xu) / 2.0
        objs, _ = form.evaluate(x)
        # d values should be capped at 2.0
        d_values = objs[lircmop1.n_obj:]
        assert all(d <= 2.0 for d in d_values)
