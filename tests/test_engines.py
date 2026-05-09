# -*- coding: utf-8 -*-
"""Smoke tests for engine adapters (pymoo and Platypus)."""

import pytest
import numpy as np

from dual_bench import Spring
from dual_bench.problems.cmop import BNH
from dual_bench.formulations import CMOPEnvelope, BiobjRestFormulation
from dual_bench.engines.base import Solution


class TestSolutionDataclass:

    def test_default_values(self):
        s = Solution(x=[1.0, 2.0], objectives=[3.0])
        assert s.x == [1.0, 2.0]
        assert s.objectives == [3.0]
        assert s.constraints == []
        assert s.feasible is True

    def test_custom_values(self):
        s = Solution(x=[1.0], objectives=[2.0], constraints=[0.5], feasible=False)
        assert s.feasible is False
        assert s.constraints == [0.5]


# ======================================================================
# pymoo adapter tests
# ======================================================================

pymoo = pytest.importorskip("pymoo")


class TestPymooAdapter:

    def test_topymoo_shapes(self):
        from dual_bench.engines.pymoo_engine import ToPymoo

        form = BiobjRestFormulation(Spring())
        prob = ToPymoo(form)
        assert prob.n_var == 3
        assert prob.n_obj == 2
        assert prob.n_ieq_constr == 4

    def test_topymoo_evaluate(self):
        from dual_bench.engines.pymoo_engine import ToPymoo

        form = BiobjRestFormulation(Spring())
        prob = ToPymoo(form)
        X = np.array([[1.0, 0.5, 8.0]])
        out = {}
        prob._evaluate(X, out)
        assert "F" in out
        assert out["F"].shape == (1, 2)
        assert "G" in out
        assert out["G"].shape == (1, 4)

    def test_topymoo_unconstrained(self):
        from dual_bench.engines.pymoo_engine import ToPymoo
        from dual_bench.formulations import BiobjFormulation

        form = BiobjFormulation(Spring())
        prob = ToPymoo(form)
        assert prob.n_ieq_constr == 0
        X = np.array([[1.0, 0.5, 8.0]])
        out = {}
        prob._evaluate(X, out)
        assert "G" not in out

    def test_epsilon_callback(self):
        from dual_bench.engines.pymoo_engine import EpsilonCallback

        form = CMOPEnvelope(BNH(), eps_max=0.3)
        cb = EpsilonCallback(form, 10000)
        assert form.epsilon == 0.3

    @pytest.mark.slow
    def test_pymoo_short_run(self):
        from dual_bench.engines.pymoo_engine import ToPymoo, EpsilonCallback
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize

        # Use CMOPEnvelope which has set_progress for the callback
        form = CMOPEnvelope(BNH())
        prob = ToPymoo(form)
        result = minimize(
            prob, NSGA2(pop_size=20),
            ('n_eval', 200),
            callback=EpsilonCallback(form, 200),
            verbose=False,
        )
        assert result.F is not None
        assert len(result.F) > 0


# ======================================================================
# Platypus adapter tests
# ======================================================================

platypus_mod = pytest.importorskip("platypus")


class TestPlatypusAdapter:

    def test_toplatypus_shapes(self):
        from dual_bench.engines.platypus_engine import ToPlatypus

        form = BiobjRestFormulation(Spring())
        prob = ToPlatypus(form)
        assert prob.nvars == 3
        assert prob.nobjs == 2
        assert prob.nconstrs == 4

    def test_toplatypus_unconstrained(self):
        from dual_bench.engines.platypus_engine import ToPlatypus
        from dual_bench.formulations import BiobjFormulation

        form = BiobjFormulation(Spring())
        prob = ToPlatypus(form)
        assert prob.nconstrs == 0

    def test_epsilon_updater(self):
        from dual_bench.engines.platypus_engine import EpsilonUpdater

        form = CMOPEnvelope(BNH(), eps_max=0.3)
        updater = EpsilonUpdater(form, 10000)
        assert form.epsilon == 0.3

    @pytest.mark.slow
    def test_platypus_short_run(self):
        from dual_bench.engines.platypus_engine import ToPlatypus, EpsilonUpdater
        from platypus import NSGAII

        form = Spring().to_2ob_rest()
        prob = ToPlatypus(form)
        algo = NSGAII(prob, population_size=20)
        updater = EpsilonUpdater(form, 200)
        algo.run(200)
        assert len(algo.result) > 0
