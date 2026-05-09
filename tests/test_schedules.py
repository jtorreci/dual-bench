# -*- coding: utf-8 -*-
"""Tests for epsilon-decay schedules."""

import pytest

from dual_bench.schedules import (
    epsilon_pow,
    epsilon_lin,
    epsilon_exp,
    DEFAULT_SCHEDULE,
    DEFAULT_EPS_MAX,
    DEFAULT_CP,
)


class TestEpsilonPow:

    def test_initial_value(self):
        assert epsilon_pow(0, 1000) == pytest.approx(0.2)

    def test_final_value(self):
        assert epsilon_pow(1000, 1000) == pytest.approx(0.0, abs=1e-15)

    def test_monotonically_decreasing(self):
        T = 1000
        values = [epsilon_pow(t, T) for t in range(0, T + 1, 10)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_custom_params(self):
        val = epsilon_pow(0, 100, eps_max=1.0, cp=3)
        assert val == pytest.approx(1.0)

    def test_beyond_T(self):
        assert epsilon_pow(1500, 1000) == pytest.approx(0.0, abs=1e-15)


class TestEpsilonLin:

    def test_initial_value(self):
        assert epsilon_lin(0, 1000) == pytest.approx(0.2)

    def test_final_value(self):
        assert epsilon_lin(1000, 1000) == pytest.approx(0.0, abs=1e-15)

    def test_midpoint(self):
        assert epsilon_lin(500, 1000) == pytest.approx(0.1)

    def test_monotonically_decreasing(self):
        T = 1000
        values = [epsilon_lin(t, T) for t in range(0, T + 1, 10)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]


class TestEpsilonExp:

    def test_initial_value(self):
        assert epsilon_exp(0, 1000) == pytest.approx(0.2)

    def test_final_value_near_zero(self):
        val = epsilon_exp(1000, 1000)
        assert val < 0.01  # exp(-5) * 0.2 ~ 0.00135

    def test_monotonically_decreasing(self):
        T = 1000
        values = [epsilon_exp(t, T) for t in range(0, T + 1, 10)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_custom_ce(self):
        val = epsilon_exp(1000, 1000, eps_max=1.0, ce=0)
        assert val == pytest.approx(1.0)


class TestDefaults:

    def test_default_schedule_is_pow(self):
        assert DEFAULT_SCHEDULE is epsilon_pow

    def test_default_eps_max(self):
        assert DEFAULT_EPS_MAX == 0.2

    def test_default_cp(self):
        assert DEFAULT_CP == 5
