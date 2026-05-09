# -*- coding: utf-8 -*-
"""Tests for dual_bench.analysis -- statistics, metrics, and tables."""

import math

import numpy as np
import pytest

from dual_bench.analysis import (
    bootstrap_ci,
    cliff_delta,
    delta_f_star,
    feasibility_rate,
    friedman_test,
    hypervolume_2d,
    igd,
    kruskal_wallis,
    nemenyi_posthoc,
    spread,
    summary_table,
    to_latex,
)


# ============================================================================
# statistics.py
# ============================================================================

class TestFriedmanTest:
    """Friedman rank test."""

    def test_clearly_different_groups(self):
        data = {
            "A": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "B": np.array([6.0, 7.0, 8.0, 9.0, 10.0]),
            "C": np.array([11.0, 12.0, 13.0, 14.0, 15.0]),
        }
        result = friedman_test(data)
        assert result["p_value"] < 0.05
        assert result["statistic"] > 0
        # A should have rank 1 (smallest), C rank 3
        assert result["rankings"]["A"] < result["rankings"]["C"]

    def test_identical_groups(self):
        # When all values are tied, scipy returns NaN for both stat and p.
        arr = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        data = {"X": arr.copy(), "Y": arr.copy(), "Z": arr.copy()}
        result = friedman_test(data)
        assert (
            math.isnan(result["p_value"])
            or result["p_value"] == 1.0
            or result["statistic"] == 0.0
        )

    def test_unequal_lengths_raises(self):
        data = {"A": np.array([1, 2, 3]), "B": np.array([4, 5])}
        with pytest.raises(ValueError):
            friedman_test(data)

    def test_rankings_are_mean_ranks(self):
        # Friedman requires >= 3 groups in scipy
        data = {
            "lo": np.array([1.0, 1.0, 1.0]),
            "mid": np.array([5.0, 5.0, 5.0]),
            "hi": np.array([9.0, 9.0, 9.0]),
        }
        result = friedman_test(data)
        assert result["rankings"]["lo"] == pytest.approx(1.0)
        assert result["rankings"]["mid"] == pytest.approx(2.0)
        assert result["rankings"]["hi"] == pytest.approx(3.0)


class TestNemenyiPosthoc:
    """Nemenyi post-hoc test."""

    def test_returns_cd_and_comparisons(self):
        data = {
            "A": np.array([1, 2, 3, 4, 5], dtype=float),
            "B": np.array([6, 7, 8, 9, 10], dtype=float),
            "C": np.array([11, 12, 13, 14, 15], dtype=float),
        }
        result = nemenyi_posthoc(data, alpha=0.05)
        assert "cd" in result
        assert result["cd"] > 0
        assert len(result["comparisons"]) == 3  # C(3,2)

    def test_significant_pairs(self):
        data = {
            "A": np.array([1, 2, 3, 4, 5], dtype=float),
            "B": np.array([6, 7, 8, 9, 10], dtype=float),
            "C": np.array([11, 12, 13, 14, 15], dtype=float),
        }
        result = nemenyi_posthoc(data)
        # A vs C should be significant (rank diff = 2.0, CD for k=3 n=5)
        ac = [c for c in result["comparisons"]
              if {c["group_a"], c["group_b"]} == {"A", "C"}][0]
        assert ac["significant"] is True


class TestKruskalWallis:
    """Kruskal-Wallis H-test."""

    def test_different_groups_significant(self):
        data = {
            "A": np.array([1, 2, 3, 4]),
            "B": np.array([10, 11, 12, 13]),
        }
        result = kruskal_wallis(data)
        assert result["p_value"] < 0.05

    def test_same_distribution_not_significant(self):
        rng = np.random.default_rng(0)
        data = {
            "X": rng.normal(0, 1, 50),
            "Y": rng.normal(0, 1, 50),
        }
        result = kruskal_wallis(data)
        assert result["p_value"] > 0.05


class TestCliffDelta:
    """Cliff's delta effect size."""

    def test_identical_arrays(self):
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        delta, mag = cliff_delta(x, x.copy())
        assert delta == pytest.approx(0.0)
        assert mag == "negligible"

    def test_non_overlapping_positive(self):
        x = np.array([1, 2, 3], dtype=float)
        y = np.array([4, 5, 6], dtype=float)
        delta, mag = cliff_delta(x, y)
        assert delta == pytest.approx(-1.0)
        assert mag == "large"

    def test_non_overlapping_reversed(self):
        x = np.array([7, 8, 9], dtype=float)
        y = np.array([1, 2, 3], dtype=float)
        delta, mag = cliff_delta(x, y)
        assert delta == pytest.approx(1.0)
        assert mag == "large"

    def test_empty_array(self):
        delta, mag = cliff_delta(np.array([]), np.array([1, 2]))
        assert delta == 0.0
        assert mag == "negligible"

    def test_small_effect(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        delta, mag = cliff_delta(x, y)
        assert mag in ("negligible", "small")


class TestBootstrapCI:
    """Bootstrap confidence interval."""

    def test_contains_true_mean(self):
        rng = np.random.default_rng(123)
        data = rng.normal(loc=10.0, scale=1.0, size=100)
        lo, hi = bootstrap_ci(data, n_boot=5000, ci=0.95, seed=42)
        assert lo < 10.0 < hi

    def test_narrow_for_low_variance(self):
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        lo, hi = bootstrap_ci(data)
        assert lo == pytest.approx(5.0)
        assert hi == pytest.approx(5.0)

    def test_returns_tuple_of_floats(self):
        lo, hi = bootstrap_ci(np.array([1.0, 2.0, 3.0]))
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo <= hi


# ============================================================================
# metrics.py
# ============================================================================

class TestHypervolume2D:
    """2-D hypervolume indicator."""

    def test_single_point(self):
        pts = np.array([[1.0, 2.0]])
        ref = np.array([5.0, 5.0])
        hv = hypervolume_2d(pts, ref)
        assert hv == pytest.approx((5.0 - 1.0) * (5.0 - 2.0))  # 12.0

    def test_staircase(self):
        # Three points forming a staircase:
        # (1,4), (2,2), (4,1) with ref (5,5)
        pts = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]])
        ref = np.array([5.0, 5.0])
        # Area = (5-1)*(5-4) + (5-2)*(4-2) + (5-4)*(2-1) = 4+6+1 = 11
        hv = hypervolume_2d(pts, ref)
        assert hv == pytest.approx(11.0)

    def test_dominated_points_ignored(self):
        # (3,3) is dominated by (1,1), should not change HV
        pts = np.array([[1.0, 1.0], [3.0, 3.0]])
        ref = np.array([5.0, 5.0])
        hv_with = hypervolume_2d(pts, ref)
        hv_without = hypervolume_2d(np.array([[1.0, 1.0]]), ref)
        assert hv_with == pytest.approx(hv_without)

    def test_empty_after_filter(self):
        # All points outside reference
        pts = np.array([[6.0, 6.0]])
        ref = np.array([5.0, 5.0])
        assert hypervolume_2d(pts, ref) == 0.0

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            hypervolume_2d(np.array([1, 2, 3]), np.array([5, 5]))


class TestDeltaFStar:
    """Relative distance to known optimum."""

    def test_at_optimum(self):
        assert delta_f_star(10.0, 10.0) == pytest.approx(0.0)

    def test_positive_distance(self):
        assert delta_f_star(12.0, 10.0) == pytest.approx(0.2)

    def test_zero_optimum_returns_nan(self):
        assert math.isnan(delta_f_star(5.0, 0.0))


class TestFeasibilityRate:
    """Feasibility rate metric."""

    def test_all_feasible(self):
        ratios = np.array([[0.5, 0.8], [0.9, 1.0], [0.1, 0.3]])
        assert feasibility_rate(ratios) == pytest.approx(1.0)

    def test_none_feasible(self):
        ratios = np.array([[1.1, 0.5], [0.5, 1.5]])
        assert feasibility_rate(ratios) == pytest.approx(0.0)

    def test_half_feasible(self):
        ratios = np.array([[0.5, 0.5], [1.5, 0.5]])
        assert feasibility_rate(ratios) == pytest.approx(0.5)

    def test_1d_input(self):
        # Single solution, single ratio
        assert feasibility_rate(np.array([0.5])) == pytest.approx(1.0)
        assert feasibility_rate(np.array([1.5])) == pytest.approx(0.0)


class TestIGD:
    """Inverted Generational Distance."""

    def test_perfect_match(self):
        ref = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        assert igd(ref.copy(), ref) == pytest.approx(0.0)

    def test_positive_distance(self):
        approx = np.array([[0.0, 0.0]])
        ref = np.array([[3.0, 4.0]])
        assert igd(approx, ref) == pytest.approx(5.0)

    def test_multiple_reference_points(self):
        approx = np.array([[0.0, 0.0], [1.0, 1.0]])
        ref = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        # First two ref points have distance 0; third is sqrt(0.5) from either
        val = igd(approx, ref)
        expected = (0.0 + 0.0 + np.sqrt(0.5)) / 3
        assert val == pytest.approx(expected)


class TestSpread:
    """Spread / diversity indicator."""

    def test_two_points(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        assert spread(pts) == pytest.approx(5.0)

    def test_single_point(self):
        assert spread(np.array([[1.0, 2.0]])) == 0.0

    def test_collinear_points(self):
        pts = np.array([[0, 0], [1, 0], [2, 0], [5, 0]], dtype=float)
        assert spread(pts) == pytest.approx(5.0)


# ============================================================================
# tables.py
# ============================================================================

class TestToLatex:
    """LaTeX table generation."""

    def test_basic_table(self):
        data = [
            {"name": "A", "val": 1.5},
            {"name": "B", "val": 2.3},
        ]
        tex = to_latex(data, columns=["name", "val"])
        assert "\\begin{table}" in tex
        assert "\\toprule" in tex
        assert "\\bottomrule" in tex
        assert "A" in tex

    def test_highlight_best(self):
        data = [
            {"alg": "X", "score": 0.9},
            {"alg": "Y", "score": 0.5},
        ]
        tex = to_latex(data, columns=["alg", "score"], highlight_best=True)
        assert "\\textbf{0.5000}" in tex

    def test_no_highlight(self):
        data = [
            {"alg": "X", "score": 0.9},
            {"alg": "Y", "score": 0.5},
        ]
        tex = to_latex(data, columns=["alg", "score"], highlight_best=False)
        assert "\\textbf" not in tex

    def test_caption_and_label(self):
        data = [{"a": 1}]
        tex = to_latex(data, columns=["a"], caption="My Table", label="tab:my")
        assert "\\caption{My Table}" in tex
        assert "\\label{tab:my}" in tex

    def test_nan_rendered_as_dash(self):
        data = [{"x": float("nan")}]
        tex = to_latex(data, columns=["x"])
        assert "--" in tex


class TestSummaryTable:
    """Pivot-style summary table."""

    def test_basic_pivot(self):
        results = [
            {"formulation": "MO", "algorithm": "NSGA-II", "hv_mean": 0.8},
            {"formulation": "MO", "algorithm": "GDE3", "hv_mean": 0.7},
            {"formulation": "2OB", "algorithm": "NSGA-II", "hv_mean": 0.9},
            {"formulation": "2OB", "algorithm": "GDE3", "hv_mean": 0.6},
        ]
        tex = summary_table(results, metric="hv_mean",
                            rows="formulation", cols="algorithm")
        assert "\\begin{table}" in tex
        assert "MO" in tex
        assert "NSGA-II" in tex
        assert "GDE3" in tex

    def test_best_highlighted(self):
        results = [
            {"r": "A", "c": "X", "v": 1.0},
            {"r": "B", "c": "X", "v": 2.0},
        ]
        tex = summary_table(results, metric="v", rows="r", cols="c")
        assert "\\textbf{1.0000}" in tex


# ============================================================================
# Import smoke test
# ============================================================================

def test_all_reexported():
    """All public functions are accessible from dual_bench.analysis."""
    from dual_bench import analysis
    for name in [
        "friedman_test", "nemenyi_posthoc", "kruskal_wallis",
        "cliff_delta", "bootstrap_ci",
        "hypervolume_2d", "delta_f_star", "feasibility_rate", "igd", "spread",
        "to_latex", "summary_table",
    ]:
        assert hasattr(analysis, name), f"Missing: {name}"
