# -*- coding: utf-8 -*-
"""Non-parametric hypothesis tests for algorithm comparison."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats as sp_stats

# Nemenyi q_alpha for alpha=0.05, k = 2..10  (Studentized Range / sqrt(2))
_NEMENYI_Q_005: dict[int, float] = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def _nemenyi_q(k: int, alpha: float = 0.05) -> float:
    """Return the Nemenyi critical q value."""
    if hasattr(sp_stats, "studentized_range"):
        q_raw = sp_stats.studentized_range.ppf(1 - alpha, k, np.inf)
        return q_raw / np.sqrt(2)
    if alpha == 0.05 and k in _NEMENYI_Q_005:
        return _NEMENYI_Q_005[k]
    raise ValueError(
        f"No critical value for k={k}, alpha={alpha}. "
        f"Upgrade scipy>=1.7 or use k in 2..10 with alpha=0.05."
    )


def friedman_test(data: dict[str, np.ndarray]) -> dict:
    """Friedman rank test across multiple groups.

    Parameters
    ----------
    data : dict mapping group_name -> 1-D array of values (one per block).
        All arrays must have the same length.

    Returns
    -------
    dict with keys ``statistic``, ``p_value``, ``rankings``
        (dict of mean ranks, 1 = best/lowest).
    """
    names = list(data.keys())
    arrays = [np.asarray(data[n], dtype=float) for n in names]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("All groups must have the same number of observations.")
    stat, p = sp_stats.friedmanchisquare(*arrays)
    # Compute mean ranks (rank 1 = smallest value)
    matrix = np.column_stack(arrays)  # (n, k)
    ranks = np.zeros_like(matrix)
    for i in range(n):
        ranks[i] = sp_stats.rankdata(matrix[i])
    mean_ranks = {name: float(ranks[:, j].mean()) for j, name in enumerate(names)}
    return {"statistic": float(stat), "p_value": float(p), "rankings": mean_ranks}


def nemenyi_posthoc(data: dict[str, np.ndarray], alpha: float = 0.05) -> dict:
    """Nemenyi post-hoc test after a significant Friedman test.

    Returns dict with ``cd`` (critical difference) and ``comparisons``
    (list of dicts with ``group_a``, ``group_b``, ``rank_diff``, ``significant``).
    """
    result = friedman_test(data)
    rankings = result["rankings"]
    names = list(data.keys())
    k = len(names)
    n = len(next(iter(data.values())))
    q_alpha = _nemenyi_q(k, alpha)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    comparisons = []
    for a, b in combinations(names, 2):
        diff = abs(rankings[a] - rankings[b])
        comparisons.append({
            "group_a": a, "group_b": b,
            "rank_diff": float(diff), "significant": bool(diff > cd),
        })
    return {"cd": float(cd), "comparisons": comparisons}


def kruskal_wallis(data: dict[str, np.ndarray]) -> dict:
    """Kruskal-Wallis H-test.  Returns dict with ``statistic``, ``p_value``."""
    arrays = [np.asarray(v, dtype=float) for v in data.values()]
    stat, p = sp_stats.kruskal(*arrays)
    return {"statistic": float(stat), "p_value": float(p)}


def cliff_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Cliff's delta effect size with magnitude label.

    Returns (delta, magnitude) where magnitude is one of
    ``'negligible'``, ``'small'``, ``'medium'``, ``'large'``.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0, "negligible"
    delta = float(np.sign(x[:, None] - y[None, :]).sum() / (nx * ny))
    mag = abs(delta)
    if mag < 0.147:
        label = "negligible"
    elif mag < 0.33:
        label = "small"
    elif mag < 0.474:
        label = "medium"
    else:
        label = "large"
    return delta, label


def bootstrap_ci(
    data: np.ndarray, n_boot: int = 10000, ci: float = 0.95, seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Returns (lower, upper) bounds.
    """
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(data, size=len(data), replace=True).mean() for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))
