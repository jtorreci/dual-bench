# -*- coding: utf-8 -*-
"""Performance metrics for constrained optimization results."""
from __future__ import annotations

import numpy as np


def hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2-D hypervolume indicator (dominated area, minimization).

    Parameters
    ----------
    points : (n, 2) array of objective values.
    ref : (2,) reference point.
    """
    pts = np.asarray(points, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (n, 2) array.")
    if ref.shape != (2,):
        raise ValueError("ref must be a (2,) array.")
    # Keep only points strictly dominated by the reference
    mask = (pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])
    pts = pts[mask]
    if len(pts) == 0:
        return 0.0
    # Sort by first objective, sweep to compute dominated area
    pts = pts[np.argsort(pts[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in pts:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return float(hv)


def delta_f_star(f_best: float, f_star: float) -> float:
    """Relative distance to known optimum: ``(f_best - f*) / |f*|``.

    Returns ``NaN`` when *f_star* is zero.
    """
    if f_star == 0.0:
        return float("nan")
    return (f_best - f_star) / abs(f_star)


def feasibility_rate(ratios: np.ndarray) -> float:
    """Fraction of solutions where all d/c ratios <= 1.0.

    Parameters
    ----------
    ratios : (n_solutions, n_ratios) array of demand-to-capacity ratios.

    Returns float in [0, 1].
    """
    ratios = np.asarray(ratios, dtype=float)
    if ratios.ndim == 1:
        ratios = ratios.reshape(1, -1)
    return float(np.all(ratios <= 1.0, axis=1).mean())


def igd(approx: np.ndarray, reference: np.ndarray) -> float:
    """Inverted Generational Distance (lower is better).

    Mean minimum Euclidean distance from each reference point to the
    nearest point in the approximation set.
    """
    approx = np.asarray(approx, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if approx.ndim == 1:
        approx = approx.reshape(1, -1)
    if reference.ndim == 1:
        reference = reference.reshape(1, -1)
    dists = np.linalg.norm(reference[:, None, :] - approx[None, :, :], axis=2)
    return float(np.mean(np.min(dists, axis=1)))


def spread(points: np.ndarray) -> float:
    """Spread indicator: max Euclidean distance between any two points."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if len(pts) < 2:
        return 0.0
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    return float(np.max(dists))
