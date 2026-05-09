# -*- coding: utf-8 -*-
"""Reference data for dual-bench benchmark problems.

Submodules
----------
:mod:`~dual_bench.references.optima`
    Known optimal values and problem metadata.
:mod:`~dual_bench.references.paper_a`
    Summary statistics from Paper A (3,000 runs on 5 engineering benchmarks).

Quick access
------------
>>> import dual_bench.references as ref
>>> ref.optimum("Spring")
0.012665
>>> ref.paper_a_results("Spring", formulation="2OB-REST")
[{'problem': 'Spring', 'formulation': '2OB-REST', ...}, ...]
"""

from dual_bench.references.optima import info, list_problems, optimum
from dual_bench.references.paper_a import (
    by_algorithm as paper_a_by_algorithm,
)
from dual_bench.references.paper_a import (
    by_formulation as paper_a_by_formulation,
)
from dual_bench.references.paper_a import (
    experimental_setup as paper_a_setup,
)


def paper_a_results(problem=None, formulation=None, algorithm=None):
    """Return Paper A reference results.

    When *algorithm* is given, returns per-algorithm data (n=20 per cell).
    Otherwise returns per-formulation data pooled across algorithms (n=120).

    Parameters
    ----------
    problem, formulation : str, optional
        Filter by problem or formulation name.
    algorithm : str, optional
        If given, returns per-algorithm granularity.

    Returns
    -------
    list of dict
    """
    if algorithm is not None:
        return paper_a_by_algorithm(problem, formulation, algorithm)
    return paper_a_by_formulation(problem, formulation)


__all__ = [
    "optimum", "info", "list_problems",
    "paper_a_results", "paper_a_by_algorithm", "paper_a_by_formulation",
    "paper_a_setup",
]
