# -*- coding: utf-8 -*-
"""Analysis pipeline -- statistical tests, performance metrics, and LaTeX tables."""

from dual_bench.analysis.metrics import (
    delta_f_star,
    feasibility_rate,
    hypervolume_2d,
    igd,
    spread,
)
from dual_bench.analysis.statistics import (
    bootstrap_ci,
    cliff_delta,
    friedman_test,
    kruskal_wallis,
    nemenyi_posthoc,
)
from dual_bench.analysis.tables import summary_table, to_latex

__all__ = [
    # statistics
    "friedman_test",
    "nemenyi_posthoc",
    "kruskal_wallis",
    "cliff_delta",
    "bootstrap_ci",
    # metrics
    "hypervolume_2d",
    "delta_f_star",
    "feasibility_rate",
    "igd",
    "spread",
    # tables
    "to_latex",
    "summary_table",
]
