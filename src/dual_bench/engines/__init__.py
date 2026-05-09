# -*- coding: utf-8 -*-
"""Optimisation engine adapters.

Adapters convert dual-bench formulations into solver-specific problem
representations.  Each engine is optional and requires its respective
backend to be installed:

- ``pymoo_engine``: requires ``pip install dual-bench[pymoo]``
- ``platypus_engine``: requires ``pip install dual-bench[platypus]``

The :class:`Solution` dataclass is always available for working with results.
"""

from dual_bench.engines.base import Solution

__all__ = ["Solution"]
