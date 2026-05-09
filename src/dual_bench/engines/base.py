# -*- coding: utf-8 -*-
"""Solution dataclass shared by all engine adapters.

A :class:`Solution` holds the result of evaluating a single candidate
in the decision space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Solution:
    """A single evaluated solution returned by an engine.

    Attributes
    ----------
    x : list of float
        Decision variable values.
    objectives : list of float
        Objective function values.
    constraints : list of float
        Constraint values (empty if unconstrained).
    feasible : bool
        Whether all constraints are satisfied.
    """

    x: List[float]
    objectives: List[float]
    constraints: List[float] = field(default_factory=list)
    feasible: bool = True
