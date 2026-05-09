# -*- coding: utf-8 -*-
"""Epsilon-decay schedules for constraint annealing.

All schedules satisfy ``epsilon(0) = eps_max`` and ``epsilon(T) ~ 0``.
They control how quickly the feasibility boundary relaxation decays
during an optimization run.

Functions
---------
epsilon_pow : Power-law decay (default, recommended in Paper B).
epsilon_lin : Linear decay.
epsilon_exp : Exponential decay.
"""

import math


def epsilon_pow(t, T, eps_max=0.2, cp=5):
    """Power-law decay: ``eps(t) = eps_max * (1 - t/T)^cp``.

    Parameters
    ----------
    t : int or float
        Current number of function evaluations.
    T : int or float
        Maximum number of function evaluations.
    eps_max : float
        Initial epsilon value at ``t=0``.
    cp : int or float
        Exponent controlling decay speed.

    Returns
    -------
    float
        Current epsilon value.
    """
    return eps_max * max(0.0, 1.0 - t / T) ** cp


def epsilon_lin(t, T, eps_max=0.2):
    """Linear decay: ``eps(t) = eps_max * (1 - t/T)``.

    Parameters
    ----------
    t : int or float
        Current number of function evaluations.
    T : int or float
        Maximum number of function evaluations.
    eps_max : float
        Initial epsilon value at ``t=0``.

    Returns
    -------
    float
        Current epsilon value.
    """
    return eps_max * max(0.0, 1.0 - t / T)


def epsilon_exp(t, T, eps_max=0.2, ce=5):
    """Exponential decay: ``eps(t) = eps_max * exp(-ce * t/T)``.

    Parameters
    ----------
    t : int or float
        Current number of function evaluations.
    T : int or float
        Maximum number of function evaluations.
    eps_max : float
        Initial epsilon value at ``t=0``.
    ce : int or float
        Exponential constant controlling decay speed.

    Returns
    -------
    float
        Current epsilon value.
    """
    return eps_max * math.exp(-ce * t / T)


# Default schedule (Paper B recommendation)
DEFAULT_SCHEDULE = epsilon_pow
DEFAULT_EPS_MAX = 0.2
DEFAULT_CP = 5
