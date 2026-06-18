"""
3-metric benchmark evaluation framework + JAS combined metric.

Focus: jerk reduction conditioned on task completion.

Metrics:
  M1  peak_jerk    – max |d³pos/dt³|  (smoothness — primary objective)
  M2  endpoint_err – |final_pos - 0|  (accuracy — validity gate)
  M3  osc_amp      – RMS oscillation after stop (vibration)
  JAS              – validity-gated combined smoothness-accuracy score
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_jerk_series(positions: np.ndarray, dt: float = 0.02) -> float:
    """Numerical 3rd derivative of the position array; returns peak |jerk|."""
    if len(positions) < 4:
        return 0.0
    vel = np.gradient(positions, dt)
    acc = np.gradient(vel, dt)
    jerk = np.gradient(acc, dt)
    return float(np.max(np.abs(jerk)))


def oscillation_amplitude(
    positions: np.ndarray,
    stop_idx: int = -50,
    window: int = 50,
) -> float:
    """RMS of position deviation in the tail window after the stop index."""
    if stop_idx < 0:
        stop_idx = max(0, len(positions) + stop_idx)
    tail = positions[stop_idx: stop_idx + window]
    if len(tail) == 0:
        return 0.0
    return float(np.sqrt(np.mean((tail - tail.mean()) ** 2)))


def log_episode(
    controller_name: str,
    positions: list[float] | np.ndarray,
) -> dict:
    """
    Compute the three jerk-focused metrics for a single episode.

    Returns a flat dict suitable for pandas DataFrame construction.
    """
    pos = np.asarray(positions, dtype=np.float64)
    return {
        "controller":   controller_name,
        "peak_jerk":    compute_jerk_series(pos),
        "endpoint_err": float(abs(pos[-1])) if len(pos) > 0 else float("nan"),
        "osc_amp":      oscillation_amplitude(pos, stop_idx=-50),
    }


# ---------------------------------------------------------------------------
# JAS: Jerk-Accuracy Score
# ---------------------------------------------------------------------------

def jerk_accuracy_score(
    peak_jerk: float,
    endpoint_err: float,
    jerk_baseline: float,
    err_threshold: float = 0.15,
) -> float:
    """
    Validity-gated combined smoothness-accuracy metric.

    Motivation
    ----------
    Raw peak_jerk alone rewards controllers that never apply force (zero jerk,
    infinite endpoint error). JAS conditions the smoothness reward on whether
    the controller actually completed the task within an acceptable accuracy
    bound.

    Formula
    -------
    jerk_score = jerk_baseline / peak_jerk     (>1 = smoother than baseline)

    If endpoint_err <= err_threshold:
        JAS = jerk_score                        (full credit)
    Else:
        JAS = jerk_score * (err_threshold / endpoint_err)   (penalised)

    Parameters
    ----------
    peak_jerk     : mean peak jerk of the controller (m/s^3)
    endpoint_err  : mean endpoint error of the controller (m)
    jerk_baseline : peak_jerk of the reference controller (e.g. quintic mean)
    err_threshold : max acceptable endpoint error before penalty is applied.
                    Suggested values:
                      0.05m  strict (2x quintic baseline ~0.015m)
                      0.10m  moderate
                      0.15m  lenient (matches SNN+STDP actual error)

    Returns
    -------
    float: JAS score. 1.0 = exactly matches the jerk baseline. >1 = better.
    """
    jerk_score = jerk_baseline / peak_jerk
    if endpoint_err > err_threshold:
        return jerk_score * (err_threshold / endpoint_err)
    return jerk_score
