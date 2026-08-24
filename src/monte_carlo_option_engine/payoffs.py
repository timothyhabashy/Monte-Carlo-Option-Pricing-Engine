"""Vectorized payoff functions.

The arithmetic Asian average uses ``path[1:].mean(axis=0)`` — every simulated
grid point after ``t=0``, excluding the spot ``S0``. Barrier contracts are
monitored on the simulated grid; continuous monitoring is approximated by a
Broadie–Glasserman–Kou shift of ``B`` before that discrete check.
"""

from __future__ import annotations

import numpy as np


def payoff_european_call(spot: np.ndarray, strike: float) -> np.ndarray:
    return np.maximum(np.asarray(spot, dtype=float) - strike, 0.0)


def payoff_european_put(spot: np.ndarray, strike: float) -> np.ndarray:
    return np.maximum(strike - np.asarray(spot, dtype=float), 0.0)


def payoff_digital_call(spot: np.ndarray, strike: float, payout: float = 1.0) -> np.ndarray:
    return payout * (np.asarray(spot, dtype=float) > strike).astype(float)


def payoff_digital_put(spot: np.ndarray, strike: float, payout: float = 1.0) -> np.ndarray:
    return payout * (np.asarray(spot, dtype=float) < strike).astype(float)


def payoff_asian_arithmetic_call(path: np.ndarray, strike: float) -> np.ndarray:
    """Arithmetic average of ``path[1:]`` vs strike, call payoff."""

    path = np.asarray(path, dtype=float)
    avg = path[1:].mean(axis=0)
    return np.maximum(avg - strike, 0.0)


def payoff_asian_arithmetic_put(path: np.ndarray, strike: float) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    avg = path[1:].mean(axis=0)
    return np.maximum(strike - avg, 0.0)


def payoff_asian_geometric_call(path: np.ndarray, strike: float) -> np.ndarray:
    """Call on the geometric mean of ``path[1:]``."""

    path = np.asarray(path, dtype=float)
    geo = np.exp(np.log(path[1:]).mean(axis=0))
    return np.maximum(geo - strike, 0.0)


def payoff_asian_geometric_put(path: np.ndarray, strike: float) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    geo = np.exp(np.log(path[1:]).mean(axis=0))
    return np.maximum(strike - geo, 0.0)


def payoff_lookback_fixed_call(path: np.ndarray, strike: float) -> np.ndarray:
    """Fixed-strike lookback call: ``max(max(S) - K, 0)``."""

    path = np.asarray(path, dtype=float)
    return np.maximum(path.max(axis=0) - strike, 0.0)


def _vanilla_at_expiry(path: np.ndarray, strike: float, is_call: bool) -> np.ndarray:
    spot = np.asarray(path, dtype=float)[-1]
    if is_call:
        return np.maximum(spot - strike, 0.0)
    return np.maximum(strike - spot, 0.0)


def payoff_barrier(
    path: np.ndarray,
    strike: float,
    barrier: float,
    *,
    up: bool,
    knock_out: bool,
    is_call: bool,
) -> np.ndarray:
    """Discrete knock-in/out call or put."""

    path = np.asarray(path, dtype=float)
    hit = path.max(axis=0) >= barrier if up else path.min(axis=0) <= barrier
    vanilla = _vanilla_at_expiry(path, strike, is_call)
    if knock_out:
        return np.where(hit, 0.0, vanilla)
    return np.where(hit, vanilla, 0.0)


def payoff_up_and_out_call(path: np.ndarray, strike: float, barrier: float) -> np.ndarray:
    """Discrete up-and-out call: zero if any grid point is ``>= barrier``."""

    return payoff_barrier(path, strike, barrier, up=True, knock_out=True, is_call=True)
