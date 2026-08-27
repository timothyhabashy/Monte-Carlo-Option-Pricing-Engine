"""Cox–Ross–Rubinstein binomial tree for American puts under GBM."""

from __future__ import annotations

import numpy as np

from monte_carlo_option_engine.gbm import _positive_int
from monte_carlo_option_engine.types import Market


def crr_american_put(market: Market, strike: float, steps: int = 200) -> float:
    """CRR American put. Deterministic; ``steps`` time intervals."""

    if strike <= 0:
        raise ValueError("strike must be positive")
    steps = _positive_int(steps, "steps")
    if market.T <= 0:
        return float(max(strike - market.S, 0.0))
    dt = market.T / steps
    u = float(np.exp(market.sigma * np.sqrt(dt)))
    d = 1.0 / u
    growth = float(np.exp((market.r - market.q) * dt))
    p_up = (growth - d) / (u - d)
    if not 0.0 < p_up < 1.0:
        raise ValueError("CRR risk-neutral probability is outside (0, 1)")
    disc = float(np.exp(-market.r * dt))
    # Node i at expiry: i up-moves, (steps - i) down-moves.
    i = np.arange(steps + 1, dtype=float)
    spots = market.S * (u**i) * (d ** (steps - i))
    values = np.maximum(strike - spots, 0.0)
    for n in range(steps - 1, -1, -1):
        continuation = disc * (p_up * values[1:] + (1.0 - p_up) * values[:-1])
        i_n = np.arange(n + 1, dtype=float)
        spots_n = market.S * (u**i_n) * (d ** (n - i_n))
        values = np.maximum(strike - spots_n, continuation)
    return float(values[0])
