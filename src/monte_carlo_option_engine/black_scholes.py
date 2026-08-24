"""Closed-form Black–Scholes prices used as checks (and later control variates)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from monte_carlo_option_engine.types import Contract, ContractKind, Market


def _d1_d2(market: Market, strike: float) -> tuple[float, float]:
    vol_sqrt = market.sigma * np.sqrt(market.T)
    mu = market.r - market.q
    d1 = (np.log(market.S / strike) + (mu + 0.5 * market.sigma**2) * market.T) / vol_sqrt
    d2 = d1 - vol_sqrt
    return float(d1), float(d2)


def black_scholes(market: Market, contract: Contract) -> float:
    """Black–Scholes price for euro call/put and cash-or-nothing digital call."""

    kind = ContractKind(contract.kind)
    s = market.S
    t = market.T
    k = contract.K
    r = market.r
    q = market.q

    if t <= 0:
        if kind is ContractKind.euro_call:
            return float(max(s - k, 0.0))
        if kind is ContractKind.euro_put:
            return float(max(k - s, 0.0))
        if kind is ContractKind.digital_call:
            return float(contract.Q if s > k else 0.0)
        if kind is ContractKind.digital_put:
            return float(contract.Q if s < k else 0.0)
        raise ValueError(
            "Black–Scholes only supports euro_call, euro_put, digital_call, digital_put"
        )

    d1, d2 = _d1_d2(market, k)

    if kind is ContractKind.euro_call:
        return float(
            s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
        )
    if kind is ContractKind.euro_put:
        return float(
            k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-q * t) * norm.cdf(-d1)
        )
    if kind is ContractKind.digital_call:
        return float(contract.Q * np.exp(-r * t) * norm.cdf(d2))
    if kind is ContractKind.digital_put:
        return float(contract.Q * np.exp(-r * t) * norm.cdf(-d2))

    raise ValueError(
        "Black–Scholes only supports euro_call, euro_put, digital_call, digital_put"
    )


def geometric_asian_call(market: Market, contract: Contract, steps: int) -> float:
    """Closed-form call on the discrete geometric mean of ``path[1:]``.

    Averaging dates are ``dt, 2dt, ..., T`` with ``dt = T / steps``, matching
    ``payoff_asian_geometric_call``.
    """

    n = int(steps)
    if n <= 0:
        raise ValueError("steps must be positive")
    t = market.T
    k = contract.K
    if t <= 0:
        return float(max(market.S - k, 0.0))

    t_bar = t * (n + 1) / (2 * n)
    # Var((1/n) Σ W_{t_i}) → T/3; note n² not n³ in the denominator.
    var_log = market.sigma**2 * t * (n + 1) * (2 * n + 1) / (6.0 * n**2)
    mu = market.r - market.q
    mean_log = np.log(market.S) + (mu - 0.5 * market.sigma**2) * t_bar
    sigma_log = float(np.sqrt(var_log))
    d2 = (mean_log - np.log(k)) / sigma_log
    d1 = d2 + sigma_log
    expected_g = float(np.exp(mean_log + 0.5 * var_log))
    return float(
        np.exp(-market.r * t) * (expected_g * norm.cdf(d1) - k * norm.cdf(d2))
    )


def geometric_asian_put(market: Market, contract: Contract, steps: int) -> float:
    """Closed-form put on the discrete geometric mean of ``path[1:]``."""

    n = int(steps)
    if n <= 0:
        raise ValueError("steps must be positive")
    t = market.T
    k = contract.K
    if t <= 0:
        return float(max(k - market.S, 0.0))

    t_bar = t * (n + 1) / (2 * n)
    var_log = market.sigma**2 * t * (n + 1) * (2 * n + 1) / (6.0 * n**2)
    mu = market.r - market.q
    mean_log = np.log(market.S) + (mu - 0.5 * market.sigma**2) * t_bar
    sigma_log = float(np.sqrt(var_log))
    d2 = (mean_log - np.log(k)) / sigma_log
    d1 = d2 + sigma_log
    expected_g = float(np.exp(mean_log + 0.5 * var_log))
    return float(
        np.exp(-market.r * t) * (k * norm.cdf(-d2) - expected_g * norm.cdf(-d1))
    )
