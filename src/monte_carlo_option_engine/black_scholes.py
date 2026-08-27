"""Closed-form Black–Scholes prices used as checks (and later control variates)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from monte_carlo_option_engine.types import Contract, ContractKind, GreeksResult, Market


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


def black_scholes_greeks(market: Market, contract: Contract) -> GreeksResult:
    """Black–Scholes Δ, Γ, ν, θ (calendar), and ρ for European calls and puts."""

    kind = ContractKind(contract.kind)
    if kind not in (ContractKind.euro_call, ContractKind.euro_put):
        raise ValueError("black_scholes_greeks supports euro_call and euro_put")
    s, t, k, r, q, sigma = (
        market.S,
        market.T,
        contract.K,
        market.r,
        market.q,
        market.sigma,
    )
    if t <= 0:
        if kind is ContractKind.euro_call:
            delta = 1.0 if s > k else 0.0
        else:
            delta = -1.0 if s < k else 0.0
        return GreeksResult(delta, 0.0, 0.0, 0.0, 0.0, method="closed_form")

    d1, d2 = _d1_d2(market, k)
    disc_q = float(np.exp(-q * t))
    disc_r = float(np.exp(-r * t))
    n_d1 = float(norm.pdf(d1))
    sqrt_t = float(np.sqrt(t))
    gamma = disc_q * n_d1 / (s * sigma * sqrt_t)
    vega = s * disc_q * n_d1 * sqrt_t
    theta_spot = -s * disc_q * n_d1 * sigma / (2.0 * sqrt_t)
    if kind is ContractKind.euro_call:
        delta = disc_q * float(norm.cdf(d1))
        theta = (
            theta_spot
            - r * k * disc_r * float(norm.cdf(d2))
            + q * s * disc_q * float(norm.cdf(d1))
        )
        rho = k * t * disc_r * float(norm.cdf(d2))
    else:
        delta = -disc_q * float(norm.cdf(-d1))
        theta = (
            theta_spot
            + r * k * disc_r * float(norm.cdf(-d2))
            - q * s * disc_q * float(norm.cdf(-d1))
        )
        rho = -k * t * disc_r * float(norm.cdf(-d2))
    return GreeksResult(delta, gamma, vega, theta, rho, method="closed_form")


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
