"""Analytic barrier prices (Reiner–Rubinstein) under GBM. Rebate is 0."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from monte_carlo_option_engine.black_scholes import black_scholes
from monte_carlo_option_engine.types import (
    BARRIER_KINDS,
    Contract,
    ContractKind,
    Market,
)


def _terms(
    spot: float,
    strike: float,
    barrier: float,
    tenor: float,
    rate: float,
    cost: float,
    sigma: float,
    phi: float,
    eta: float,
) -> tuple[float, float, float, float]:
    """Haug A, B, C, D (rebate terms E, F omitted). ``cost`` is r − q."""

    vol_sqrt = sigma * np.sqrt(tenor)
    mu = (cost - 0.5 * sigma**2) / sigma**2
    x1 = np.log(spot / strike) / vol_sqrt + (1.0 + mu) * vol_sqrt
    x2 = np.log(spot / barrier) / vol_sqrt + (1.0 + mu) * vol_sqrt
    y1 = np.log(barrier**2 / (spot * strike)) / vol_sqrt + (1.0 + mu) * vol_sqrt
    y2 = np.log(barrier / spot) / vol_sqrt + (1.0 + mu) * vol_sqrt
    hs = barrier / spot
    disc_s = np.exp((cost - rate) * tenor)
    disc_k = np.exp(-rate * tenor)
    pow_s = hs ** (2.0 * (mu + 1.0))
    pow_k = hs ** (2.0 * mu)

    term_a = phi * spot * disc_s * norm.cdf(phi * x1) - phi * strike * disc_k * norm.cdf(
        phi * x1 - phi * vol_sqrt
    )
    term_b = phi * spot * disc_s * norm.cdf(phi * x2) - phi * strike * disc_k * norm.cdf(
        phi * x2 - phi * vol_sqrt
    )
    term_c = phi * spot * disc_s * pow_s * norm.cdf(eta * y1) - phi * strike * disc_k * pow_k * norm.cdf(
        eta * y1 - eta * vol_sqrt
    )
    term_d = phi * spot * disc_s * pow_s * norm.cdf(eta * y2) - phi * strike * disc_k * pow_k * norm.cdf(
        eta * y2 - eta * vol_sqrt
    )
    return float(term_a), float(term_b), float(term_c), float(term_d)


def _knock_out(
    spot: float,
    strike: float,
    barrier: float,
    tenor: float,
    rate: float,
    div: float,
    sigma: float,
    *,
    up: bool,
    is_call: bool,
) -> float:
    if tenor <= 0:
        hit = spot >= barrier if up else spot <= barrier
        if hit:
            return 0.0
        if is_call:
            return float(max(spot - strike, 0.0))
        return float(max(strike - spot, 0.0))
    if up and spot >= barrier:
        return 0.0
    if (not up) and spot <= barrier:
        return 0.0

    phi = 1.0 if is_call else -1.0
    eta = -1.0 if up else 1.0
    cost = rate - div
    a, b, c, d = _terms(spot, strike, barrier, tenor, rate, cost, sigma, phi, eta)
    k_gt_h = strike > barrier
    if is_call and not up:
        return a - c if k_gt_h else b - d
    if is_call and up:
        return 0.0 if k_gt_h else a - b + c - d
    if (not is_call) and not up:
        return a - b + c - d if k_gt_h else 0.0
    return b - d if k_gt_h else a - c


def barrier_closed_form(market: Market, contract: Contract) -> float:
    """Continuous single-barrier price (Reiner–Rubinstein), rebate 0."""

    kind = ContractKind(contract.kind)
    if kind not in BARRIER_KINDS:
        raise ValueError(f"barrier_closed_form does not support {kind}")
    if contract.B is None:
        raise ValueError(f"Barrier B required for {kind}")
    name = str(kind)
    up = "up_and" in name
    knock_out = "_out_" in name
    is_call = name.endswith("_call")
    vanilla_kind = ContractKind.euro_call if is_call else ContractKind.euro_put
    vanilla = black_scholes(market, Contract(contract.K, vanilla_kind))
    knocked_out = _knock_out(
        market.S,
        contract.K,
        contract.B,
        market.T,
        market.r,
        market.q,
        market.sigma,
        up=up,
        is_call=is_call,
    )
    if knock_out:
        return max(knocked_out, 0.0)
    return max(vanilla - knocked_out, 0.0)
