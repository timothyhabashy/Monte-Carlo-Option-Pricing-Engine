"""Delta / vega / gamma / theta / rho via bump, pathwise, or likelihood ratio."""

from __future__ import annotations

from typing import Literal

import numpy as np

from monte_carlo_option_engine.gbm import paths_from_shocks, terminal_from_shocks
from monte_carlo_option_engine.payoffs import (
    payoff_digital_call,
    payoff_digital_put,
    payoff_european_call,
    payoff_european_put,
)
from monte_carlo_option_engine.pricer import _resolve_rng, price_mc
from monte_carlo_option_engine.types import (
    ASIAN_KINDS,
    BARRIER_KINDS,
    TERMINAL_KINDS,
    Contract,
    ContractKind,
    GreeksResult,
    Market,
)

GreeksMethod = Literal["bump", "pathwise", "likelihood_ratio"]
_NAN = float("nan")


def _copy_market(market: Market, **kwargs: float) -> Market:
    data = dict(S=market.S, T=market.T, r=market.r, q=market.q, sigma=market.sigma)
    data.update(kwargs)
    return Market(**data)


def _terminal_payoff(kind: ContractKind, spots: np.ndarray, contract: Contract) -> np.ndarray:
    if kind is ContractKind.euro_call:
        return payoff_european_call(spots, contract.K)
    if kind is ContractKind.euro_put:
        return payoff_european_put(spots, contract.K)
    if kind is ContractKind.digital_call:
        return payoff_digital_call(spots, contract.K, contract.Q)
    if kind is ContractKind.digital_put:
        return payoff_digital_put(spots, contract.K, contract.Q)
    raise ValueError(f"unsupported kind for terminal greeks: {kind}")


def _bump_greeks(
    market: Market,
    contract: Contract,
    trial_count: int,
    seed: int,
    steps: int,
    **price_kwargs: object,
) -> GreeksResult:
    if market.T <= 0:
        kind = ContractKind(contract.kind)
        s, k = market.S, contract.K
        if kind is ContractKind.euro_call:
            delta = 1.0 if s > k else 0.0
        elif kind is ContractKind.euro_put:
            delta = -1.0 if s < k else 0.0
        elif kind is ContractKind.digital_call:
            delta = 0.0
        else:
            delta = _NAN
        return GreeksResult(delta, _NAN, 0.0, 0.0, 0.0, method="bump")

    ds = max(1e-4 * market.S, 1e-8)
    d_sigma = min(1e-4, 0.25 * market.sigma)
    dr = 1e-4
    d_t = min(1e-4, 0.25 * market.T) if market.T > 0 else 1e-4

    def px(mkt: Market) -> float:
        return price_mc(
            mkt,
            contract,
            trial_count=trial_count,
            steps=steps,
            seed=seed,
            **price_kwargs,
        ).price

    v0 = px(market)
    v_up_s = px(_copy_market(market, S=market.S + ds))
    v_dn_s = px(_copy_market(market, S=market.S - ds))
    v_up_sig = px(_copy_market(market, sigma=market.sigma + d_sigma))
    v_dn_sig = px(_copy_market(market, sigma=market.sigma - d_sigma))
    v_up_r = px(_copy_market(market, r=market.r + dr))
    v_dn_r = px(_copy_market(market, r=market.r - dr))
    v_dn_t = px(_copy_market(market, T=market.T - d_t))
    v_up_t = px(_copy_market(market, T=market.T + d_t))

    delta = (v_up_s - v_dn_s) / (2.0 * ds)
    gamma = (v_up_s - 2.0 * v0 + v_dn_s) / (ds**2)
    vega = (v_up_sig - v_dn_sig) / (2.0 * d_sigma)
    rho = (v_up_r - v_dn_r) / (2.0 * dr)
    # Calendar theta: dV/dt = -dV/dT
    theta = -(v_up_t - v_dn_t) / (2.0 * d_t)
    return GreeksResult(delta, gamma, vega, theta, rho, method="bump")


def _pathwise_greeks(
    market: Market,
    contract: Contract,
    trial_count: int,
    rng: np.random.Generator,
    steps: int = 200,
) -> GreeksResult:
    kind = ContractKind(contract.kind)
    if kind in (ContractKind.digital_call, ContractKind.digital_put):
        raise ValueError(
            "pathwise greeks are not valid for digital payoffs; use likelihood_ratio"
        )
    if kind in BARRIER_KINDS:
        raise ValueError("pathwise greeks are not valid for barrier payoffs")
    if market.T <= 0:
        return _bump_greeks(market, contract, trial_count, seed=0, steps=1)

    disc = np.exp(-market.r * market.T)
    if kind in (ContractKind.euro_call, ContractKind.euro_put):
        z = rng.normal(size=trial_count)
        spots = terminal_from_shocks(market, z)
        w_t = np.sqrt(market.T) * z
        if kind is ContractKind.euro_call:
            itm = spots > contract.K
            sign = 1.0
        else:
            itm = spots < contract.K
            sign = -1.0
        d_pay = itm.astype(float)
        delta_paths = disc * sign * d_pay * spots / market.S
        vega_paths = disc * sign * d_pay * spots * (w_t - market.sigma * market.T)
        delta = float(delta_paths.mean())
        vega = float(vega_paths.mean())
        se_d = float(delta_paths.std(ddof=1) / np.sqrt(trial_count)) if trial_count > 1 else 0.0
        se_v = float(vega_paths.std(ddof=1) / np.sqrt(trial_count)) if trial_count > 1 else 0.0
        return GreeksResult(
            delta,
            _NAN,
            vega,
            _NAN,
            _NAN,
            stderr_delta=se_d,
            stderr_vega=se_v,
            method="pathwise",
        )

    steps = max(int(steps), 1)
    z = rng.normal(size=(steps, trial_count))
    paths = paths_from_shocks(market, z)
    if kind in ASIAN_KINDS:
        avg = paths[1:].mean(axis=0)
        itm = (avg > contract.K) if kind is ContractKind.asian_call else (avg < contract.K)
        sign = 1.0 if kind is ContractKind.asian_call else -1.0
        under = avg
    elif kind is ContractKind.lookback_call:
        under = paths.max(axis=0)
        itm = under > contract.K
        sign = 1.0
    else:
        raise ValueError(
            "pathwise greeks are implemented for europeans, arithmetic asians, "
            "and fixed-strike lookback calls"
        )
    delta_paths = disc * sign * itm.astype(float) * under / market.S
    delta = float(delta_paths.mean())
    se_d = float(delta_paths.std(ddof=1) / np.sqrt(trial_count)) if trial_count > 1 else 0.0
    return GreeksResult(
        delta,
        _NAN,
        _NAN,
        _NAN,
        _NAN,
        stderr_delta=se_d,
        method="pathwise",
    )


def _lr_greeks(
    market: Market,
    contract: Contract,
    trial_count: int,
    rng: np.random.Generator,
) -> GreeksResult:
    kind = ContractKind(contract.kind)
    if kind not in TERMINAL_KINDS:
        raise ValueError("likelihood-ratio greeks are implemented for terminal payoffs")
    if market.T <= 0:
        return _bump_greeks(market, contract, trial_count, seed=0, steps=1)

    z = rng.normal(size=trial_count)
    spots = terminal_from_shocks(market, z)
    disc = np.exp(-market.r * market.T)
    pay = disc * _terminal_payoff(kind, spots, contract)
    vol_sqrt = market.sigma * np.sqrt(market.T)
    score_s = z / (market.S * vol_sqrt)
    # score_σ = -1/σ + Z²/σ - √T Z   (density of log S_T)
    score_sigma = -1.0 / market.sigma + (z**2) / market.sigma - np.sqrt(market.T) * z
    delta_paths = pay * score_s
    vega_paths = pay * score_sigma
    delta = float(delta_paths.mean())
    vega = float(vega_paths.mean())
    n = trial_count
    se_d = float(delta_paths.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    se_v = float(vega_paths.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return GreeksResult(
        delta,
        _NAN,
        vega,
        _NAN,
        _NAN,
        stderr_delta=se_d,
        stderr_vega=se_v,
        method="likelihood_ratio",
    )


def greeks(
    market: Market,
    contract: Contract,
    method: GreeksMethod = "bump",
    trial_count: int = 50_000,
    steps: int = 200,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    **price_kwargs: object,
) -> GreeksResult:
    """Estimate Greeks.

    ``bump`` uses common-random-number finite differences through ``price_mc``.
    ``pathwise`` supports European calls/puts (delta and vega) and, on GBM,
    arithmetic Asians and fixed-strike lookback calls (delta).
    ``likelihood_ratio`` supports terminal payoffs and is the method to use
    for digitals.
    """

    if method not in ("bump", "pathwise", "likelihood_ratio"):
        raise ValueError("method must be bump, pathwise, or likelihood_ratio")
    rng, used_seed = _resolve_rng(rng, seed)
    if method == "bump":
        if used_seed is None:
            used_seed = int(rng.integers(0, 2**31 - 1))
        bump_kwargs = dict(price_kwargs)
        bump_kwargs.pop("seed", None)
        bump_kwargs.pop("rng", None)
        bump_kwargs.pop("trial_count", None)
        bump_kwargs.pop("steps", None)
        return _bump_greeks(
            market, contract, trial_count, used_seed, steps, **bump_kwargs
        )
    if method == "pathwise":
        return _pathwise_greeks(market, contract, trial_count, rng, steps=steps)
    return _lr_greeks(market, contract, trial_count, rng)
