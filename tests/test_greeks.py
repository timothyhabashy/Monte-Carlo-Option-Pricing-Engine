import numpy as np
import pytest
from scipy.stats import norm

from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    Market,
    black_scholes_greeks,
    greeks,
)
from tests.checks import assert_within_se


def bs_digital_delta(market: Market, strike: float, payout: float = 1.0) -> float:
    vol_sqrt = market.sigma * np.sqrt(market.T)
    mu = market.r - market.q
    d2 = (
        np.log(market.S / strike) + (mu - 0.5 * market.sigma**2) * market.T
    ) / vol_sqrt
    return float(
        payout
        * np.exp(-market.r * market.T)
        * norm.pdf(d2)
        / (market.S * market.sigma * np.sqrt(market.T))
    )


def test_pathwise_delta_vs_bs(market: Market, strike: float) -> None:
    result = greeks(
        market,
        Contract(strike, ContractKind.euro_call),
        method="pathwise",
        trial_count=50_000,
        seed=0,
    )
    assert_within_se(result.delta, black_scholes_greeks(market, Contract(strike, ContractKind.euro_call)).delta, result.stderr_delta)


def test_pathwise_vega_vs_bs(market: Market, strike: float) -> None:
    result = greeks(
        market,
        Contract(strike, ContractKind.euro_call),
        method="pathwise",
        trial_count=50_000,
        seed=0,
    )
    assert_within_se(result.vega, black_scholes_greeks(market, Contract(strike, ContractKind.euro_call)).vega, result.stderr_vega)


def test_pathwise_delta_near_bump(market: Market, strike: float) -> None:
    contract = Contract(strike, ContractKind.euro_call)
    pw = greeks(market, contract, method="pathwise", trial_count=30_000, seed=0)
    bumped = greeks(
        market,
        contract,
        method="bump",
        trial_count=30_000,
        seed=0,
        control_variate=False,
        antithetic=True,
    )
    assert abs(pw.delta - bumped.delta) < 0.05


def test_digital_pathwise_rejected(market: Market, strike: float) -> None:
    with pytest.raises(ValueError, match="digital"):
        greeks(
            market,
            Contract(strike, ContractKind.digital_call),
            method="pathwise",
            trial_count=100,
            seed=0,
        )


def test_digital_lr_delta_vs_closed_form(market: Market, strike: float) -> None:
    result = greeks(
        market,
        Contract(strike, ContractKind.digital_call),
        method="likelihood_ratio",
        trial_count=50_000,
        seed=0,
    )
    assert_within_se(result.delta, bs_digital_delta(market, strike), result.stderr_delta)


def test_digital_lr_near_bump(market: Market, strike: float) -> None:
    contract = Contract(strike, ContractKind.digital_call)
    lr = greeks(market, contract, method="likelihood_ratio", trial_count=30_000, seed=0)
    bumped = greeks(
        market,
        contract,
        method="bump",
        trial_count=30_000,
        seed=0,
        control_variate=False,
        antithetic=True,
    )
    assert abs(lr.delta - bumped.delta) < 0.02


def test_bump_euro_with_cv_matches_bs_delta(market: Market, strike: float) -> None:
    # Default CV makes bump a finite difference of the exact BS price.
    result = greeks(
        market,
        Contract(strike, ContractKind.euro_call),
        method="bump",
        trial_count=1_000,
        seed=0,
    )
    assert result.delta == pytest.approx(
        black_scholes_greeks(market, Contract(strike, ContractKind.euro_call)).delta,
        abs=5e-4,
    )


def test_pathwise_barrier_rejected(market: Market, strike: float) -> None:
    with pytest.raises(ValueError, match="barrier"):
        greeks(
            market,
            Contract(strike, ContractKind.up_and_out_call, B=130.0),
            method="pathwise",
            trial_count=100,
            seed=0,
        )


def test_pathwise_asian_delta_near_bump(market: Market, strike: float) -> None:
    contract = Contract(strike, ContractKind.asian_call)
    kwargs = dict(trial_count=12_000, steps=20, seed=0)
    pw = greeks(market, contract, method="pathwise", **kwargs)
    bumped = greeks(
        market, contract, method="bump", control_variate=False, antithetic=True, **kwargs
    )
    assert abs(pw.delta - bumped.delta) < 0.2


def test_pathwise_lookback_delta_near_bump(market: Market, strike: float) -> None:
    contract = Contract(strike, ContractKind.lookback_call)
    kwargs = dict(trial_count=12_000, steps=20, seed=0)
    pw = greeks(market, contract, method="pathwise", **kwargs)
    bumped = greeks(
        market, contract, method="bump", control_variate=False, antithetic=True, **kwargs
    )
    assert abs(pw.delta - bumped.delta) < 0.25
