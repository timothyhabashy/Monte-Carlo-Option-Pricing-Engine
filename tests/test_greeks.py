import numpy as np
import pytest
from scipy.stats import norm

from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    Market,
    greeks,
)
from tests.checks import assert_within_se


def _d1_d2(market: Market, strike: float) -> tuple[float, float]:
    vol_sqrt = market.sigma * np.sqrt(market.T)
    mu = market.r - market.q
    d1 = (np.log(market.S / strike) + (mu + 0.5 * market.sigma**2) * market.T) / vol_sqrt
    d2 = d1 - vol_sqrt
    return float(d1), float(d2)


def bs_call_delta(market: Market, strike: float) -> float:
    d1, _ = _d1_d2(market, strike)
    return float(np.exp(-market.q * market.T) * norm.cdf(d1))


def bs_call_vega(market: Market, strike: float) -> float:
    d1, _ = _d1_d2(market, strike)
    return float(
        market.S * np.exp(-market.q * market.T) * norm.pdf(d1) * np.sqrt(market.T)
    )


def bs_digital_delta(market: Market, strike: float, payout: float = 1.0) -> float:
    _, d2 = _d1_d2(market, strike)
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
    assert_within_se(result.delta, bs_call_delta(market, strike), result.stderr_delta)


def test_pathwise_vega_vs_bs(market: Market, strike: float) -> None:
    result = greeks(
        market,
        Contract(strike, ContractKind.euro_call),
        method="pathwise",
        trial_count=50_000,
        seed=0,
    )
    assert_within_se(result.vega, bs_call_vega(market, strike), result.stderr_vega)


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
    assert result.delta == pytest.approx(bs_call_delta(market, strike), abs=5e-4)
