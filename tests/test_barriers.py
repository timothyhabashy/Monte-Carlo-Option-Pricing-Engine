import numpy as np
import pytest

from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    Market,
    barrier_closed_form,
    black_scholes,
    price_mc,
)
from tests.checks import assert_within_se


def test_knocked_out_at_spot_is_zero(market: Market, strike: float) -> None:
    result = price_mc(
        market,
        Contract(strike, ContractKind.up_and_out_call, B=market.S),
        steps=40,
        trial_count=2_000,
        seed=0,
    )
    assert result.price == 0.0
    assert result.stderr == 0.0


def test_barrier_below_spot_is_zero(market: Market, strike: float) -> None:
    result = price_mc(
        market,
        Contract(strike, ContractKind.up_and_out_call, B=90.0),
        steps=40,
        trial_count=1_000,
        seed=0,
    )
    assert result.price == 0.0


def test_huge_barrier_matches_european_call(market: Market, strike: float) -> None:
    n = 20_000
    result = price_mc(
        market,
        Contract(strike, ContractKind.up_and_out_call, B=1e12),
        steps=50,
        trial_count=n,
        seed=0,
    )
    bs = black_scholes(market, Contract(strike, ContractKind.euro_call))
    assert_within_se(result.price, bs, result.stderr)


def test_missing_barrier_rejected_at_contract(strike: float) -> None:
    with pytest.raises(ValueError, match="Barrier B required"):
        Contract(strike, ContractKind.up_and_out_call)


def test_continuous_up_and_out_not_above_discrete(market: Market, strike: float) -> None:
    n = 8_000
    steps = 50
    discrete = Contract(
        strike, ContractKind.up_and_out_call, B=115.0, monitoring="discrete"
    )
    continuous = Contract(
        strike, ContractKind.up_and_out_call, B=115.0, monitoring="continuous"
    )
    kwargs = dict(
        steps=steps,
        trial_count=n,
        seed=0,
        control_variate=False,
        antithetic=True,
    )
    disc = price_mc(market, discrete, **kwargs)
    cont = price_mc(market, continuous, **kwargs)
    assert cont.price <= disc.price + 1e-12


def test_far_down_barrier_call_is_vanilla() -> None:
    market = Market(S=100.0, T=1.0, r=0.08, q=0.04, sigma=0.25)
    vanilla = black_scholes(market, Contract(90.0, ContractKind.euro_call))
    far = barrier_closed_form(
        market, Contract(90.0, ContractKind.down_and_out_call, B=1.0)
    )
    assert far == pytest.approx(vanilla, rel=1e-10)


def test_barrier_in_out_parity_closed_form(market: Market, strike: float) -> None:
    vanilla = black_scholes(market, Contract(strike, ContractKind.euro_call))
    uo = barrier_closed_form(
        market, Contract(strike, ContractKind.up_and_out_call, B=115.0)
    )
    ui = barrier_closed_form(
        market, Contract(strike, ContractKind.up_and_in_call, B=115.0)
    )
    assert uo + ui == pytest.approx(vanilla, rel=1e-10)


def test_continuous_mc_near_reiner_rubinstein(market: Market, strike: float) -> None:
    contract = Contract(
        strike, ContractKind.up_and_out_call, B=115.0, monitoring="continuous"
    )
    closed = barrier_closed_form(market, contract)
    result = price_mc(
        market,
        contract,
        steps=200,
        trial_count=20_000,
        seed=0,
        control_variate=False,
    )
    assert abs(result.price - closed) < 0.05
