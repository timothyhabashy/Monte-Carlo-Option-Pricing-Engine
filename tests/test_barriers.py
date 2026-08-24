import pytest

from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    Market,
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
