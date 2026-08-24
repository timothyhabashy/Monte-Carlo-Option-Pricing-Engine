import numpy as np
import pytest

from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    Market,
    black_scholes,
    price_mc,
)
from monte_carlo_option_engine.payoffs import (
    payoff_barrier,
    payoff_european_call,
    payoff_lookback_fixed_call,
)
from tests.checks import assert_within_se


def test_digital_put_plus_call_is_discounted_cash(market: Market, strike: float) -> None:
    call = price_mc(
        market, Contract(strike, ContractKind.digital_call, Q=1.0), trial_count=4_000, seed=0
    )
    put = price_mc(
        market, Contract(strike, ContractKind.digital_put, Q=1.0), trial_count=4_000, seed=1
    )
    cash = float(np.exp(-market.r * market.T))
    assert call.price + put.price == pytest.approx(cash, abs=1e-10)


def test_digital_put_matches_closed_form(market: Market, strike: float) -> None:
    contract = Contract(strike, ContractKind.digital_put)
    result = price_mc(market, contract, trial_count=8_000, seed=0)
    assert result.price == pytest.approx(black_scholes(market, contract), abs=1e-10)
    assert result.stderr < 1e-8


def test_asian_put_nonnegative(market: Market, strike: float) -> None:
    result = price_mc(
        market,
        Contract(strike, ContractKind.asian_put),
        steps=40,
        trial_count=4_000,
        seed=0,
    )
    assert result.price >= 0.0


def test_in_out_parity_on_paths() -> None:
    path = np.array(
        [
            [100.0, 100.0, 100.0],
            [120.0, 90.0, 105.0],
            [110.0, 80.0, 108.0],
        ]
    )
    k, b = 100.0, 115.0
    uo = payoff_barrier(path, k, b, up=True, knock_out=True, is_call=True)
    ui = payoff_barrier(path, k, b, up=True, knock_out=False, is_call=True)
    euro = payoff_european_call(path[-1], k)
    assert np.allclose(uo + ui, euro)


def test_up_and_in_at_spot_matches_bs_call(market: Market, strike: float) -> None:
    result = price_mc(
        market,
        Contract(strike, ContractKind.up_and_in_call, B=market.S),
        steps=40,
        trial_count=20_000,
        seed=0,
        control_variate=False,
    )
    bs = black_scholes(market, Contract(strike, ContractKind.euro_call))
    assert_within_se(result.price, bs, result.stderr)


def test_down_and_out_at_or_above_spot_is_zero(market: Market, strike: float) -> None:
    result = price_mc(
        market,
        Contract(strike, ContractKind.down_and_out_call, B=market.S),
        steps=20,
        trial_count=1_000,
        seed=0,
    )
    assert result.price == 0.0


def test_lookback_dominates_european_on_a_path() -> None:
    path = np.array([[100.0, 100.0], [90.0, 120.0], [95.0, 110.0]])
    look = payoff_lookback_fixed_call(path, 100.0)
    euro = payoff_european_call(path[-1], 100.0)
    assert np.all(look >= euro - 1e-12)


def test_lookback_call_above_bs(market: Market, strike: float) -> None:
    result = price_mc(
        market,
        Contract(strike, ContractKind.lookback_call),
        steps=40,
        trial_count=8_000,
        seed=0,
        control_variate=False,
    )
    bs = black_scholes(market, Contract(strike, ContractKind.euro_call))
    assert result.price > bs
