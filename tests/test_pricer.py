import numpy as np
import pytest

from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    Market,
    PriceResult,
    black_scholes,
    price_mc,
    print_price,
)
from monte_carlo_option_engine.engine import price_mc as facade_price_mc
from tests.checks import assert_within_se

_N = 50_000


def test_euro_call_within_four_se(market: Market, strike: float) -> None:
    contract = Contract(strike, ContractKind.euro_call)
    result = price_mc(
        market, contract, trial_count=_N, batch_size=10_000, seed=0
    )
    bs = black_scholes(market, contract)
    assert isinstance(result, PriceResult)
    assert result.n_paths == _N
    assert result.seed == 0
    assert result.stderr > 0
    assert_within_se(result.price, bs, result.stderr)


def test_euro_put_within_four_se(market: Market, strike: float) -> None:
    contract = Contract(strike, ContractKind.euro_put)
    result = price_mc(market, contract, trial_count=_N, seed=0)
    bs = black_scholes(market, contract)
    assert_within_se(result.price, bs, result.stderr)


def test_digital_call_within_four_se(market: Market, strike: float) -> None:
    contract = Contract(strike, ContractKind.digital_call, Q=1.0)
    result = price_mc(market, contract, trial_count=_N, seed=0)
    bs = black_scholes(market, contract)
    assert_within_se(result.price, bs, result.stderr)


def test_asian_price_nonnegative(market: Market, strike: float) -> None:
    result = price_mc(
        market,
        Contract(strike, ContractKind.asian_call),
        steps=50,
        trial_count=10_000,
        seed=0,
    )
    assert result.price >= 0.0
    assert result.n_paths == 10_000


def test_string_kind_accepted(market: Market, strike: float) -> None:
    result = price_mc(
        market, Contract(strike, "euro_call"), trial_count=2_000, seed=1
    )
    assert result.price > 0


def test_seed_and_rng_are_exclusive(market: Market, strike: float) -> None:
    with pytest.raises(ValueError, match="only one"):
        price_mc(
            market,
            Contract(strike, ContractKind.euro_call),
            trial_count=10,
            rng=np.random.default_rng(0),
            seed=0,
        )


def test_rng_leaves_seed_unset(market: Market, strike: float) -> None:
    result = price_mc(
        market,
        Contract(strike, ContractKind.euro_call),
        trial_count=100,
        rng=np.random.default_rng(0),
    )
    assert result.seed is None


def test_facade_matches_package(market: Market, strike: float) -> None:
    contract = Contract(strike, ContractKind.euro_call)
    a = price_mc(market, contract, trial_count=4_000, seed=2)
    b = facade_price_mc(market, contract, trial_count=4_000, seed=2)
    assert a.price == b.price
    assert a.stderr == b.stderr


def test_print_price_returns_result(
    market: Market, strike: float, capsys: pytest.CaptureFixture[str]
) -> None:
    result = print_price(
        market, Contract(strike, ContractKind.euro_call), trial_count=1_000, seed=0
    )
    captured = capsys.readouterr()
    assert "euro_call" in captured.out
    assert "price=" in captured.out
    assert result.n_paths == 1_000


def test_odd_trial_count_with_antithetic(market: Market, strike: float) -> None:
    result = price_mc(
        market,
        Contract(strike, ContractKind.euro_call),
        trial_count=1_001,
        seed=0,
    )
    assert result.n_paths == 1_001
    assert result.price > 0


def test_trial_count_must_be_positive(market: Market, strike: float) -> None:
    with pytest.raises(ValueError, match="trial_count"):
        price_mc(market, Contract(strike, ContractKind.euro_call), trial_count=0)
