import pytest

from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    Market,
    black_scholes,
    crr_american_put,
    price_american_call,
    price_american_put,
    price_mc,
)


def test_american_put_at_least_european(market: Market, strike: float) -> None:
    american = price_american_put(
        market, strike, steps=40, trial_count=8_000, seed=0
    )
    european = price_mc(
        market, Contract(strike, ContractKind.euro_put), trial_count=8_000, seed=0
    )
    assert american.price + 1e-8 >= european.price


def test_deep_itm_short_american_near_intrinsic() -> None:
    market = Market(S=40.0, T=1.0 / 12.0, r=0.04, q=0.0, sigma=0.15)
    strike = 100.0
    intrinsic = strike - market.S
    result = price_american_put(
        market, strike, steps=20, trial_count=6_000, seed=0
    )
    assert result.price >= intrinsic - 1e-8
    assert result.price == intrinsic or abs(result.price - intrinsic) / intrinsic < 0.03
    assert black_scholes(market, Contract(strike, ContractKind.euro_put)) <= result.price + 1e-8
    assert result.stderr >= 0.0


def test_lsm_near_crr(market: Market, strike: float) -> None:
    tree = crr_american_put(market, strike, steps=200)
    lsm = price_american_put(market, strike, steps=50, trial_count=12_000, seed=0)
    assert abs(lsm.price - tree) / max(tree, 1e-8) < 0.08
    assert lsm.price + 1e-8 >= black_scholes(
        market, Contract(strike, ContractKind.euro_put)
    )


def test_american_call_zero_div_near_european() -> None:
    market = Market(S=100.0, T=0.5, r=0.04, q=0.0, sigma=0.25)
    strike = 100.0
    american = price_american_call(market, strike, steps=40, trial_count=10_000, seed=0)
    european = black_scholes(market, Contract(strike, ContractKind.euro_call))
    assert abs(american.price - european) / max(european, 1e-8) < 0.08


def test_american_call_with_dividend_beats_european() -> None:
    market = Market(S=100.0, T=1.0, r=0.05, q=0.08, sigma=0.25)
    strike = 100.0
    american = price_american_call(market, strike, steps=50, trial_count=12_000, seed=0)
    european = black_scholes(market, Contract(strike, ContractKind.euro_call))
    assert american.price + 0.05 >= european


def test_american_call_kind_matches_wrapper(market: Market, strike: float) -> None:
    via_kind = price_mc(
        market,
        Contract(strike, ContractKind.american_call),
        steps=25,
        trial_count=3_000,
        seed=1,
    )
    via_fn = price_american_call(market, strike, steps=25, trial_count=3_000, seed=1)
    assert via_kind.price == pytest.approx(via_fn.price, rel=1e-12)


def test_laguerre_basis_american_put(market: Market, strike: float) -> None:
    result = price_american_put(
        market, strike, steps=30, trial_count=6_000, seed=0, basis="laguerre"
    )
    assert result.price >= 0.0
    assert result.stderr >= 0.0
