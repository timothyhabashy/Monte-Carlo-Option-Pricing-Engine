from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    Market,
    black_scholes,
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
