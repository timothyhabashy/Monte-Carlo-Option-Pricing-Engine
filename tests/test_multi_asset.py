import numpy as np

from monte_carlo_option_engine import (
    BasketMarket,
    Contract,
    ContractKind,
    Market,
    black_scholes,
    price_basket_call,
    price_bestof_call,
)
from tests.checks import assert_within_se


def _twin(rho: float) -> BasketMarket:
    return BasketMarket(
        S=np.array([100.0, 100.0]),
        T=0.5,
        r=0.04,
        q=np.array([0.01, 0.01]),
        sigma=np.array([0.25, 0.25]),
        corr=np.array([[1.0, rho], [rho, 1.0]]),
    )


def test_perfectly_correlated_basket_is_european() -> None:
    basket = _twin(rho=1.0)
    result = price_basket_call(basket, strike=105.0, trial_count=20_000, seed=0)
    single = Market(S=100.0, T=0.5, r=0.04, q=0.01, sigma=0.25)
    bs = black_scholes(single, Contract(105.0, ContractKind.euro_call))
    assert_within_se(result.price, bs, result.stderr)


def test_perfectly_correlated_bestof_is_european() -> None:
    basket = _twin(rho=1.0)
    result = price_bestof_call(basket, strike=105.0, trial_count=20_000, seed=0)
    single = Market(S=100.0, T=0.5, r=0.04, q=0.01, sigma=0.25)
    bs = black_scholes(single, Contract(105.0, ContractKind.euro_call))
    assert_within_se(result.price, bs, result.stderr)


def test_bestof_at_least_basket() -> None:
    market = _twin(rho=0.2)
    basket = price_basket_call(market, strike=105.0, trial_count=8_000, seed=0)
    best = price_bestof_call(market, strike=105.0, trial_count=8_000, seed=1)
    assert best.price >= basket.price - 4.0 * np.sqrt(best.stderr**2 + basket.stderr**2)
