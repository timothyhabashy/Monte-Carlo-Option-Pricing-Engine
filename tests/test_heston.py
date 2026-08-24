from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    HestonParams,
    Market,
    black_scholes,
    heston_call_cf,
    price_heston_call,
)
from tests.checks import assert_within_se


def test_heston_zero_volvol_matches_bs() -> None:
    market = Market(S=100.0, T=0.5, r=0.04, q=0.01, sigma=0.25)
    params = HestonParams(kappa=2.0, theta=0.0625, xi=0.0, rho=0.0, v0=0.0625)
    result = price_heston_call(
        market, params, strike=105.0, steps=40, trial_count=20_000, seed=0, antithetic=False
    )
    bs = black_scholes(market, Contract(105.0, ContractKind.euro_call))
    assert_within_se(result.price, bs, result.stderr)


def test_heston_mc_vs_cf() -> None:
    market = Market(S=100.0, T=0.5, r=0.03, q=0.0, sigma=0.2)
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, v0=0.04)
    cf = heston_call_cf(market, params, strike=100.0)
    result = price_heston_call(
        market,
        params,
        strike=100.0,
        steps=64,
        trial_count=12_000,
        seed=0,
        antithetic=False,
    )
    assert cf > 0.0
    assert_within_se(result.price, cf, result.stderr)


def test_heston_cf_small_xi_near_bs() -> None:
    market = Market(S=100.0, T=0.5, r=0.04, q=0.0, sigma=0.2)
    params = HestonParams(kappa=3.0, theta=0.04, xi=1e-7, rho=0.0, v0=0.04)
    cf = heston_call_cf(market, params, strike=100.0)
    bs = black_scholes(market, Contract(100.0, ContractKind.euro_call))
    assert cf == bs
