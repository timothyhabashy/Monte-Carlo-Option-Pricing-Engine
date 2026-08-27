import numpy as np
import pytest

from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    HestonParams,
    Market,
    black_scholes,
    black_scholes_greeks,
    heston_call_cf,
    heston_greeks_cf,
    heston_put_cf,
    price_heston_call,
)
from monte_carlo_option_engine.heston import simulate_heston_terminal
from tests.checks import assert_within_se


def test_heston_zero_volvol_matches_bs() -> None:
    market = Market(S=100.0, T=0.5, r=0.04, q=0.01, sigma=0.25)
    params = HestonParams(kappa=2.0, theta=0.0625, xi=0.0, rho=0.0, v0=0.0625)
    result = price_heston_call(
        market,
        params,
        strike=105.0,
        steps=40,
        trial_count=20_000,
        seed=0,
        antithetic=False,
        control_variate=False,
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
        control_variate=False,
    )
    assert cf > 0.0
    assert_within_se(result.price, cf, result.stderr)


def test_heston_cf_small_xi_near_bs() -> None:
    market = Market(S=100.0, T=0.5, r=0.04, q=0.0, sigma=0.2)
    params = HestonParams(kappa=3.0, theta=0.04, xi=1e-7, rho=0.0, v0=0.04)
    cf = heston_call_cf(market, params, strike=100.0)
    bs = black_scholes(market, Contract(100.0, ContractKind.euro_call))
    assert cf == bs


def test_heston_cv_recovers_cf() -> None:
    market = Market(S=100.0, T=0.5, r=0.03, q=0.0, sigma=0.2)
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, v0=0.04)
    cf = heston_call_cf(market, params, strike=100.0)
    result = price_heston_call(
        market, params, strike=100.0, steps=32, trial_count=4_000, seed=0
    )
    assert result.price == pytest.approx(cf, abs=1e-10)
    assert result.stderr < 1e-12


def test_heston_antithetic_reduces_stderr() -> None:
    market = Market(S=100.0, T=0.5, r=0.03, q=0.0, sigma=0.2)
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, v0=0.04)
    kwargs = dict(
        strike=100.0,
        steps=32,
        trial_count=8_000,
        seed=0,
        control_variate=False,
    )
    raw = price_heston_call(market, params, antithetic=False, **kwargs)
    anti = price_heston_call(market, params, antithetic=True, **kwargs)
    assert anti.stderr < raw.stderr


def test_heston_martingale_after_k0() -> None:
    market = Market(S=100.0, T=1.0, r=0.03, q=0.0, sigma=0.2)
    params = HestonParams(kappa=1.0, theta=0.04, xi=1.0, rho=-0.8, v0=0.04)
    rng = np.random.default_rng(0)
    n = 8_000
    spots = simulate_heston_terminal(
        market, params, steps=48, n_paths=n, rng=rng, martingale_correction=True
    )
    forward = market.S * np.exp((market.r - market.q) * market.T)
    stderr = float(spots.std(ddof=1) / np.sqrt(n))
    assert_within_se(float(spots.mean()), forward, stderr)


def test_heston_put_call_parity() -> None:
    market = Market(S=100.0, T=0.5, r=0.03, q=0.01, sigma=0.2)
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, v0=0.04)
    strike = 100.0
    call = heston_call_cf(market, params, strike)
    put = heston_put_cf(market, params, strike)
    forward = market.S * np.exp(-market.q * market.T) - strike * np.exp(
        -market.r * market.T
    )
    assert call - put == pytest.approx(forward, abs=1e-10)


def test_heston_cf_greeks_zero_volvol_match_bs() -> None:
    market = Market(S=100.0, T=0.5, r=0.04, q=0.01, sigma=0.2)
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.0, rho=0.0, v0=0.04)
    greeks_h = heston_greeks_cf(market, params, 100.0)
    greeks_bs = black_scholes_greeks(market, Contract(100.0, ContractKind.euro_call))
    assert greeks_h.delta == pytest.approx(greeks_bs.delta, abs=5e-3)
    assert greeks_h.vega > 0.0
