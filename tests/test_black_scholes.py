import numpy as np
import pytest
from scipy.stats import norm

from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    Market,
    black_scholes,
    geometric_asian_call,
    simulate_paths,
)
from monte_carlo_option_engine.payoffs import payoff_asian_geometric_call
from tests.checks import assert_within_se

# Notebook synthetic market, 6-decimal BS printout.
_BS_CALL = 5.548166
_BS_PUT = 8.967779


def test_bs_call_matches_notebook(market: Market, strike: float) -> None:
    price = black_scholes(market, Contract(strike, ContractKind.euro_call))
    assert price == pytest.approx(_BS_CALL, abs=5e-7)


def test_bs_put_matches_notebook(market: Market, strike: float) -> None:
    price = black_scholes(market, Contract(strike, ContractKind.euro_put))
    assert price == pytest.approx(_BS_PUT, abs=5e-7)


def test_put_call_parity_on_bs(market: Market, strike: float) -> None:
    call = black_scholes(market, Contract(strike, ContractKind.euro_call))
    put = black_scholes(market, Contract(strike, ContractKind.euro_put))
    forward = market.S * np.exp(-market.q * market.T) - strike * np.exp(
        -market.r * market.T
    )
    assert call - put == pytest.approx(forward, rel=1e-12)


def test_digital_call_is_discounted_nd2(market: Market, strike: float) -> None:
    vol_sqrt = market.sigma * np.sqrt(market.T)
    mu = market.r - market.q
    d1 = (np.log(market.S / strike) + (mu + 0.5 * market.sigma**2) * market.T) / vol_sqrt
    d2 = d1 - vol_sqrt
    expected = np.exp(-market.r * market.T) * norm.cdf(d2)
    got = black_scholes(market, Contract(strike, ContractKind.digital_call, Q=1.0))
    assert got == pytest.approx(expected, rel=1e-12)


def test_digital_scales_with_payout(market: Market, strike: float) -> None:
    one = black_scholes(market, Contract(strike, ContractKind.digital_call, Q=1.0))
    ten = black_scholes(market, Contract(strike, ContractKind.digital_call, Q=10.0))
    assert ten == pytest.approx(10.0 * one, rel=1e-12)


def test_t_zero_intrinsics() -> None:
    mkt = Market(S=100.0, T=0.0, r=0.04, q=0.01, sigma=0.25)
    assert black_scholes(mkt, Contract(105.0, ContractKind.euro_call)) == 0.0
    assert black_scholes(mkt, Contract(90.0, ContractKind.euro_call)) == 10.0
    assert black_scholes(mkt, Contract(105.0, ContractKind.euro_put)) == 5.0
    assert black_scholes(mkt, Contract(90.0, ContractKind.euro_put)) == 0.0
    assert black_scholes(mkt, Contract(90.0, ContractKind.digital_call, Q=2.5)) == 2.5
    assert black_scholes(mkt, Contract(105.0, ContractKind.digital_call)) == 0.0
    assert black_scholes(mkt, Contract(90.0, ContractKind.digital_put, Q=2.5)) == 0.0
    assert black_scholes(mkt, Contract(105.0, ContractKind.digital_put)) == 1.0


def test_bs_rejects_asian(market: Market, strike: float) -> None:
    with pytest.raises(ValueError, match="only supports"):
        black_scholes(market, Contract(strike, ContractKind.asian_call))


def test_geometric_asian_one_step_is_european(market: Market, strike: float) -> None:
    geo = geometric_asian_call(market, Contract(strike, ContractKind.asian_call), steps=1)
    euro = black_scholes(market, Contract(strike, ContractKind.euro_call))
    assert geo == pytest.approx(euro, rel=1e-12)


def test_geometric_asian_matches_mc(market: Market, strike: float) -> None:
    steps = 50
    n = 30_000
    rng = np.random.default_rng(0)
    paths = simulate_paths(market, steps, n, rng)
    disc = np.exp(-market.r * market.T)
    pay = disc * payoff_asian_geometric_call(paths, strike)
    closed = geometric_asian_call(
        market, Contract(strike, ContractKind.asian_call), steps
    )
    stderr = float(pay.std(ddof=1) / np.sqrt(n))
    assert_within_se(float(pay.mean()), closed, stderr)
