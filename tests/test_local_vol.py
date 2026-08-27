import numpy as np

from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    HestonParams,
    Market,
    black_scholes,
    heston_call_cf,
    implied_vol_from_call,
    local_vol_from_surface,
    simulate_local_vol_terminal,
    surface_from_flat,
    surface_from_iv_grid,
)
from tests.checks import assert_within_se


def test_flat_smile_local_vol_near_sigma() -> None:
    sigma = 0.25
    surface = surface_from_flat(100.0, 0.04, 0.01, sigma)
    local = local_vol_from_surface(surface, n_times=11, n_strikes=21)
    mid_t = float(local.times[local.times.size // 2])
    sig_atm = float(local.sigma_at(np.array([100.0]), mid_t)[0])
    assert abs(sig_atm - sigma) < 0.03


def test_flat_local_vol_european_matches_bs() -> None:
    market = Market(S=100.0, T=0.5, r=0.04, q=0.01, sigma=0.25)
    surface = surface_from_flat(market.S, market.r, market.q, market.sigma)
    local = local_vol_from_surface(surface, n_times=11, n_strikes=21)
    rng = np.random.default_rng(0)
    n = 12_000
    spots = simulate_local_vol_terminal(market, local, steps=40, n_paths=n, rng=rng)
    disc = np.exp(-market.r * market.T)
    pay = disc * np.maximum(spots - 105.0, 0.0)
    bs = black_scholes(market, Contract(105.0, ContractKind.euro_call))
    stderr = float(pay.std(ddof=1) / np.sqrt(n))
    assert_within_se(float(pay.mean()), bs, stderr)
    forward = market.S * np.exp((market.r - market.q) * market.T)
    se_s = float(spots.std(ddof=1) / np.sqrt(n))
    assert_within_se(float(spots.mean()), forward, se_s)


def test_heston_surface_local_vol_matches_cf_call() -> None:
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
    spot, rate, div = 100.0, 0.03, 0.0
    times = np.array([0.25, 0.5, 1.0])
    strikes = np.array([85.0, 95.0, 100.0, 105.0, 115.0])
    iv = np.empty((times.size, strikes.size))
    for i, tenor in enumerate(times):
        mkt = Market(S=spot, T=float(tenor), r=rate, q=div, sigma=0.2)
        for j, strike in enumerate(strikes):
            price = heston_call_cf(mkt, params, float(strike))
            iv[i, j] = implied_vol_from_call(
                spot, float(tenor), rate, div, float(strike), price
            )
    surface = surface_from_iv_grid(spot, rate, div, times, strikes, iv)
    local = local_vol_from_surface(surface, n_times=9, n_strikes=21)
    tenor = 0.5
    market = Market(S=spot, T=tenor, r=rate, q=div, sigma=0.2)
    rng = np.random.default_rng(1)
    n = 10_000
    spots = simulate_local_vol_terminal(market, local, steps=48, n_paths=n, rng=rng)
    disc = np.exp(-rate * tenor)
    pay = disc * np.maximum(spots - 100.0, 0.0)
    cf = heston_call_cf(market, params, 100.0)
    stderr = float(pay.std(ddof=1) / np.sqrt(n))
    assert_within_se(float(pay.mean()), cf, stderr)
