import numpy as np

from monte_carlo_option_engine import (
    HestonParams,
    calibrate_heston,
    heston_call_cf,
    implied_vol_from_call,
    surface_from_iv_grid,
)
from monte_carlo_option_engine.types import Market


def _heston_iv_surface(params: HestonParams) -> object:
    spot, rate, div = 100.0, 0.03, 0.0
    times = np.array([0.5, 1.0])
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    iv = np.empty((times.size, strikes.size), dtype=float)
    for i, tenor in enumerate(times):
        market = Market(S=spot, T=float(tenor), r=rate, q=div, sigma=0.2)
        for j, strike in enumerate(strikes):
            price = heston_call_cf(market, params, float(strike))
            iv[i, j] = implied_vol_from_call(
                spot, float(tenor), rate, div, float(strike), price
            )
    return surface_from_iv_grid(spot, rate, div, times, strikes, iv)


def test_calibrate_heston_recovers_skew_and_v0() -> None:
    truth = HestonParams(kappa=1.5, theta=0.04, xi=0.5, rho=-0.6, v0=0.04)
    surface = _heston_iv_surface(truth)
    result = calibrate_heston(
        surface, n_strikes=5, n_expiries=2, max_nfev=40, feller_weight=0.02
    )
    assert result.params.rho < 0.0
    assert abs(result.params.v0 - truth.v0) < 0.03
    assert result.rmse < 0.15
    assert result.n_quotes == 10
