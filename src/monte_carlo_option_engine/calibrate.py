"""Heston calibration to an implied-vol surface via the CF pricer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from monte_carlo_option_engine.heston import HestonParams, heston_call_cf
from monte_carlo_option_engine.surface import ImpliedVolSurface
from monte_carlo_option_engine.types import Market


@dataclass(frozen=True)
class HestonCalibResult:
    params: HestonParams
    rmse: float
    n_quotes: int


def _pack(theta: np.ndarray) -> HestonParams:
    kappa, theta_v, xi, v0 = np.exp(theta[:4])
    rho = np.tanh(theta[4])
    kappa = float(np.clip(kappa, 1e-3, 20.0))
    theta_v = float(np.clip(theta_v, 1e-4, 1.0))
    xi = float(np.clip(xi, 1e-4, 5.0))
    v0 = float(np.clip(v0, 1e-4, 1.0))
    rho = float(np.clip(rho, -0.999, 0.999))
    return HestonParams(kappa=kappa, theta=theta_v, xi=xi, rho=rho, v0=v0)


def _unpack(params: HestonParams) -> np.ndarray:
    rho = float(np.clip(params.rho, -0.999, 0.999))
    atanh = 0.5 * np.log((1.0 + rho) / (1.0 - rho))
    return np.array(
        [
            np.log(max(params.kappa, 1e-6)),
            np.log(max(params.theta, 1e-6)),
            np.log(max(params.xi, 1e-6)),
            np.log(max(params.v0, 1e-6)),
            atanh,
        ],
        dtype=float,
    )


def _quote_grid(
    surface: ImpliedVolSurface, n_strikes: int, n_expiries: int
) -> tuple[np.ndarray, np.ndarray]:
    times = surface.expiries
    if times.size <= n_expiries:
        chosen_t = times
    else:
        idx = np.linspace(0, times.size - 1, n_expiries).round().astype(int)
        chosen_t = times[np.unique(idx)]
    k_lo, k_hi = float(surface.log_moneyness[1]), float(surface.log_moneyness[-2])
    log_k = np.linspace(k_lo, k_hi, n_strikes)
    strikes = surface.S * np.exp(log_k)
    tenors: list[float] = []
    ks: list[float] = []
    for tenor in chosen_t:
        for strike in strikes:
            tenors.append(float(tenor))
            ks.append(float(strike))
    return np.asarray(ks), np.asarray(tenors)


def calibrate_heston(
    surface: ImpliedVolSurface,
    *,
    n_strikes: int = 7,
    n_expiries: int = 4,
    feller_weight: float = 0.05,
    max_nfev: int = 80,
) -> HestonCalibResult:
    """Fit Heston CF call prices to ``surface.call`` on a small quote grid."""

    strikes, tenors = _quote_grid(surface, n_strikes, n_expiries)
    targets = np.array(
        [surface.call(k, t) for k, t in zip(strikes, tenors, strict=True)], dtype=float
    )
    atm_iv = surface.iv(surface.S, float(surface.expiries[0]))
    seed = HestonParams(
        kappa=1.5, theta=atm_iv**2, xi=1.0, rho=-0.5, v0=atm_iv**2
    )
    x0 = _unpack(seed)

    def residual(theta: np.ndarray) -> np.ndarray:
        params = _pack(theta)
        model = np.empty_like(targets)
        for i, (strike, tenor) in enumerate(zip(strikes, tenors, strict=True)):
            market = Market(S=surface.S, T=float(tenor), r=surface.r, q=surface.q, sigma=atm_iv)
            try:
                model[i] = heston_call_cf(market, params, float(strike))
            except (ValueError, FloatingPointError, TypeError):
                model[i] = 1e6
        price_err = model - targets
        feller = max(params.xi**2 - 2.0 * params.kappa * params.theta, 0.0)
        return np.append(price_err, feller_weight * feller)

    fit = least_squares(residual, x0, max_nfev=max_nfev, xtol=1e-6, ftol=1e-6)
    params = _pack(fit.x)
    model = np.array(
        [
            heston_call_cf(
                Market(S=surface.S, T=float(t), r=surface.r, q=surface.q, sigma=atm_iv),
                params,
                float(k),
            )
            for k, t in zip(strikes, tenors, strict=True)
        ]
    )
    rmse = float(np.sqrt(np.mean((model - targets) ** 2)))
    return HestonCalibResult(params=params, rmse=rmse, n_quotes=int(targets.size))
