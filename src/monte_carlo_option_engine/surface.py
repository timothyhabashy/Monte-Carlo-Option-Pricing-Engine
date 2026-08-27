"""Implied-vol surface: total-variance interpolation in log-moneyness and T."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monte_carlo_option_engine.black_scholes import black_scholes
from monte_carlo_option_engine.types import Contract, ContractKind, Market

_IV_MIN = 1e-4
_IV_MAX = 3.0


def _as_1d(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


@dataclass(frozen=True)
class ImpliedVolSurface:
    """Rectilinear total-variance grid in log-moneyness ``k = log(K/S)``.

    Interpolation is linear in ``k`` (flat outside the knot range) then linear
    in total variance ``w = iv² T`` across expiries. Tenors outside
    ``[T_min, T_max]`` raise.
    """

    S: float
    r: float
    q: float
    expiries: np.ndarray
    log_moneyness: np.ndarray
    total_variance: np.ndarray

    def __post_init__(self) -> None:
        expiries = _as_1d(self.expiries, "expiries")
        knots = _as_1d(self.log_moneyness, "log_moneyness")
        grid = np.asarray(self.total_variance, dtype=float)
        if expiries.size < 1 or knots.size < 2:
            raise ValueError("need at least one expiry and two log-moneyness knots")
        if np.any(np.diff(expiries) <= 0) or np.any(np.diff(knots) <= 0):
            raise ValueError("expiries and log_moneyness must be strictly increasing")
        if grid.shape != (expiries.size, knots.size):
            raise ValueError("total_variance must have shape (n_expiries, n_knots)")
        if self.S <= 0:
            raise ValueError("S must be positive")
        object.__setattr__(self, "expiries", expiries)
        object.__setattr__(self, "log_moneyness", knots)
        object.__setattr__(self, "total_variance", np.maximum(grid, 1e-12))

    @property
    def t_min(self) -> float:
        return float(self.expiries[0])

    @property
    def t_max(self) -> float:
        return float(self.expiries[-1])

    def _w_at_expiry(self, expiry_index: int, log_k: float) -> float:
        knots = self.log_moneyness
        row = self.total_variance[expiry_index]
        return float(np.interp(log_k, knots, row))

    def total_var(self, strike: float, tenor: float) -> float:
        if strike <= 0:
            raise ValueError("strike must be positive")
        if tenor < self.t_min - 1e-12 or tenor > self.t_max + 1e-12:
            raise ValueError(
                f"T={tenor} is outside the surface range [{self.t_min}, {self.t_max}]"
            )
        tenor = min(max(tenor, self.t_min), self.t_max)
        log_k = float(np.log(strike / self.S))
        times = self.expiries
        if times.size == 1:
            return self._w_at_expiry(0, log_k)
        idx = int(np.searchsorted(times, tenor, side="right") - 1)
        idx = min(max(idx, 0), times.size - 2)
        t0, t1 = float(times[idx]), float(times[idx + 1])
        w0 = self._w_at_expiry(idx, log_k)
        w1 = self._w_at_expiry(idx + 1, log_k)
        weight = 0.0 if t1 <= t0 else (tenor - t0) / (t1 - t0)
        return float((1.0 - weight) * w0 + weight * w1)

    def iv(self, strike: float, tenor: float) -> float:
        if tenor <= 0:
            raise ValueError("T must be positive to read implied vol")
        w = self.total_var(strike, tenor)
        return float(np.clip(np.sqrt(w / tenor), _IV_MIN, _IV_MAX))

    def call(self, strike: float, tenor: float) -> float:
        sigma = self.iv(strike, tenor)
        market = Market(S=self.S, T=tenor, r=self.r, q=self.q, sigma=sigma)
        return black_scholes(market, Contract(strike, ContractKind.euro_call))

    def put(self, strike: float, tenor: float) -> float:
        sigma = self.iv(strike, tenor)
        market = Market(S=self.S, T=tenor, r=self.r, q=self.q, sigma=sigma)
        return black_scholes(market, Contract(strike, ContractKind.euro_put))


def implied_vol_from_call(
    spot: float,
    tenor: float,
    rate: float,
    div: float,
    strike: float,
    price: float,
) -> float:
    """Black–Scholes implied vol for a European call price."""

    from scipy.optimize import brentq

    if tenor <= 0:
        raise ValueError("T must be positive")
    disc_k = strike * np.exp(-rate * tenor)
    disc_s = spot * np.exp(-div * tenor)
    intrinsic = max(disc_s - disc_k, 0.0)
    upper = disc_s
    if price < intrinsic - 1e-10 or price > upper + 1e-8:
        raise ValueError("call price is outside arbitrage bounds")

    def objective(sigma: float) -> float:
        market = Market(S=spot, T=tenor, r=rate, q=div, sigma=float(sigma))
        return black_scholes(market, Contract(strike, ContractKind.euro_call)) - price

    lo, hi = _IV_MIN, _IV_MAX
    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0:
        return lo if abs(f_lo) < abs(f_hi) else hi
    return float(brentq(objective, lo, hi, xtol=1e-10))


def surface_from_flat(
    spot: float,
    rate: float,
    div: float,
    sigma: float,
    expiries: np.ndarray | None = None,
    log_moneyness: np.ndarray | None = None,
) -> ImpliedVolSurface:
    """Constant-IV surface (Dupire local vol should recover ``sigma``)."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    times = (
        np.asarray(expiries, dtype=float).reshape(-1)
        if expiries is not None
        else np.array([0.25, 0.5, 1.0, 2.0])
    )
    knots = (
        np.asarray(log_moneyness, dtype=float).reshape(-1)
        if log_moneyness is not None
        else np.linspace(-0.8, 0.8, 17)
    )
    w = np.outer(times, np.full(knots.size, sigma**2))
    return ImpliedVolSurface(
        S=spot, r=rate, q=div, expiries=times, log_moneyness=knots, total_variance=w
    )


def surface_from_iv_grid(
    spot: float,
    rate: float,
    div: float,
    expiries: np.ndarray,
    strikes: np.ndarray,
    implied_vols: np.ndarray,
) -> ImpliedVolSurface:
    """Build a surface from IV on a rectangular ``(expiry, strike)`` grid."""

    times = _as_1d(expiries, "expiries")
    k_abs = _as_1d(strikes, "strikes")
    iv = np.asarray(implied_vols, dtype=float)
    if iv.shape != (times.size, k_abs.size):
        raise ValueError("implied_vols must have shape (n_expiries, n_strikes)")
    log_k = np.log(k_abs / spot)
    order = np.argsort(log_k)
    log_k = log_k[order]
    iv = np.clip(iv[:, order], _IV_MIN, _IV_MAX)
    total = iv**2 * times[:, None]
    return ImpliedVolSurface(
        S=spot,
        r=rate,
        q=div,
        expiries=times,
        log_moneyness=log_k,
        total_variance=total,
    )
