"""Dupire local volatility from an implied-vol surface, plus Euler paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monte_carlo_option_engine.gbm import _positive_int
from monte_carlo_option_engine.surface import ImpliedVolSurface
from monte_carlo_option_engine.types import Market

_VAR_FLOOR = 1e-8


@dataclass(frozen=True)
class LocalVol:
    """Local vol σ(S, t) on a rectilinear ``(t, log K)`` grid."""

    times: np.ndarray
    log_strikes: np.ndarray
    sigma: np.ndarray
    r: float
    q: float

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float).reshape(-1)
        log_k = np.asarray(self.log_strikes, dtype=float).reshape(-1)
        sig = np.asarray(self.sigma, dtype=float)
        if times.size < 2 or log_k.size < 2:
            raise ValueError("local-vol grid needs at least 2 times and 2 strikes")
        if sig.shape != (times.size, log_k.size):
            raise ValueError("sigma must have shape (n_times, n_strikes)")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "log_strikes", log_k)
        object.__setattr__(self, "sigma", np.maximum(sig, np.sqrt(_VAR_FLOOR)))

    def sigma_at(self, spots: np.ndarray, time: float) -> np.ndarray:
        """Vectorized local vol at calendar time ``time`` and spots ``spots``."""

        spots = np.asarray(spots, dtype=float)
        log_s = np.log(np.maximum(spots, 1e-16))
        times = self.times
        t = min(max(float(time), float(times[0])), float(times[-1]))
        j = int(np.searchsorted(times, t, side="right") - 1)
        j = min(max(j, 0), times.size - 2)
        w = 0.0 if times[j + 1] <= times[j] else (t - times[j]) / (times[j + 1] - times[j])
        lo = np.interp(log_s, self.log_strikes, self.sigma[j])
        hi = np.interp(log_s, self.log_strikes, self.sigma[j + 1])
        return (1.0 - w) * lo + w * hi


def local_vol_from_surface(
    surface: ImpliedVolSurface, n_times: int = 21, n_strikes: int = 41
) -> LocalVol:
    """Gatheral / IV-form Dupire on a dense ``(T, y)`` grid, ``y = log(K/F)``."""

    t_grid = np.linspace(surface.t_min, surface.t_max, n_times)
    k_lo = float(surface.log_moneyness[0])
    k_hi = float(surface.log_moneyness[-1])
    # Work in log-strike; y = log(K/F) = log(K/S) - (r-q)T
    log_k_grid = np.linspace(k_lo, k_hi, n_strikes)
    w = np.empty((n_times, n_strikes), dtype=float)
    for i, tenor in enumerate(t_grid):
        for j, log_k in enumerate(log_k_grid):
            strike = surface.S * np.exp(log_k)
            w[i, j] = surface.total_var(float(strike), float(tenor))

    dt = np.gradient(t_grid)
    dy = np.gradient(log_k_grid)
    # Convert to y = log(K/F) = log_k - (r-q)T for Gatheral derivatives.
    # On a log-K grid, ∂/∂y = ∂/∂logK because F is independent of K.
    dw_dt = np.gradient(w, t_grid, axis=0)
    dw_dy = np.gradient(w, log_k_grid, axis=1)
    d2w_dy2 = np.gradient(dw_dy, log_k_grid, axis=1)

    y = log_k_grid[None, :] - (surface.r - surface.q) * t_grid[:, None]
    w_safe = np.maximum(w, _VAR_FLOOR)
    denom = (
        (1.0 - y * dw_dy / w_safe) ** 2
        + 0.5 * d2w_dy2 * w_safe
        - 0.25 * (dw_dy**2) * (0.25 + 1.0 / w_safe)
    )
    denom = np.maximum(denom, _VAR_FLOOR)
    loc_var = np.maximum(dw_dt / denom, _VAR_FLOOR)
    sigma = np.sqrt(loc_var)
    return LocalVol(
        times=t_grid,
        log_strikes=np.log(surface.S) + log_k_grid,
        sigma=sigma,
        r=surface.r,
        q=surface.q,
    )


def paths_from_local_vol(
    market: Market, local_vol: LocalVol, shocks: np.ndarray
) -> np.ndarray:
    """Euler-on-log-S local-vol paths. ``shocks`` is ``(steps, n_paths)``."""

    z = np.asarray(shocks, dtype=float)
    if z.ndim != 2:
        raise ValueError("shocks must have shape (steps, n_paths)")
    steps, n_paths = z.shape
    dt = market.T / steps
    log_s = np.full(n_paths, np.log(market.S), dtype=float)
    out = np.empty((steps + 1, n_paths), dtype=float)
    out[0] = market.S
    mu = market.r - market.q
    for i in range(steps):
        t = i * dt
        sig = local_vol.sigma_at(np.exp(log_s), t)
        log_s = log_s + (mu - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * z[i]
        out[i + 1] = np.exp(log_s)
    return out


def simulate_local_vol_paths(
    market: Market,
    local_vol: LocalVol,
    steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    steps = _positive_int(steps, "steps")
    n_paths = _positive_int(n_paths, "n_paths")
    z = rng.normal(size=(steps, n_paths))
    return paths_from_local_vol(market, local_vol, z)


def simulate_local_vol_terminal(
    market: Market,
    local_vol: LocalVol,
    steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return simulate_local_vol_paths(market, local_vol, steps, n_paths, rng)[-1]
