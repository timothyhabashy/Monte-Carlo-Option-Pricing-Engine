"""Correlated multi-asset GBM: basket and best-of calls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monte_carlo_option_engine.pricer import _Stats, _resolve_rng
from monte_carlo_option_engine.types import PriceResult


@dataclass(frozen=True)
class BasketMarket:
    """d-dimensional GBM: spots, vols, dividend yields, and correlation."""

    S: np.ndarray
    T: float
    r: float
    q: np.ndarray
    sigma: np.ndarray
    corr: np.ndarray

    def __post_init__(self) -> None:
        s = np.asarray(self.S, dtype=float).reshape(-1)
        q = np.asarray(self.q, dtype=float).reshape(-1)
        sig = np.asarray(self.sigma, dtype=float).reshape(-1)
        corr = np.asarray(self.corr, dtype=float)
        n = s.size
        if n < 2:
            raise ValueError("BasketMarket needs at least two underlyings")
        if q.size == 1:
            q = np.full(n, float(q[0]))
        if sig.size == 1:
            sig = np.full(n, float(sig[0]))
        if q.size != n or sig.size != n:
            raise ValueError("S, q, and sigma must share the same length")
        if corr.shape != (n, n):
            raise ValueError("corr must be (d, d)")
        if np.any(s <= 0) or np.any(sig <= 0):
            raise ValueError("spots and vols must be positive")
        if self.T < 0:
            raise ValueError("T must be non-negative")
        object.__setattr__(self, "S", s)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "sigma", sig)
        object.__setattr__(self, "corr", corr)


def _correlate(z: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """``z`` is ``(d, n)`` i.i.d. normals. Returns correlated shocks."""

    try:
        factor = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        eigval, eigvec = np.linalg.eigh(corr)
        eigval = np.clip(eigval, 0.0, None)
        factor = eigvec @ np.diag(np.sqrt(eigval))
    return factor @ z


def simulate_basket_terminal(
    market: BasketMarket, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """Terminal spots, shape ``(d, n_paths)``."""

    d = market.S.size
    z = rng.normal(size=(d, n_paths))
    z_corr = _correlate(z, market.corr)
    drift = (market.r - market.q - 0.5 * market.sigma**2) * market.T
    diff = (market.sigma * np.sqrt(market.T))[:, None] * z_corr
    return market.S[:, None] * np.exp(drift[:, None] + diff)


def _price_multi(
    market: BasketMarket,
    strike: float,
    payoff,
    trial_count: int,
    antithetic: bool,
    rng: np.random.Generator | None,
    seed: int | None,
) -> PriceResult:
    if strike <= 0:
        raise ValueError("strike must be positive")
    if trial_count <= 0:
        raise ValueError("trial_count must be positive")
    rng, used_seed = _resolve_rng(rng, seed)
    discount = float(np.exp(-market.r * market.T))
    stats = _Stats()

    def discounted_pay(n: int, z: np.ndarray) -> np.ndarray:
        z_corr = _correlate(z, market.corr)
        drift = (market.r - market.q - 0.5 * market.sigma**2) * market.T
        diff = (market.sigma * np.sqrt(market.T))[:, None] * z_corr
        spots = market.S[:, None] * np.exp(drift[:, None] + diff)
        return discount * payoff(spots, strike)

    d = market.S.size
    if not antithetic or trial_count == 1:
        z = rng.normal(size=(d, trial_count))
        stats.add_individuals(discounted_pay(trial_count, z))
    else:
        n_pairs = trial_count // 2
        remainder = trial_count % 2
        z = rng.normal(size=(d, n_pairs))
        stats.add_pairs(discounted_pay(n_pairs, z), discounted_pay(n_pairs, -z))
        if remainder:
            z_extra = rng.normal(size=(d, 1))
            stats.add_individuals(discounted_pay(1, z_extra))
    return stats.result(used_seed)


def _basket_payoff(spots: np.ndarray, strike: float) -> np.ndarray:
    return np.maximum(spots.mean(axis=0) - strike, 0.0)


def _bestof_payoff(spots: np.ndarray, strike: float) -> np.ndarray:
    return np.maximum(spots.max(axis=0) - strike, 0.0)


def price_basket_call(
    market: BasketMarket,
    strike: float,
    trial_count: int = 50_000,
    antithetic: bool = True,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> PriceResult:
    """Call on the arithmetic average of the terminal spots."""

    return _price_multi(
        market, strike, _basket_payoff, trial_count, antithetic, rng, seed
    )


def price_bestof_call(
    market: BasketMarket,
    strike: float,
    trial_count: int = 50_000,
    antithetic: bool = True,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> PriceResult:
    """Call on the maximum of the terminal spots."""

    return _price_multi(
        market, strike, _bestof_payoff, trial_count, antithetic, rng, seed
    )
