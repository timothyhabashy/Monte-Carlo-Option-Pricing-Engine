"""Risk-neutral geometric Brownian motion simulators."""

from __future__ import annotations

import numpy as np

from monte_carlo_option_engine.types import Market


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_generator(rng: np.random.Generator) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return rng


def terminal_from_shocks(market: Market, z: np.ndarray) -> np.ndarray:
    """Map standard normal shocks to terminal spots. ``z`` has shape ``(n,)``."""

    z = np.asarray(z, dtype=float)
    mu = market.r - market.q
    drift = (mu - 0.5 * market.sigma**2) * market.T
    diff = market.sigma * np.sqrt(market.T) * z
    return market.S * np.exp(drift + diff)


def paths_from_shocks(market: Market, z: np.ndarray) -> np.ndarray:
    """Map standard normal shocks to GBM paths.

    ``z`` has shape ``(steps, n_paths)``. Returns ``(steps + 1, n_paths)``
    including ``S0``.
    """

    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("z must have shape (steps, n_paths)")
    steps, n_paths = z.shape
    dt = market.T / steps
    mu = market.r - market.q
    increments = (mu - 0.5 * market.sigma**2) * dt + market.sigma * np.sqrt(dt) * z
    ln_s = np.log(market.S) + np.cumsum(increments, axis=0)
    spots = np.exp(ln_s)
    s0 = np.full((1, n_paths), market.S, dtype=float)
    return np.vstack([s0, spots])


def paths_from_brownian(market: Market, w: np.ndarray) -> np.ndarray:
    """Map Brownian motion values at ``dt,...,T`` to GBM paths.

    ``w`` has shape ``(steps, n_paths)``. Returns ``(steps + 1, n_paths)``.
    """

    w = np.asarray(w, dtype=float)
    if w.ndim != 2:
        raise ValueError("w must have shape (steps, n_paths)")
    steps, n_paths = w.shape
    times = (np.arange(1, steps + 1, dtype=float) / steps) * market.T
    mu = market.r - market.q
    log_s = (
        np.log(market.S)
        + (mu - 0.5 * market.sigma**2) * times[:, None]
        + market.sigma * w
    )
    spots = np.exp(log_s)
    s0 = np.full((1, n_paths), market.S, dtype=float)
    return np.vstack([s0, spots])


def simulate_terminal(
    market: Market, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """Simulate terminal spots only. Returns shape ``(n_paths,)``."""

    n_paths = _positive_int(n_paths, "n_paths")
    rng = _require_generator(rng)
    z = rng.normal(size=n_paths)
    return terminal_from_shocks(market, z)


def simulate_paths(
    market: Market, steps: int, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """Simulate full GBM paths. Returns shape ``(steps + 1, n_paths)``."""

    steps = _positive_int(steps, "steps")
    n_paths = _positive_int(n_paths, "n_paths")
    rng = _require_generator(rng)
    z = rng.normal(size=(steps, n_paths))
    return paths_from_shocks(market, z)
