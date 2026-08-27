"""One-factor processes: GBM, Heston, and Dupire local vol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from monte_carlo_option_engine.gbm import (
    paths_from_shocks,
    simulate_paths,
    simulate_terminal,
    terminal_from_shocks,
)
from monte_carlo_option_engine.heston import (
    HestonParams,
    simulate_heston_paths,
    simulate_heston_terminal,
)
from monte_carlo_option_engine.local_vol import LocalVol, paths_from_local_vol
from monte_carlo_option_engine.types import Market


class Process(Protocol):
    """One-factor spot dynamics. Baskets stay outside this protocol."""

    def simulate_terminal(self, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        """Terminal spots, shape ``(n_paths,)``."""
        ...

    def simulate_paths(
        self, steps: int, n_paths: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Spot paths including ``S0``, shape ``(steps + 1, n_paths)``."""
        ...


@dataclass(frozen=True)
class GBMProcess:
    """Constant-σ GBM. ``market.sigma`` is used."""

    market: Market
    model_name: str = "gbm"

    def simulate_terminal(self, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        return simulate_terminal(self.market, n_paths, rng)

    def simulate_paths(self, steps: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        return simulate_paths(self.market, steps, n_paths, rng)

    def terminal_from_shocks(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return terminal_from_shocks(self.market, z.reshape(-1))

    def paths_from_shocks(self, z: np.ndarray) -> np.ndarray:
        return paths_from_shocks(self.market, z)


@dataclass(frozen=True)
class HestonProcess:
    """Heston QE. ``market.sigma`` is ignored; variance is ``params.v0``."""

    market: Market
    params: HestonParams
    martingale_correction: bool = True
    model_name: str = "heston"

    def simulate_terminal(self, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        steps = 64
        return simulate_heston_terminal(
            self.market,
            self.params,
            steps,
            n_paths,
            rng,
            martingale_correction=self.martingale_correction,
        )

    def simulate_paths(self, steps: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        return simulate_heston_paths(
            self.market,
            self.params,
            steps,
            n_paths,
            rng,
            martingale_correction=self.martingale_correction,
        )


@dataclass(frozen=True)
class LocalVolProcess:
    """Dupire local vol. ``market.sigma`` is ignored."""

    market: Market
    local_vol: LocalVol
    model_name: str = "localvol"

    def paths_from_shocks(self, z: np.ndarray) -> np.ndarray:
        return paths_from_local_vol(self.market, self.local_vol, z)

    def terminal_from_shocks(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            raise ValueError("local-vol terminals need step-wise shocks of shape (steps, n)")
        return paths_from_local_vol(self.market, self.local_vol, z)[-1]

    def simulate_paths(self, steps: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.normal(size=(steps, n_paths))
        return self.paths_from_shocks(z)

    def simulate_terminal(self, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        return self.simulate_paths(64, n_paths, rng)[-1]
