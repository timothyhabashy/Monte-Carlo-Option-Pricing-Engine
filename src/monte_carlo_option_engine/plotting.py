"""Path plots for demos and notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from monte_carlo_option_engine.gbm import simulate_paths
from monte_carlo_option_engine.types import Market


def plot_paths(
    market: Market,
    steps: int,
    n_paths: int,
    rng: np.random.Generator,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot simulated GBM paths against calendar time in years."""

    paths = simulate_paths(market, steps, n_paths, rng)
    time = np.linspace(0.0, market.T, steps + 1)
    fig: Figure | None = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(time, paths)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Spot")
    ax.set_title("Example GBM paths")
    return fig, ax
