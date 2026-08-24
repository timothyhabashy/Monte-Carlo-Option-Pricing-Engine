import numpy as np

from monte_carlo_option_engine import Market, plot_paths


def test_plot_paths_uses_time_axis(market: Market) -> None:
    fig, ax = plot_paths(market, steps=20, n_paths=4, rng=np.random.default_rng(0))
    assert ax.get_xlabel() == "Time (years)"
    line = ax.get_lines()[0]
    x = line.get_xdata()
    assert x[0] == 0.0
    assert x[-1] == market.T
    assert len(ax.get_lines()) == 4
    fig.clf()
