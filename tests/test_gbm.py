import numpy as np
import pytest

from monte_carlo_option_engine import Market, simulate_paths, simulate_terminal
from tests.checks import assert_within_se


def test_paths_shape_and_spot(market: Market) -> None:
    rng = np.random.default_rng(0)
    paths = simulate_paths(market, steps=200, n_paths=512, rng=rng)
    assert paths.shape == (201, 512)
    assert np.allclose(paths[0], market.S)


def test_terminal_shape(market: Market) -> None:
    rng = np.random.default_rng(0)
    spots = simulate_terminal(market, n_paths=128, rng=rng)
    assert spots.shape == (128,)
    assert np.all(spots > 0)


def test_martingale_terminal(market: Market) -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    spots = simulate_terminal(market, n_paths=n, rng=rng)
    expected = market.S * np.exp((market.r - market.q) * market.T)
    stderr = spots.std(ddof=1) / np.sqrt(n)
    assert_within_se(float(spots.mean()), expected, float(stderr))


def test_float_steps_rejected(market: Market) -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="steps"):
        simulate_paths(market, steps=100.0, n_paths=10, rng=rng)  # type: ignore[arg-type]


def test_int_rng_rejected(market: Market) -> None:
    with pytest.raises(TypeError, match="Generator"):
        simulate_paths(market, steps=10, n_paths=10, rng=1)  # type: ignore[arg-type]
