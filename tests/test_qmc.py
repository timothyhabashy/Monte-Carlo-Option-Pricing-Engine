import numpy as np
import pytest

from monte_carlo_option_engine import Contract, ContractKind, black_scholes, price_mc
from monte_carlo_option_engine.qmc import brownian_bridge
from tests.checks import assert_within_se


def test_brownian_bridge_terminal_and_variance() -> None:
    rng = np.random.default_rng(0)
    steps = 8
    n = 25_000
    time_horizon = 1.0
    z = rng.normal(size=(steps, n))
    w = brownian_bridge(z, time_horizon)
    assert w.shape == (steps, n)
    assert np.allclose(w[-1], np.sqrt(time_horizon) * z[0])
    times = (np.arange(1, steps + 1) / steps) * time_horizon
    sample_var = w.var(axis=1, ddof=1)
    # Gaussian sample-variance SE is about t * sqrt(2/n)
    for t, v in zip(times, sample_var, strict=True):
        se = t * np.sqrt(2.0 / n)
        assert abs(v - t) < 4.0 * se


def test_sobol_euro_call_within_four_se(market, strike: float) -> None:
    contract = Contract(strike, ContractKind.euro_call)
    result = price_mc(
        market,
        contract,
        trial_count=4096,
        seed=0,
        method="sobol",
        control_variate=False,
        antithetic=False,
    )
    bs = black_scholes(market, contract)
    assert_within_se(result.price, bs, result.stderr)


def test_sobol_rejects_antithetic(market, strike: float) -> None:
    with pytest.raises(ValueError, match="sobol"):
        price_mc(
            market,
            Contract(strike, ContractKind.euro_call),
            trial_count=16,
            method="sobol",
            antithetic=True,
            control_variate=False,
        )


def test_invalid_draw_method(market, strike: float) -> None:
    with pytest.raises(ValueError, match="method"):
        price_mc(
            market,
            Contract(strike, ContractKind.euro_call),
            trial_count=10,
            method="halton",  # type: ignore[arg-type]
        )

