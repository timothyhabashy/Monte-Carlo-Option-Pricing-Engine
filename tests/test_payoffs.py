import numpy as np

from monte_carlo_option_engine.payoffs import (
    payoff_asian_arithmetic_call,
    payoff_asian_geometric_call,
    payoff_digital_call,
    payoff_digital_put,
    payoff_european_call,
    payoff_european_put,
    payoff_up_and_out_call,
)


def test_european_call_put() -> None:
    spots = np.array([90.0, 100.0, 110.0])
    assert np.allclose(payoff_european_call(spots, 100.0), [0.0, 0.0, 10.0])
    assert np.allclose(payoff_european_put(spots, 100.0), [10.0, 0.0, 0.0])


def test_digital_call_strict() -> None:
    spots = np.array([99.0, 100.0, 101.0])
    assert np.allclose(payoff_digital_call(spots, 100.0, 2.0), [0.0, 0.0, 2.0])
    assert np.allclose(payoff_digital_put(spots, 100.0, 2.0), [2.0, 0.0, 0.0])


def test_asian_excludes_s0() -> None:
    path = np.array(
        [
            [100.0, 100.0],
            [110.0, 90.0],
            [130.0, 80.0],
        ]
    )
    # averages of rows 1: are 120 and 85
    assert np.allclose(payoff_asian_arithmetic_call(path, 100.0), [20.0, 0.0])
    geo = np.exp(np.array([np.log(110.0 * 130.0) / 2.0, np.log(90.0 * 80.0) / 2.0]))
    assert np.allclose(payoff_asian_geometric_call(path, 100.0), np.maximum(geo - 100.0, 0.0))


def test_up_and_out_discrete() -> None:
    path = np.array(
        [
            [100.0, 100.0],
            [120.0, 105.0],
            [110.0, 108.0],
        ]
    )
    # first path max=120 knocks out at B=115; second never hits 115, call vs 100 = 8
    assert np.allclose(payoff_up_and_out_call(path, 100.0, 115.0), [0.0, 8.0])
