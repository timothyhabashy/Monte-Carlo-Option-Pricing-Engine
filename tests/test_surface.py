import numpy as np
import pytest

from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    Market,
    black_scholes,
    implied_vol_from_call,
    surface_from_flat,
    surface_from_iv_grid,
)


def test_flat_surface_round_trip_iv() -> None:
    surface = surface_from_flat(100.0, 0.04, 0.01, 0.25)
    assert surface.iv(100.0, 0.5) == pytest.approx(0.25, abs=1e-10)
    assert surface.iv(90.0, 1.0) == pytest.approx(0.25, abs=1e-10)


def test_surface_call_matches_bs() -> None:
    surface = surface_from_flat(100.0, 0.04, 0.01, 0.25)
    price = surface.call(105.0, 0.5)
    bs = black_scholes(
        Market(S=100.0, T=0.5, r=0.04, q=0.01, sigma=0.25),
        Contract(105.0, ContractKind.euro_call),
    )
    assert price == pytest.approx(bs, rel=1e-12)


def test_surface_rejects_tenor_outside_range() -> None:
    surface = surface_from_flat(100.0, 0.04, 0.01, 0.25, expiries=np.array([0.25, 0.5, 1.0]))
    with pytest.raises(ValueError, match="outside"):
        surface.iv(100.0, 3.0)


def test_iv_grid_knots() -> None:
    times = np.array([0.5, 1.0])
    strikes = np.array([90.0, 100.0, 110.0])
    iv = np.array([[0.3, 0.25, 0.22], [0.28, 0.24, 0.21]])
    surface = surface_from_iv_grid(100.0, 0.04, 0.0, times, strikes, iv)
    assert surface.iv(100.0, 0.5) == pytest.approx(0.25, abs=1e-12)
    assert surface.iv(100.0, 1.0) == pytest.approx(0.24, abs=1e-12)


def test_implied_vol_from_call_recovers_sigma() -> None:
    market = Market(S=100.0, T=0.5, r=0.04, q=0.01, sigma=0.25)
    price = black_scholes(market, Contract(105.0, ContractKind.euro_call))
    iv = implied_vol_from_call(100.0, 0.5, 0.04, 0.01, 105.0, price)
    assert iv == pytest.approx(0.25, abs=1e-6)
