import pytest

from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    HestonParams,
    Market,
    price_american_put,
    price_heston_call,
    price_mc,
    surface_from_flat,
)
from monte_carlo_option_engine.heston import heston_call_cf
from monte_carlo_option_engine.local_vol import local_vol_from_surface
from monte_carlo_option_engine.process import GBMProcess, HestonProcess, LocalVolProcess
from tests.checks import assert_within_se


def test_asian_under_heston_is_finite(market: Market, strike: float) -> None:
    params = HestonParams(kappa=1.5, theta=0.04, xi=0.4, rho=-0.5, v0=0.04)
    result = price_mc(
        market,
        Contract(strike, ContractKind.asian_call),
        steps=20,
        trial_count=2_000,
        seed=0,
        process=HestonProcess(market, params),
    )
    assert result.price >= 0.0
    assert result.price < market.S


def test_flat_local_vol_barrier_matches_gbm(market: Market, strike: float) -> None:
    surface = surface_from_flat(market.S, market.r, market.q, market.sigma)
    lv = local_vol_from_surface(surface)
    contract = Contract(strike, ContractKind.up_and_out_call, B=130.0)
    kwargs = dict(
        steps=40,
        trial_count=8_000,
        seed=0,
        control_variate=False,
        antithetic=True,
    )
    gbm = price_mc(market, contract, **kwargs)
    local = price_mc(
        market, contract, process=LocalVolProcess(market, lv), **kwargs
    )
    se = (gbm.stderr**2 + local.stderr**2) ** 0.5
    assert_within_se(local.price, gbm.price, se)


def test_price_heston_call_via_process_matches_cf() -> None:
    market = Market(S=100.0, T=0.5, r=0.03, q=0.0, sigma=0.2)
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, v0=0.04)
    cf = heston_call_cf(market, params, strike=100.0)
    via_api = price_heston_call(market, params, strike=100.0, steps=32, trial_count=4_000, seed=0)
    via_process = price_mc(
        market,
        Contract(100.0, ContractKind.euro_call),
        steps=32,
        trial_count=4_000,
        seed=0,
        process=HestonProcess(market, params),
    )
    assert via_api.price == pytest.approx(cf, abs=1e-10)
    assert via_process.price == pytest.approx(cf, abs=1e-10)


def test_american_put_kind_matches_wrapper(market: Market, strike: float) -> None:
    via_kind = price_mc(
        market,
        Contract(strike, ContractKind.american_put),
        steps=30,
        trial_count=4_000,
        seed=0,
    )
    via_fn = price_american_put(market, strike, steps=30, trial_count=4_000, seed=0)
    assert via_kind.price == pytest.approx(via_fn.price, rel=1e-12)
    assert via_kind.stderr == pytest.approx(via_fn.stderr, rel=1e-12)


def test_heston_rejects_sobol(market: Market, strike: float) -> None:
    params = HestonParams(kappa=1.5, theta=0.04, xi=0.3, rho=-0.4, v0=0.04)
    with pytest.raises(ValueError, match="sobol"):
        price_mc(
            market,
            Contract(strike, ContractKind.euro_call),
            method="sobol",
            antithetic=False,
            trial_count=128,
            process=HestonProcess(market, params),
        )


def test_gbm_process_matches_default(market: Market, strike: float) -> None:
    contract = Contract(strike, ContractKind.euro_call)
    default = price_mc(market, contract, trial_count=2_000, seed=1)
    wrapped = price_mc(
        market, contract, trial_count=2_000, seed=1, process=GBMProcess(market)
    )
    assert default.price == pytest.approx(wrapped.price, abs=1e-12)
