import numpy as np
import pytest

from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    black_scholes,
    price_mc,
)
from tests.checks import assert_within_se


def test_antithetic_reduces_stderr(market, strike: float) -> None:
    naive = price_mc(
        market,
        Contract(strike, ContractKind.euro_call),
        trial_count=20_000,
        seed=0,
        control_variate=False,
        antithetic=False,
    )
    anti = price_mc(
        market,
        Contract(strike, ContractKind.euro_call),
        trial_count=20_000,
        seed=0,
        control_variate=False,
        antithetic=True,
    )
    assert anti.stderr < naive.stderr


def test_control_variate_euro_recovers_bs(market, strike: float) -> None:
    contract = Contract(strike, ContractKind.euro_call)
    raw = price_mc(
        market, contract, trial_count=20_000, seed=0, control_variate=False
    )
    cv = price_mc(
        market, contract, trial_count=20_000, seed=0, control_variate=True
    )
    bs = black_scholes(market, contract)
    assert cv.stderr < raw.stderr
    assert cv.price == pytest.approx(bs, abs=1e-12)
    assert cv.stderr < 1e-12
    assert_within_se(raw.price, bs, raw.stderr)


def test_estimated_beta_still_near_bs(market, strike: float) -> None:
    contract = Contract(strike, ContractKind.euro_call)
    cv = price_mc(
        market,
        contract,
        trial_count=20_000,
        seed=0,
        control_variate=True,
        estimate_beta=True,
    )
    bs = black_scholes(market, contract)
    assert_within_se(cv.price, bs, cv.stderr if cv.stderr > 0 else 1e-12)


def test_geometric_cv_tightens_asian(market, strike: float) -> None:
    contract = Contract(strike, ContractKind.asian_call)
    kwargs = dict(steps=50, trial_count=15_000, seed=0, antithetic=False)
    raw = price_mc(market, contract, control_variate=False, **kwargs)
    cv = price_mc(market, contract, control_variate=True, **kwargs)
    assert cv.stderr < raw.stderr
    se = float(np.sqrt(cv.stderr**2 + raw.stderr**2))
    assert_within_se(cv.price, raw.price, se)
