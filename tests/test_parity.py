import numpy as np

from monte_carlo_option_engine import Contract, ContractKind, Market, price_mc
from tests.checks import assert_within_se


def test_mc_put_call_parity(market: Market, strike: float) -> None:
    n = 50_000
    call = price_mc(
        market, Contract(strike, ContractKind.euro_call), trial_count=n, seed=0
    )
    put = price_mc(
        market, Contract(strike, ContractKind.euro_put), trial_count=n, seed=1
    )
    forward = market.S * np.exp(-market.q * market.T) - strike * np.exp(
        -market.r * market.T
    )
    se = float(np.sqrt(call.stderr**2 + put.stderr**2))
    assert_within_se(call.price - put.price, forward, se)
