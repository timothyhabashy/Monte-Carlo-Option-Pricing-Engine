"""Longstaff–Schwartz American put under GBM."""

from __future__ import annotations

import numpy as np

from monte_carlo_option_engine.gbm import _positive_int, paths_from_shocks
from monte_carlo_option_engine.pricer import _Stats, _resolve_rng
from monte_carlo_option_engine.types import Market, PriceResult


def price_american_put(
    market: Market,
    strike: float,
    steps: int = 50,
    trial_count: int = 20_000,
    antithetic: bool = True,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> PriceResult:
    """LSM American put with polynomial basis ``{1, S, S^2}`` on ITM paths.

    The estimator is slightly biased (as usual for Longstaff–Schwartz) but
    should sit at or above the European put.
    """

    if strike <= 0:
        raise ValueError("strike must be positive")
    steps = _positive_int(steps, "steps")
    trial_count = _positive_int(trial_count, "trial_count")
    rng, used_seed = _resolve_rng(rng, seed)
    dt = market.T / steps
    df = float(np.exp(-market.r * dt))

    def _cashflows(n: int, shocks: np.ndarray) -> np.ndarray:
        paths = paths_from_shocks(market, shocks)
        # cf is the residual payoff valued at the current time index
        cf = np.maximum(strike - paths[-1], 0.0)
        for i in range(steps - 1, 0, -1):
            disc_cf = cf * df
            spot = paths[i]
            itm = spot < strike
            itm_idx = np.flatnonzero(itm)
            if itm_idx.size < 3:
                cf = disc_cf
                continue
            s_itm = spot[itm_idx]
            y = disc_cf[itm_idx]
            design = np.column_stack((np.ones(itm_idx.size), s_itm, s_itm**2))
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            continuation = beta[0] + beta[1] * s_itm + beta[2] * s_itm**2
            intrinsic = strike - s_itm
            exercise = intrinsic > continuation
            cf = disc_cf
            cf[itm_idx[exercise]] = intrinsic[exercise]
        return cf * df

    stats = _Stats()
    if not antithetic or trial_count == 1:
        z = rng.normal(size=(steps, trial_count))
        stats.add_individuals(_cashflows(trial_count, z))
    else:
        n_pairs = trial_count // 2
        remainder = trial_count % 2
        z = rng.normal(size=(steps, n_pairs))
        stats.add_pairs(_cashflows(n_pairs, z), _cashflows(n_pairs, -z))
        if remainder:
            z_extra = rng.normal(size=(steps, 1))
            stats.add_individuals(_cashflows(1, z_extra))

    result = stats.result(used_seed)
    intrinsic0 = max(strike - market.S, 0.0)
    if result.price < intrinsic0:
        result = PriceResult(
            price=intrinsic0,
            stderr=0.0,
            ci_low=intrinsic0,
            ci_high=intrinsic0,
            n_paths=trial_count,
            seed=used_seed,
        )
    return result
