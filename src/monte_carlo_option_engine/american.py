"""Longstaff–Schwartz American options."""

from __future__ import annotations

from typing import Literal

import numpy as np

from monte_carlo_option_engine.gbm import _positive_int, paths_from_shocks
from monte_carlo_option_engine.pricer import _Stats, _Z95, _resolve_rng
from monte_carlo_option_engine.types import Market, PriceResult

BasisName = Literal["poly", "laguerre"]


def _laguerre_basis(spots: np.ndarray, strike: float) -> np.ndarray:
    x = spots / strike
    w = np.exp(-0.5 * x)
    l0 = np.ones_like(x)
    l1 = 1.0 - x
    l2 = 1.0 - 2.0 * x + 0.5 * x * x
    l3 = 1.0 - 3.0 * x + 1.5 * x * x - (x**3) / 6.0
    return np.column_stack((w * l0, w * l1, w * l2, w * l3))


def _poly_basis(spots: np.ndarray, strike: float) -> np.ndarray:
    del strike
    return np.column_stack((np.ones(spots.size), spots, spots**2))


def lsm_cashflows(
    paths: np.ndarray,
    strike: float,
    df: float,
    *,
    is_call: bool = False,
    basis: BasisName = "poly",
    out_of_sample: bool = True,
) -> np.ndarray:
    """Discounted LSM cashflows to t=0. ``paths`` is ``(steps+1, n)``.

    When ``out_of_sample`` is true, even-indexed paths fit the continuation
    regression and odd-indexed paths are used for valuation.
    """

    paths = np.asarray(paths, dtype=float)
    steps = int(paths.shape[0] - 1)
    n = int(paths.shape[1])
    if is_call:
        cf = np.maximum(paths[-1] - strike, 0.0)
    else:
        cf = np.maximum(strike - paths[-1], 0.0)

    if basis == "laguerre":
        design_fn = _laguerre_basis
        min_itm = 4
    elif basis == "poly":
        design_fn = _poly_basis
        min_itm = 3
    else:
        raise ValueError("basis must be 'poly' or 'laguerre'")

    train = np.zeros(n, dtype=bool)
    val = np.ones(n, dtype=bool)
    if out_of_sample and n >= 8:
        idx = np.arange(n)
        train = idx % 2 == 0
        val = ~train

    for i in range(steps - 1, 0, -1):
        disc_cf = cf * df
        spot = paths[i]
        itm = (spot > strike) if is_call else (spot < strike)
        fit_idx = np.flatnonzero(itm & train) if out_of_sample and n >= 8 else np.flatnonzero(itm)
        if fit_idx.size < min_itm:
            cf = disc_cf
            continue
        y = disc_cf[fit_idx]
        design = design_fn(spot[fit_idx], strike)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        apply_idx = np.flatnonzero(itm)
        continuation = design_fn(spot[apply_idx], strike) @ beta
        intrinsic = (spot[apply_idx] - strike) if is_call else (strike - spot[apply_idx])
        exercise = intrinsic > continuation
        cf = disc_cf
        cf[apply_idx[exercise]] = intrinsic[exercise]

    cash = cf * df
    if out_of_sample and n >= 8:
        return cash[val]
    return cash


def _clamp_intrinsic(
    result: PriceResult, intrinsic0: float, trial_count: int, used_seed: int | None
) -> PriceResult:
    if result.price >= intrinsic0:
        return result
    return PriceResult(
        price=intrinsic0,
        stderr=result.stderr,
        ci_low=intrinsic0 - _Z95 * result.stderr,
        ci_high=intrinsic0 + _Z95 * result.stderr,
        n_paths=result.n_paths,
        seed=used_seed,
    )


def _simulate_american_paths(
    market: Market,
    process: object | None,
    steps: int,
    n: int,
    rng: np.random.Generator,
    shocks: object | None,
) -> np.ndarray:
    model = "gbm" if process is None else getattr(process, "model_name", "gbm")
    if model == "heston":
        from monte_carlo_option_engine.heston import simulate_heston_paths

        return simulate_heston_paths(
            market,
            process.params,  # type: ignore[union-attr]
            steps,
            n,
            rng,
            martingale_correction=getattr(process, "martingale_correction", True),
            shocks=shocks,  # type: ignore[arg-type]
        )
    if shocks is None:
        z = rng.normal(size=(steps, n))
    else:
        z = shocks  # type: ignore[assignment]
    if model == "localvol":
        return process.paths_from_shocks(z)  # type: ignore[union-attr]
    return paths_from_shocks(market, z)


def price_american(
    market: Market,
    strike: float,
    *,
    is_call: bool,
    steps: int = 50,
    trial_count: int = 20_000,
    antithetic: bool = True,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    process: object | None = None,
    basis: BasisName = "poly",
    out_of_sample: bool = True,
) -> PriceResult:
    """LSM American call or put. Default basis ``{1, S, S^2}``; optional Laguerre."""

    if strike <= 0:
        raise ValueError("strike must be positive")
    steps = _positive_int(steps, "steps")
    trial_count = _positive_int(trial_count, "trial_count")
    rng, used_seed = _resolve_rng(rng, seed)
    dt = market.T / steps
    df = float(np.exp(-market.r * dt))
    model = "gbm" if process is None else getattr(process, "model_name", "gbm")
    stats = _Stats()

    def cash_from_shocks(n: int, shocks: object | None) -> np.ndarray:
        paths = _simulate_american_paths(market, process, steps, n, rng, shocks)
        return lsm_cashflows(
            paths,
            strike,
            df,
            is_call=is_call,
            basis=basis,
            out_of_sample=out_of_sample,
        )

    if model == "heston":
        from monte_carlo_option_engine.heston import _antithetic_shocks, _draw_qe_shocks

        if not antithetic or trial_count == 1:
            stats.add_individuals(cash_from_shocks(trial_count, None))
        else:
            n_pairs = trial_count // 2
            remainder = trial_count % 2
            shocks = _draw_qe_shocks(rng, steps, n_pairs)
            anti = _antithetic_shocks(*shocks)
            stats.add_pairs(
                cash_from_shocks(n_pairs, shocks), cash_from_shocks(n_pairs, anti)
            )
            if remainder:
                stats.add_individuals(cash_from_shocks(1, None))
    else:
        if not antithetic or trial_count == 1:
            stats.add_individuals(cash_from_shocks(trial_count, None))
        else:
            n_pairs = trial_count // 2
            remainder = trial_count % 2
            z = rng.normal(size=(steps, n_pairs))
            stats.add_pairs(cash_from_shocks(n_pairs, z), cash_from_shocks(n_pairs, -z))
            if remainder:
                stats.add_individuals(cash_from_shocks(1, None))

    result = stats.result(used_seed)
    if is_call:
        intrinsic0 = max(market.S - strike, 0.0)
    else:
        intrinsic0 = max(strike - market.S, 0.0)
    return _clamp_intrinsic(result, intrinsic0, trial_count, used_seed)


def price_american_put(
    market: Market,
    strike: float,
    steps: int = 50,
    trial_count: int = 20_000,
    antithetic: bool = True,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    process: object | None = None,
    basis: BasisName = "poly",
    out_of_sample: bool = True,
) -> PriceResult:
    """LSM American put. Default polynomial basis on ITM paths; even/odd split-sample."""

    return price_american(
        market,
        strike,
        is_call=False,
        steps=steps,
        trial_count=trial_count,
        antithetic=antithetic,
        rng=rng,
        seed=seed,
        process=process,
        basis=basis,
        out_of_sample=out_of_sample,
    )


def price_american_call(
    market: Market,
    strike: float,
    steps: int = 50,
    trial_count: int = 20_000,
    antithetic: bool = True,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    process: object | None = None,
    basis: BasisName = "poly",
    out_of_sample: bool = True,
) -> PriceResult:
    """LSM American call. Equals the European when ``q = 0``; early exercise if ``q > 0``."""

    return price_american(
        market,
        strike,
        is_call=True,
        steps=steps,
        trial_count=trial_count,
        antithetic=antithetic,
        rng=rng,
        seed=seed,
        process=process,
        basis=basis,
        out_of_sample=out_of_sample,
    )
