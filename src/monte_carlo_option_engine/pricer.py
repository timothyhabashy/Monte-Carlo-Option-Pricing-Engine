"""Batched Monte Carlo pricer with antithetic, CV, and optional Sobol QMC."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.special import zeta
from scipy.stats import norm
from scipy.stats.qmc import Sobol

from monte_carlo_option_engine.black_scholes import (
    black_scholes,
    geometric_asian_call,
    geometric_asian_put,
)
from monte_carlo_option_engine.gbm import (
    _positive_int,
    paths_from_brownian,
    paths_from_shocks,
    terminal_from_shocks,
)
from monte_carlo_option_engine.payoffs import (
    payoff_asian_arithmetic_call,
    payoff_asian_arithmetic_put,
    payoff_asian_geometric_call,
    payoff_asian_geometric_put,
    payoff_barrier,
    payoff_digital_call,
    payoff_digital_put,
    payoff_european_call,
    payoff_european_put,
    payoff_lookback_fixed_call,
)
from monte_carlo_option_engine.qmc import brownian_bridge
from monte_carlo_option_engine.types import (
    CLOSED_FORM_KINDS,
    EARLY_EXERCISE_KINDS,
    PATH_KINDS,
    TERMINAL_KINDS,
    UP_BARRIER_KINDS,
    Contract,
    ContractKind,
    Market,
    Monitoring,
    PriceResult,
)

_Z95 = 1.96
# Broadie–Glasserman–Kou continuity constant, -ζ(1/2) / √(2π) ≈ 0.5826
_BGK_BETA = float(-zeta(0.5) / np.sqrt(2.0 * np.pi))

DrawMethod = Literal["iid", "sobol"]


class _Stats:
    """Price from all path payoffs; SE from IID observations (pair averages)."""

    def __init__(self) -> None:
        self.n_paths = 0
        self.n_obs = 0
        self.sum_x_all = 0.0
        self.sum_y_all = 0.0
        self.sum_obs_x = 0.0
        self.sum_obs_x2 = 0.0
        self.sum_obs_y = 0.0
        self.sum_obs_y2 = 0.0
        self.sum_obs_xy = 0.0
        self.has_y = False

    def add_individuals(self, x: np.ndarray, y: np.ndarray | None = None) -> None:
        x = np.asarray(x, dtype=float).ravel()
        if x.size == 0:
            return
        self.n_paths += int(x.size)
        self.n_obs += int(x.size)
        self.sum_x_all += float(x.sum())
        self.sum_obs_x += float(x.sum())
        self.sum_obs_x2 += float(np.square(x).sum())
        if y is not None:
            y = np.asarray(y, dtype=float).ravel()
            if y.size != x.size:
                raise ValueError("x and y must have the same length")
            self.has_y = True
            self.sum_y_all += float(y.sum())
            self.sum_obs_y += float(y.sum())
            self.sum_obs_y2 += float(np.square(y).sum())
            self.sum_obs_xy += float((x * y).sum())

    def add_pairs(
        self,
        x: np.ndarray,
        x_anti: np.ndarray,
        y: np.ndarray | None = None,
        y_anti: np.ndarray | None = None,
    ) -> None:
        x = np.asarray(x, dtype=float).ravel()
        x_anti = np.asarray(x_anti, dtype=float).ravel()
        if x.size != x_anti.size:
            raise ValueError("antithetic batches must have the same length")
        if x.size == 0:
            return
        avg_x = 0.5 * (x + x_anti)
        self.n_paths += int(x.size + x_anti.size)
        self.n_obs += int(avg_x.size)
        self.sum_x_all += float(x.sum() + x_anti.sum())
        self.sum_obs_x += float(avg_x.sum())
        self.sum_obs_x2 += float(np.square(avg_x).sum())
        if y is not None:
            y = np.asarray(y, dtype=float).ravel()
            y_anti = np.asarray(y_anti, dtype=float).ravel()
            avg_y = 0.5 * (y + y_anti)
            self.has_y = True
            self.sum_y_all += float(y.sum() + y_anti.sum())
            self.sum_obs_y += float(avg_y.sum())
            self.sum_obs_y2 += float(np.square(avg_y).sum())
            self.sum_obs_xy += float((avg_x * avg_y).sum())

    def result(
        self,
        seed: int | None,
        *,
        use_cv: bool = False,
        expected_y: float = 0.0,
        estimate_beta: bool = False,
    ) -> PriceResult:
        if self.n_paths <= 0:
            raise ValueError("no paths were simulated")
        mean_x = self.sum_x_all / self.n_paths
        if use_cv:
            if not self.has_y:
                raise RuntimeError("control variate requested but Y was not accumulated")
            mean_y = self.sum_y_all / self.n_paths
            beta = 1.0
            var_adj = 0.0
            if self.n_obs > 1:
                mx = self.sum_obs_x / self.n_obs
                my = self.sum_obs_y / self.n_obs
                var_x = (self.sum_obs_x2 - self.n_obs * mx**2) / (self.n_obs - 1)
                var_y = (self.sum_obs_y2 - self.n_obs * my**2) / (self.n_obs - 1)
                cov = (self.sum_obs_xy - self.n_obs * mx * my) / (self.n_obs - 1)
                if estimate_beta and var_y > 1e-18:
                    beta = float(cov / var_y)
                var_adj = float(var_x + beta**2 * var_y - 2.0 * beta * cov)
                if var_adj < 0.0:
                    var_adj = 0.0
            price = mean_x - beta * (mean_y - expected_y)
            stderr = float(np.sqrt(var_adj / self.n_obs)) if self.n_obs > 1 else 0.0
        else:
            price = mean_x
            if self.n_obs > 1:
                mean_obs = self.sum_obs_x / self.n_obs
                var = (self.sum_obs_x2 - self.n_obs * mean_obs**2) / (self.n_obs - 1)
                if var < 0.0:
                    var = 0.0
                stderr = float(np.sqrt(var / self.n_obs))
            else:
                stderr = 0.0
        return PriceResult(
            price=float(price),
            stderr=stderr,
            ci_low=float(price - _Z95 * stderr),
            ci_high=float(price + _Z95 * stderr),
            n_paths=self.n_paths,
            seed=seed,
        )


class _NormalSource:
    def __init__(
        self, dim: int, method: DrawMethod, rng: np.random.Generator, seed: int | None
    ) -> None:
        self.dim = dim
        self.method = method
        self.rng = rng
        self._sobol: Sobol | None = None
        if method == "sobol":
            sobol_seed = seed if seed is not None else int(rng.integers(0, 2**31 - 1))
            self._sobol = Sobol(d=dim, scramble=True, seed=sobol_seed)

    def draw(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty((self.dim, 0), dtype=float)
        if self.method == "iid":
            return self.rng.normal(size=(self.dim, n))
        assert self._sobol is not None
        uniforms = np.clip(self._sobol.random(n), 1e-12, 1.0 - 1e-12)
        return np.asarray(norm.ppf(uniforms).T, dtype=float)


def _resolve_rng(
    rng: np.random.Generator | None, seed: int | None
) -> tuple[np.random.Generator, int | None]:
    if rng is not None and seed is not None:
        raise ValueError("pass only one of rng or seed")
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        return rng, None
    if seed is not None:
        return np.random.default_rng(seed), int(seed)
    return np.random.default_rng(), None


def _payoff_terminal(kind: ContractKind, spots: np.ndarray, contract: Contract) -> np.ndarray:
    if kind is ContractKind.euro_call:
        return payoff_european_call(spots, contract.K)
    if kind is ContractKind.euro_put:
        return payoff_european_put(spots, contract.K)
    if kind is ContractKind.digital_call:
        return payoff_digital_call(spots, contract.K, contract.Q)
    if kind is ContractKind.digital_put:
        return payoff_digital_put(spots, contract.K, contract.Q)
    raise ValueError(f"Unsupported terminal contract kind: {kind}")


def _adjusted_barrier(market: Market, contract: Contract, steps: int) -> float:
    if contract.B is None:
        raise ValueError(f"Barrier B required for {contract.kind}")
    if contract.monitoring is Monitoring.discrete:
        return float(contract.B)
    dt = market.T / steps
    shift = _BGK_BETA * market.sigma * np.sqrt(dt)
    kind = ContractKind(contract.kind)
    if kind in UP_BARRIER_KINDS:
        return float(contract.B * np.exp(-shift))
    return float(contract.B * np.exp(shift))


def _payoff_path(
    kind: ContractKind, paths: np.ndarray, contract: Contract, market: Market, steps: int
) -> np.ndarray:
    if kind is ContractKind.asian_call:
        return payoff_asian_arithmetic_call(paths, contract.K)
    if kind is ContractKind.asian_put:
        return payoff_asian_arithmetic_put(paths, contract.K)
    if kind is ContractKind.lookback_call:
        return payoff_lookback_fixed_call(paths, contract.K)
    if kind in (
        ContractKind.up_and_out_call,
        ContractKind.up_and_in_call,
        ContractKind.down_and_out_call,
        ContractKind.down_and_in_call,
        ContractKind.up_and_out_put,
        ContractKind.up_and_in_put,
        ContractKind.down_and_out_put,
        ContractKind.down_and_in_put,
    ):
        name = str(kind)
        barrier = _adjusted_barrier(market, contract, steps)
        return payoff_barrier(
            paths,
            contract.K,
            barrier,
            up="up_and" in name,
            knock_out="_out_" in name,
            is_call=name.endswith("_call"),
        )
    raise ValueError(f"Unsupported path-dependent contract kind: {kind}")


def _cv_expected_y(
    market: Market, contract: Contract, kind: ContractKind, steps: int, use_cv: bool
) -> float | None:
    if not use_cv:
        return None
    if kind in CLOSED_FORM_KINDS:
        return float(black_scholes(market, contract))
    if kind is ContractKind.asian_call:
        return float(geometric_asian_call(market, contract, steps))
    if kind is ContractKind.asian_put:
        return float(geometric_asian_put(market, contract, steps))
    return None


def _cv_terminal_y(kind: ContractKind, spots: np.ndarray, contract: Contract) -> np.ndarray:
    return _payoff_terminal(kind, spots, contract)


def _cv_path_y(kind: ContractKind, paths: np.ndarray, contract: Contract) -> np.ndarray:
    if kind is ContractKind.asian_call:
        return payoff_asian_geometric_call(paths, contract.K)
    if kind is ContractKind.asian_put:
        return payoff_asian_geometric_put(paths, contract.K)
    raise RuntimeError(f"no control variate payoff for {kind}")


def _paths_from_draws(
    market: Market, z: np.ndarray, method: DrawMethod, process: object | None = None
) -> np.ndarray:
    if process is not None and getattr(process, "model_name", "") == "localvol":
        if method == "sobol":
            w = brownian_bridge(z, market.T)
            steps = z.shape[0]
            dt = market.T / steps if steps else 1.0
            d_w = np.empty_like(w)
            d_w[0] = w[0]
            d_w[1:] = np.diff(w, axis=0)
            z_inc = d_w / np.sqrt(max(dt, 1e-16))
            return process.paths_from_shocks(z_inc)  # type: ignore[union-attr]
        return process.paths_from_shocks(z)  # type: ignore[union-attr]
    if method == "sobol":
        w = brownian_bridge(z, market.T)
        return paths_from_brownian(market, w)
    return paths_from_shocks(market, z)


def _terminal_batch(
    market: Market,
    contract: Contract,
    kind: ContractKind,
    n: int,
    antithetic: bool,
    source: _NormalSource,
    discount: float,
    stats: _Stats,
    use_cv: bool,
) -> None:
    def y_of(spots: np.ndarray) -> np.ndarray | None:
        if not use_cv:
            return None
        return discount * _cv_terminal_y(kind, spots, contract)

    if not antithetic or n == 1:
        z = source.draw(n)[0]
        spots = terminal_from_shocks(market, z)
        stats.add_individuals(discount * _payoff_terminal(kind, spots, contract), y_of(spots))
        return

    n_pairs = n // 2
    remainder = n % 2
    z = source.draw(n_pairs)[0]
    spots = terminal_from_shocks(market, z)
    spots_anti = terminal_from_shocks(market, -z)
    stats.add_pairs(
        discount * _payoff_terminal(kind, spots, contract),
        discount * _payoff_terminal(kind, spots_anti, contract),
        y_of(spots),
        y_of(spots_anti),
    )
    if remainder:
        z_extra = source.draw(1)[0]
        extra = terminal_from_shocks(market, z_extra)
        stats.add_individuals(
            discount * _payoff_terminal(kind, extra, contract), y_of(extra)
        )


def _path_batch(
    market: Market,
    contract: Contract,
    kind: ContractKind,
    n: int,
    steps: int,
    antithetic: bool,
    source: _NormalSource,
    discount: float,
    stats: _Stats,
    use_cv: bool,
    method: DrawMethod,
    process: object | None = None,
) -> None:
    def y_of(paths: np.ndarray) -> np.ndarray | None:
        if not use_cv:
            return None
        return discount * _cv_path_y(kind, paths, contract)

    def x_of(paths: np.ndarray) -> np.ndarray:
        if kind in TERMINAL_KINDS:
            return discount * _payoff_terminal(kind, paths[-1], contract)
        return discount * _payoff_path(kind, paths, contract, market, steps)

    if not antithetic or n == 1:
        z = source.draw(n)
        paths = _paths_from_draws(market, z, method, process)
        stats.add_individuals(x_of(paths), y_of(paths) if use_cv else None)
        return

    n_pairs = n // 2
    remainder = n % 2
    z = source.draw(n_pairs)
    paths = _paths_from_draws(market, z, method, process)
    paths_anti = _paths_from_draws(market, -z, method, process)
    stats.add_pairs(
        x_of(paths),
        x_of(paths_anti),
        y_of(paths) if use_cv else None,
        y_of(paths_anti) if use_cv else None,
    )
    if remainder:
        z_extra = source.draw(1)
        extra = _paths_from_draws(market, z_extra, method, process)
        stats.add_individuals(x_of(extra), y_of(extra) if use_cv else None)


def _price_heston_contract(
    market: Market,
    contract: Contract,
    process: object,
    steps: int,
    trial_count: int,
    antithetic: bool,
    control_variate: bool,
    estimate_beta: bool,
    method: DrawMethod,
    rng: np.random.Generator,
    seed: int | None,
) -> PriceResult:
    if method == "sobol":
        raise ValueError("HestonProcess does not support method='sobol'")
    from monte_carlo_option_engine.heston import (
        _antithetic_shocks,
        _draw_qe_shocks,
        heston_put_cf,
        price_heston_call,
        simulate_heston_paths,
        simulate_heston_terminal,
    )
    from monte_carlo_option_engine.payoffs import payoff_digital_call, payoff_digital_put, payoff_european_put

    kind = ContractKind(contract.kind)
    params = process.params  # type: ignore[union-attr]
    martingale = getattr(process, "martingale_correction", True)
    if kind is ContractKind.euro_call:
        return price_heston_call(
            market,
            params,
            contract.K,
            steps=steps,
            trial_count=trial_count,
            antithetic=antithetic,
            control_variate=control_variate,
            estimate_beta=estimate_beta,
            martingale_correction=martingale,
            rng=None if seed is not None else rng,
            seed=seed,
        )

    discount = float(np.exp(-market.r * market.T))
    stats = _Stats()
    expected_y: float | None = None
    if kind is ContractKind.euro_put and control_variate:
        expected_y = heston_put_cf(market, params, contract.K)
    use_cv = expected_y is not None

    def terminal_pay(spots: np.ndarray) -> np.ndarray:
        if kind is ContractKind.euro_put:
            return discount * payoff_european_put(spots, contract.K)
        if kind is ContractKind.digital_call:
            return discount * payoff_digital_call(spots, contract.K, contract.Q)
        if kind is ContractKind.digital_put:
            return discount * payoff_digital_put(spots, contract.K, contract.Q)
        raise ValueError(f"unsupported Heston terminal kind: {kind}")

    if kind in TERMINAL_KINDS:
        if not antithetic or trial_count == 1:
            spots = simulate_heston_terminal(
                market, params, steps, trial_count, rng, martingale_correction=martingale
            )
            x = terminal_pay(spots)
            stats.add_individuals(x, x if use_cv else None)
        else:
            n_pairs = trial_count // 2
            remainder = trial_count % 2
            shocks = _draw_qe_shocks(rng, steps, n_pairs)
            spots = simulate_heston_terminal(
                market,
                params,
                steps,
                n_pairs,
                rng,
                martingale_correction=martingale,
                shocks=shocks,
            )
            anti = _antithetic_shocks(*shocks)
            spots_a = simulate_heston_terminal(
                market,
                params,
                steps,
                n_pairs,
                rng,
                martingale_correction=martingale,
                shocks=anti,
            )
            x, xa = terminal_pay(spots), terminal_pay(spots_a)
            stats.add_pairs(x, xa, x if use_cv else None, xa if use_cv else None)
            if remainder:
                extra = simulate_heston_terminal(
                    market, params, steps, 1, rng, martingale_correction=martingale
                )
                xe = terminal_pay(extra)
                stats.add_individuals(xe, xe if use_cv else None)
        return stats.result(
            seed, use_cv=use_cv, expected_y=expected_y or 0.0, estimate_beta=estimate_beta
        )

    steps = _positive_int(steps, "steps")

    def path_pay(paths: np.ndarray) -> np.ndarray:
        return discount * _payoff_path(kind, paths, contract, market, steps)

    if not antithetic or trial_count == 1:
        paths = simulate_heston_paths(
            market, params, steps, trial_count, rng, martingale_correction=martingale
        )
        stats.add_individuals(path_pay(paths))
    else:
        n_pairs = trial_count // 2
        remainder = trial_count % 2
        shocks = _draw_qe_shocks(rng, steps, n_pairs)
        paths = simulate_heston_paths(
            market,
            params,
            steps,
            n_pairs,
            rng,
            martingale_correction=martingale,
            shocks=shocks,
        )
        anti = _antithetic_shocks(*shocks)
        paths_a = simulate_heston_paths(
            market,
            params,
            steps,
            n_pairs,
            rng,
            martingale_correction=martingale,
            shocks=anti,
        )
        stats.add_pairs(path_pay(paths), path_pay(paths_a))
        if remainder:
            extra = simulate_heston_paths(
                market, params, steps, 1, rng, martingale_correction=martingale
            )
            stats.add_individuals(path_pay(extra))
    return stats.result(seed)


def price_mc(
    market: Market,
    contract: Contract,
    steps: int = 200,
    trial_count: int = 100_000,
    batch_size: int = 50_000,
    antithetic: bool = True,
    control_variate: bool = True,
    estimate_beta: bool = False,
    method: DrawMethod = "iid",
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    process: object | None = None,
) -> PriceResult:
    """Monte Carlo pricer with batching, antithetic draws, and control variates.

    Antithetic standard errors use pair-average discounted payoffs. Europeans
    and digitals use Black–Scholes as a control; arithmetic Asians use the
    geometric-Asian closed form. Default vanilla CV makes the reported price
    identical to the closed form (stderr ≈ 0). ``method='sobol'`` uses
    scrambled Sobol (Brownian bridge on path-dependent contracts). Sobol
    ``stderr`` is the usual sample SD — a heuristic, not an IID CLT SE.
    ``method='sobol'`` cannot be combined with ``antithetic=True``.

    ``process`` selects dynamics (GBM default, Heston, or local vol). BS/geo-Asian
    control variates apply only to GBM.
    """

    if method not in ("iid", "sobol"):
        raise ValueError("method must be 'iid' or 'sobol'")
    if method == "sobol" and antithetic:
        raise ValueError("method='sobol' cannot be combined with antithetic=True")
    rng, used_seed = _resolve_rng(rng, seed)
    trial_count = _positive_int(trial_count, "trial_count")
    batch_size = _positive_int(batch_size, "batch_size")
    kind = ContractKind(contract.kind)
    if process is None:
        model = "gbm"
    else:
        model = getattr(process, "model_name", None)
        if model not in ("gbm", "heston", "localvol"):
            raise ValueError(f"unknown process {type(process)!r}")
    if kind in EARLY_EXERCISE_KINDS:
        from monte_carlo_option_engine.american import (
            price_american_call,
            price_american_put,
        )

        american_fn = (
            price_american_call
            if kind is ContractKind.american_call
            else price_american_put
        )
        return american_fn(
            market,
            contract.K,
            steps=steps,
            trial_count=trial_count,
            antithetic=antithetic,
            rng=None if used_seed is not None else rng,
            seed=used_seed,
            process=process,
        )
    if model == "heston":
        return _price_heston_contract(
            market,
            contract,
            process,
            steps,
            trial_count,
            antithetic,
            control_variate,
            estimate_beta,
            method,
            rng,
            used_seed,
        )

    gbm_cv = control_variate and model == "gbm"
    discount = float(np.exp(-market.r * market.T))
    path_like = kind in PATH_KINDS or model == "localvol"
    if path_like:
        steps = _positive_int(steps, "steps")
    expected_y = _cv_expected_y(market, contract, kind, steps, gbm_cv)
    use_cv = expected_y is not None
    stats = _Stats()
    dim = steps if path_like else 1
    source = _NormalSource(dim, method, rng, used_seed)

    remaining = trial_count
    while remaining > 0:
        n = min(batch_size, remaining)
        if kind in TERMINAL_KINDS and model == "gbm":
            _terminal_batch(
                market, contract, kind, n, antithetic, source, discount, stats, use_cv
            )
        elif kind in PATH_KINDS or model == "localvol":
            _path_batch(
                market,
                contract,
                kind,
                n,
                steps,
                antithetic,
                source,
                discount,
                stats,
                use_cv,
                method,
                process,
            )
        else:
            raise ValueError(f"Unsupported contract kind: {kind}")
        remaining -= n

    result = stats.result(
        used_seed, use_cv=use_cv, expected_y=expected_y or 0.0, estimate_beta=estimate_beta
    )
    if result.n_paths != trial_count:
        raise RuntimeError(
            f"internal error: simulated {result.n_paths} paths, expected {trial_count}"
        )
    return result


def print_price(
    market: Market,
    contract: Contract,
    steps: int = 200,
    trial_count: int = 100_000,
    batch_size: int = 50_000,
    antithetic: bool = True,
    control_variate: bool = True,
    estimate_beta: bool = False,
    method: DrawMethod = "iid",
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    process: object | None = None,
) -> PriceResult:
    """Print a formatted price line and return the ``PriceResult``."""

    result = price_mc(
        market,
        contract,
        steps=steps,
        trial_count=trial_count,
        batch_size=batch_size,
        antithetic=antithetic,
        control_variate=control_variate,
        estimate_beta=estimate_beta,
        method=method,
        rng=rng,
        seed=seed,
        process=process,
    )

    def fmt(x: float) -> str:
        return f"{x:,.6f}"

    suffix = (
        " | MC (CV)"
        if control_variate and ContractKind(contract.kind) in CLOSED_FORM_KINDS
        else ""
    )
    print(
        f"{str(contract.kind):<15} price={fmt(result.price)} | "
        f"95% CI [{fmt(result.ci_low)}, {fmt(result.ci_high)}] | "
        f"stderr={fmt(result.stderr)}{suffix}"
    )
    return result
