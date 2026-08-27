"""Heston stochastic-volatility Europeans: QE Monte Carlo and a CF price."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from monte_carlo_option_engine.black_scholes import black_scholes
from monte_carlo_option_engine.gbm import _positive_int
from monte_carlo_option_engine.pricer import _Stats, _resolve_rng
from monte_carlo_option_engine.types import Contract, ContractKind, GreeksResult, Market, PriceResult

_PSI_C = 1.5
_XI_EULER = 1e-8


@dataclass(frozen=True)
class HestonParams:
    """Heston dynamics: ``dS = (r-q)S dt + √v S dW1``, ``dv = κ(θ-v)dt + ξ√v dW2``."""

    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    def __post_init__(self) -> None:
        if self.kappa <= 0 or self.theta <= 0 or self.v0 < 0:
            raise ValueError("kappa and theta must be positive; v0 must be non-negative")
        if self.xi < 0:
            raise ValueError("xi must be non-negative")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")


def _qe_moments(
    v: np.ndarray, dt: float, params: HestonParams
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional mean and ψ = var / mean² of V_{t+dt} given V_t."""

    kappa, theta, xi = params.kappa, params.theta, params.xi
    e = np.exp(-kappa * dt)
    mean_v = theta + (v - theta) * e
    var_v = v * xi**2 * e * (1.0 - e) / kappa + theta * xi**2 * (1.0 - e) ** 2 / (
        2.0 * kappa
    )
    mean_v = np.maximum(mean_v, 1e-16)
    psi = var_v / mean_v**2
    return mean_v, psi


def _log_qe_mgf(v: np.ndarray, coeff: float, dt: float, params: HestonParams) -> np.ndarray:
    """``log E[exp(coeff * V_{t+dt}) | V_t]`` under the QE law."""

    mean_v, psi = _qe_moments(v, dt, params)
    out = np.empty_like(v, dtype=float)
    quad = psi <= _PSI_C
    if np.any(quad):
        psi_q = np.clip(psi[quad], 1e-16, None)
        inv = 2.0 / psi_q
        b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
        a = mean_v[quad] / (1.0 + b2)
        two_t = 2.0 * coeff * a
        cap = 1.0 - 1e-12
        two_t = np.minimum(two_t, cap) if coeff > 0 else two_t
        denom = np.maximum(1.0 - two_t, 1e-16)
        out[quad] = -0.5 * np.log(denom) + (coeff * a * b2) / denom
    if np.any(~quad):
        psi_e = psi[~quad]
        p_exp = (psi_e - 1.0) / (psi_e + 1.0)
        beta = (1.0 - p_exp) / mean_v[~quad]
        gap = np.maximum(beta - coeff, 1e-12)
        moment = p_exp + (1.0 - p_exp) * (beta / gap)
        out[~quad] = np.log(np.maximum(moment, 1e-16))
    return out


def _next_variance(
    v: np.ndarray,
    dt: float,
    params: HestonParams,
    z_v: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    mean_v, psi = _qe_moments(v, dt, params)
    v_next = np.empty(v.size, dtype=float)
    quad = psi <= _PSI_C
    if np.any(quad):
        psi_q = np.clip(psi[quad], 1e-16, None)
        inv = 2.0 / psi_q
        b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
        a = mean_v[quad] / (1.0 + b2)
        v_next[quad] = a * (np.sqrt(b2) + z_v[quad]) ** 2
    if np.any(~quad):
        psi_e = psi[~quad]
        p_exp = (psi_e - 1.0) / (psi_e + 1.0)
        beta = (1.0 - p_exp) / mean_v[~quad]
        u_e = u[~quad]
        safe = np.clip(1.0 - u_e, 1e-16, None)
        v_next[~quad] = np.where(
            u_e <= p_exp,
            0.0,
            np.log(np.clip((1.0 - p_exp) / safe, 1e-16, None)) / np.clip(beta, 1e-16, None),
        )
    return np.maximum(v_next, 0.0)


def _qe_step(
    log_s: np.ndarray,
    v: np.ndarray,
    dt: float,
    r: float,
    q: float,
    params: HestonParams,
    z_v: np.ndarray,
    u: np.ndarray,
    z_s: np.ndarray,
    *,
    martingale_correction: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    v = np.maximum(v, 0.0)
    if params.xi < _XI_EULER:
        shock = params.rho * z_v + np.sqrt(max(1.0 - params.rho**2, 0.0)) * z_s
        v_next = np.maximum(v + params.kappa * (params.theta - v) * dt, 0.0)
        log_s = log_s + (r - q - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0.0) * dt) * shock
        return log_s, v_next

    kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
    v_next = _next_variance(v, dt, params, z_v, u)

    gamma1 = 0.5
    gamma2 = 0.5
    k1 = gamma1 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = gamma2 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = gamma1 * dt * (1.0 - rho**2)
    k4 = gamma2 * dt * (1.0 - rho**2)
    vol = np.sqrt(np.maximum(k3 * v + k4 * v_next, 0.0))
    if martingale_correction:
        coeff = k2 + 0.5 * k4
        k0 = -(k1 + 0.5 * k3) * v - _log_qe_mgf(v, coeff, dt, params)
    else:
        k0 = -rho * kappa * theta * dt / xi
    log_s = log_s + (r - q) * dt + k0 + k1 * v + k2 * v_next + vol * z_s
    return log_s, v_next


def _draw_qe_shocks(
    rng: np.random.Generator, steps: int, n_paths: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_v = rng.standard_normal((steps, n_paths))
    u = rng.random((steps, n_paths))
    z_s = rng.standard_normal((steps, n_paths))
    return z_v, u, z_s


def _antithetic_shocks(
    z_v: np.ndarray, u: np.ndarray, z_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return -z_v, 1.0 - u, -z_s


def simulate_heston_terminal(
    market: Market,
    params: HestonParams,
    steps: int,
    n_paths: int,
    rng: np.random.Generator,
    *,
    martingale_correction: bool = True,
    shocks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Terminal spots under Andersen QE (Euler if ``xi`` is essentially 0)."""

    steps = _positive_int(steps, "steps")
    n_paths = _positive_int(n_paths, "n_paths")
    dt = market.T / steps if market.T > 0 else 0.0
    log_s = np.full(n_paths, np.log(market.S), dtype=float)
    v = np.full(n_paths, params.v0, dtype=float)
    if dt == 0.0:
        return np.exp(log_s)
    if shocks is None:
        z_v, u, z_s = _draw_qe_shocks(rng, steps, n_paths)
    else:
        z_v, u, z_s = shocks
        if z_v.shape != (steps, n_paths):
            raise ValueError("shocks must have shape (steps, n_paths)")
    for i in range(steps):
        log_s, v = _qe_step(
            log_s,
            v,
            dt,
            market.r,
            market.q,
            params,
            z_v[i],
            u[i],
            z_s[i],
            martingale_correction=martingale_correction,
        )
    return np.exp(log_s)


def simulate_heston_paths(
    market: Market,
    params: HestonParams,
    steps: int,
    n_paths: int,
    rng: np.random.Generator,
    *,
    martingale_correction: bool = True,
    shocks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """QE paths including ``S0``. Shape ``(steps + 1, n_paths)``."""

    steps = _positive_int(steps, "steps")
    n_paths = _positive_int(n_paths, "n_paths")
    dt = market.T / steps if market.T > 0 else 0.0
    out = np.empty((steps + 1, n_paths), dtype=float)
    out[0] = market.S
    log_s = np.full(n_paths, np.log(market.S), dtype=float)
    v = np.full(n_paths, params.v0, dtype=float)
    if dt == 0.0:
        out[1:] = market.S
        return out
    if shocks is None:
        z_v, u, z_s = _draw_qe_shocks(rng, steps, n_paths)
    else:
        z_v, u, z_s = shocks
    for i in range(steps):
        log_s, v = _qe_step(
            log_s,
            v,
            dt,
            market.r,
            market.q,
            params,
            z_v[i],
            u[i],
            z_s[i],
            martingale_correction=martingale_correction,
        )
        out[i + 1] = np.exp(log_s)
    return out


def _heston_f(
    phi: float,
    j: int,
    spot: float,
    tenor: float,
    r: float,
    q: float,
    params: HestonParams,
) -> complex:
    """Little-Heston-Trap characteristic function piece ``f_j(φ)``."""

    i = 1j
    u = 0.5 if j == 1 else -0.5
    b = params.kappa - params.rho * params.xi if j == 1 else params.kappa
    a = params.kappa * params.theta
    xi = params.xi
    d = np.sqrt(
        (params.rho * xi * i * phi - b) ** 2 - xi**2 * (2.0 * u * i * phi - phi**2)
    )
    g = (b - params.rho * xi * i * phi + d) / (b - params.rho * xi * i * phi - d)
    c = 1.0 / g
    exp_dt = np.exp(-d * tenor)
    d_term = (b - params.rho * xi * i * phi - d) / xi**2
    big_d = d_term * (1.0 - exp_dt) / (1.0 - c * exp_dt)
    big_g = (1.0 - c * exp_dt) / (1.0 - c)
    big_c = (r - q) * i * phi * tenor + (a / xi**2) * (
        (b - params.rho * xi * i * phi - d) * tenor - 2.0 * np.log(big_g)
    )
    return np.exp(big_c + big_d * params.v0 + i * phi * np.log(spot))


def heston_call_cf(
    market: Market, params: HestonParams, strike: float, umax: float = 200.0
) -> float:
    """European call via the Heston P1/P2 integrals (Little Heston Trap)."""

    if strike <= 0:
        raise ValueError("strike must be positive")
    if market.T <= 0:
        return float(max(market.S - strike, 0.0))
    if params.xi < 1e-6:
        bs_market = Market(
            S=market.S,
            T=market.T,
            r=market.r,
            q=market.q,
            sigma=float(np.sqrt(max(params.v0, 1e-16))),
        )
        return black_scholes(bs_market, Contract(strike, ContractKind.euro_call))

    def integrand(phi: float, j: int) -> float:
        if phi == 0.0:
            return 0.0
        val = np.exp(-1j * phi * np.log(strike)) * _heston_f(
            phi, j, market.S, market.T, market.r, market.q, params
        ) / (1j * phi)
        return float(np.real(val))

    i1, _ = quad(lambda p: integrand(p, 1), 1e-10, umax, limit=250, epsabs=1e-8)
    i2, _ = quad(lambda p: integrand(p, 2), 1e-10, umax, limit=250, epsabs=1e-8)
    p1 = 0.5 + i1 / np.pi
    p2 = 0.5 + i2 / np.pi
    return float(
        market.S * np.exp(-market.q * market.T) * p1
        - strike * np.exp(-market.r * market.T) * p2
    )


def heston_put_cf(
    market: Market, params: HestonParams, strike: float, umax: float = 200.0
) -> float:
    """European put via call–put parity on :func:`heston_call_cf`."""

    if strike <= 0:
        raise ValueError("strike must be positive")
    if market.T <= 0:
        return float(max(strike - market.S, 0.0))
    call = heston_call_cf(market, params, strike, umax=umax)
    forward = market.S * np.exp(-market.q * market.T) - strike * np.exp(
        -market.r * market.T
    )
    return float(call - forward)


def price_heston_call(
    market: Market,
    params: HestonParams,
    strike: float,
    steps: int = 64,
    trial_count: int = 20_000,
    antithetic: bool = True,
    control_variate: bool = True,
    estimate_beta: bool = False,
    martingale_correction: bool = True,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> PriceResult:
    """QE Monte Carlo European call under Heston (``market.sigma`` is ignored).

    Default control variate is the characteristic-function call, so the
    reported price matches :func:`heston_call_cf` (stderr ≈ 0), analogous to
    Black–Scholes CV on GBM vanillas. Antithetic pairs flip ``(z_v, u, z_s)``
    to ``(-z_v, 1-u, -z_s)`` on the same ``(S0, v0)``.
    """

    if strike <= 0:
        raise ValueError("strike must be positive")
    steps = _positive_int(steps, "steps")
    trial_count = _positive_int(trial_count, "trial_count")
    rng, used_seed = _resolve_rng(rng, seed)
    discount = float(np.exp(-market.r * market.T))
    stats = _Stats()
    expected_y = heston_call_cf(market, params, strike) if control_variate else None
    use_cv = expected_y is not None

    def pay(spots: np.ndarray) -> np.ndarray:
        return discount * np.maximum(spots - strike, 0.0)

    def simulate(n: int, shocks: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> np.ndarray:
        return simulate_heston_terminal(
            market,
            params,
            steps,
            n,
            rng,
            martingale_correction=martingale_correction,
            shocks=shocks,
        )

    if not antithetic or trial_count == 1:
        spots = simulate(trial_count, None)
        x = pay(spots)
        stats.add_individuals(x, x if use_cv else None)
    else:
        n_pairs = trial_count // 2
        remainder = trial_count % 2
        z_v, u, z_s = _draw_qe_shocks(rng, steps, n_pairs)
        spots = simulate(n_pairs, (z_v, u, z_s))
        z_va, ua, z_sa = _antithetic_shocks(z_v, u, z_s)
        spots_anti = simulate(n_pairs, (z_va, ua, z_sa))
        x = pay(spots)
        x_anti = pay(spots_anti)
        stats.add_pairs(x, x_anti, x if use_cv else None, x_anti if use_cv else None)
        if remainder:
            extra = simulate(1, None)
            xe = pay(extra)
            stats.add_individuals(xe, xe if use_cv else None)
    return stats.result(
        used_seed,
        use_cv=use_cv,
        expected_y=expected_y or 0.0,
        estimate_beta=estimate_beta,
    )


def heston_greeks_cf(
    market: Market,
    params: HestonParams,
    strike: float,
    *,
    d_spot: float | None = None,
    d_v0: float | None = None,
    d_rho: float = 1e-4,
) -> GreeksResult:
    """Finite-difference Greeks of ``heston_call_cf``. ``vega`` is ∂C/∂v0; ``rho`` is ∂C/∂ρ."""

    ds = d_spot if d_spot is not None else max(1e-4 * market.S, 1e-6)
    dv = d_v0 if d_v0 is not None else max(1e-5, 1e-3 * max(params.v0, 1e-8))

    def call_at(
        spot: float | None = None,
        v0: float | None = None,
        rho: float | None = None,
    ) -> float:
        mkt = market if spot is None else Market(
            S=spot, T=market.T, r=market.r, q=market.q, sigma=market.sigma
        )
        par = params
        if v0 is not None or rho is not None:
            par = HestonParams(
                kappa=params.kappa,
                theta=params.theta,
                xi=params.xi,
                rho=params.rho if rho is None else float(np.clip(rho, -0.999, 0.999)),
                v0=params.v0 if v0 is None else v0,
            )
        return heston_call_cf(mkt, par, strike)

    delta = (call_at(spot=market.S + ds) - call_at(spot=market.S - ds)) / (2.0 * ds)
    gamma = (
        call_at(spot=market.S + ds) - 2.0 * call_at() + call_at(spot=market.S - ds)
    ) / (ds**2)
    vega = (call_at(v0=params.v0 + dv) - call_at(v0=max(params.v0 - dv, 1e-8))) / (
        (params.v0 + dv) - max(params.v0 - dv, 1e-8)
    )
    rho = (call_at(rho=params.rho + d_rho) - call_at(rho=params.rho - d_rho)) / (2.0 * d_rho)
    return GreeksResult(delta, gamma, vega, float("nan"), rho, method="heston_cf")
