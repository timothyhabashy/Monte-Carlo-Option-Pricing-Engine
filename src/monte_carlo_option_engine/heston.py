"""Heston stochastic-volatility European calls: QE Monte Carlo and a CF price."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from monte_carlo_option_engine.black_scholes import black_scholes
from monte_carlo_option_engine.gbm import _positive_int
from monte_carlo_option_engine.pricer import _Stats, _resolve_rng
from monte_carlo_option_engine.types import Contract, ContractKind, Market, PriceResult

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


def _qe_step(
    log_s: np.ndarray,
    v: np.ndarray,
    dt: float,
    rng: np.random.Generator,
    r: float,
    q: float,
    params: HestonParams,
) -> tuple[np.ndarray, np.ndarray]:
    v = np.maximum(v, 0.0)
    n = v.size
    if params.xi < _XI_EULER:
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        v_next = np.maximum(v + params.kappa * (params.theta - v) * dt, 0.0)
        shock = params.rho * z2 + np.sqrt(max(1.0 - params.rho**2, 0.0)) * z1
        log_s = log_s + (r - q - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0.0) * dt) * shock
        return log_s, v_next

    kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
    e = np.exp(-kappa * dt)
    mean_v = theta + (v - theta) * e
    var_v = v * xi**2 * e * (1.0 - e) / kappa + theta * xi**2 * (1.0 - e) ** 2 / (
        2.0 * kappa
    )
    mean_v = np.maximum(mean_v, 1e-16)
    psi = var_v / mean_v**2

    z_v = rng.standard_normal(n)
    u = rng.random(n)
    v_next = np.empty(n)
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
    v_next = np.maximum(v_next, 0.0)

    gamma1 = 0.5
    gamma2 = 0.5
    k0 = -rho * kappa * theta * dt / xi
    k1 = gamma1 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = gamma2 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = gamma1 * dt * (1.0 - rho**2)
    k4 = gamma2 * dt * (1.0 - rho**2)
    z_s = rng.standard_normal(n)
    vol = np.sqrt(np.maximum(k3 * v + k4 * v_next, 0.0))
    log_s = log_s + (r - q) * dt + k0 + k1 * v + k2 * v_next + vol * z_s
    return log_s, v_next


def simulate_heston_terminal(
    market: Market,
    params: HestonParams,
    steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Terminal spots under Andersen QE (Euler if ``xi`` is essentially 0)."""

    steps = _positive_int(steps, "steps")
    n_paths = _positive_int(n_paths, "n_paths")
    dt = market.T / steps if market.T > 0 else 0.0
    log_s = np.full(n_paths, np.log(market.S), dtype=float)
    v = np.full(n_paths, params.v0, dtype=float)
    if dt == 0.0:
        return np.exp(log_s)
    for _ in range(steps):
        log_s, v = _qe_step(log_s, v, dt, rng, market.r, market.q, params)
    return np.exp(log_s)


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


def price_heston_call(
    market: Market,
    params: HestonParams,
    strike: float,
    steps: int = 64,
    trial_count: int = 20_000,
    antithetic: bool = True,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> PriceResult:
    """QE Monte Carlo European call under Heston (``market.sigma`` is ignored)."""

    if strike <= 0:
        raise ValueError("strike must be positive")
    steps = _positive_int(steps, "steps")
    trial_count = _positive_int(trial_count, "trial_count")
    rng, used_seed = _resolve_rng(rng, seed)
    discount = float(np.exp(-market.r * market.T))
    stats = _Stats()

    def pay(n: int) -> np.ndarray:
        spots = simulate_heston_terminal(market, params, steps, n, rng)
        return discount * np.maximum(spots - strike, 0.0)

    if not antithetic or trial_count == 1:
        stats.add_individuals(pay(trial_count))
    else:
        # Independent paired batches (QE is not a linear Gaussian map).
        n_pairs = trial_count // 2
        remainder = trial_count % 2
        stats.add_individuals(pay(n_pairs))
        stats.add_individuals(pay(n_pairs))
        if remainder:
            stats.add_individuals(pay(1))
    return stats.result(used_seed)
