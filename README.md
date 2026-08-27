# Monte Carlo Option Pricing Engine

[![CI](https://github.com/timothyhabashy/Monte-Carlo-Option-Pricing-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/timothyhabashy/Monte-Carlo-Option-Pricing-Engine/actions/workflows/ci.yml)

A Monte Carlo pricer for common equity options. Pricing lives in the
`monte_carlo_option_engine` package (Python 3.13+); notebooks under `nbs/` are
demo clients. One-factor dynamics (GBM, Heston QE, Dupire local vol) share a
`Process` interface so Asians, barriers, lookbacks, and Americans reuse the
same payoffs. Baskets stay two-asset and outside that protocol.

## Features

- Terminal draws for Europeans and cash-or-nothing digitals; path-dependent
  Asians, fixed-strike lookbacks, and the discrete knock-in/out barrier grid
- Batching, antithetic variates (pair-average SEs), and control variates
  (Black–Scholes for vanillas/digitals; geometric Asian for arithmetic Asians)
- Optional scrambled Sobol QMC (`method="sobol"`) with a Brownian bridge on paths.
  Sobol cannot be combined with antithetic draws; reported Sobol stderr is a
  heuristic sample SD, not an IID CLT interval.
- Default vanilla/digital control variates make the Monte Carlo price identical
  to the closed form (stderr ≈ 0). Pass `control_variate=False` to see raw MC.
- Greeks: `black_scholes_greeks`; bump (CRN); pathwise Δ/ν for Europeans and
  pathwise Δ for arithmetic Asians and fixed-strike lookbacks (GBM);
  likelihood ratio for digitals; `heston_greeks_cf` (finite-difference of the
  Heston CF in S, v0, and ρ)
- Barriers: discrete monitoring, or continuous via a BGK shift (Reiner–Rubinstein
  closed form available as `barrier_closed_form`)
- Longstaff–Schwartz American puts and calls (`american_put` / `american_call`),
  even/odd out-of-sample regression, optional `basis="laguerre"`; CRR tree in
  `crr_american_put`; correlated basket / best-of calls
- Heston QE with CF control and Andersen K0 correction; implied-vol surface,
  `calibrate_heston`, and Dupire local vol
- `price_mc(..., process=GBMProcess|HestonProcess|LocalVolProcess)` — default
  `process=None` is GBM. Heston ignores `market.sigma`; Sobol is GBM/local-vol
  only
- CLI `--model {gbm,heston,localvol}` (Heston needs `--kappa --theta --xi --rho --v0`;
  local vol needs `--surface-ticker`)
- `PriceResult`: price, standard error, 95% CI, path count, optional seed
- Closed forms: European call/put, digital call/put, geometric Asian call/put
- `market_from_yfinance(..., vol_source="historical"|"implied")`
- CLI (`mcoe` / `python -m monte_carlo_option_engine`) and optional Streamlit UI

## Install

```bash
git clone https://github.com/timothyhabashy/Monte-Carlo-Option-Pricing-Engine.git
cd Monte-Carlo-Option-Pricing-Engine
uv sync --group dev
```

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

## Quick start

```bash
uv run pytest -m "not network"
```

```python
from monte_carlo_option_engine import (
    Market,
    Contract,
    ContractKind,
    price_mc,
    black_scholes,
)

market = Market(S=100, T=0.5, r=0.04, q=0.01, sigma=0.25)
call = Contract(K=105, kind=ContractKind.euro_call)
result = price_mc(market, call, trial_count=50_000, seed=0)
print(result.price, result.stderr)
print(black_scholes(market, call))
```

Greeks, American puts, Heston calls, and baskets use the same `Market` /
`Contract` types:

```python
from monte_carlo_option_engine import (
    Contract,
    ContractKind,
    HestonParams,
    HestonProcess,
    LocalVolProcess,
    calibrate_heston,
    greeks,
    local_vol_from_surface,
    price_american_put,
    price_mc,
    surface_from_flat,
)

print(greeks(market, call, method="pathwise", trial_count=20_000, seed=0))
print(price_american_put(market, strike=105, seed=0).price)
print(
    price_mc(
        market,
        Contract(105, ContractKind.asian_call),
        process=HestonProcess(
            market, HestonParams(kappa=1.5, theta=0.04, xi=0.5, rho=-0.5, v0=0.04)
        ),
        trial_count=8_000,
        seed=0,
    ).price
)

surface = surface_from_flat(market.S, market.r, market.q, market.sigma)
print(calibrate_heston(surface).params)
print(
    price_mc(
        market,
        call,
        process=LocalVolProcess(market, local_vol_from_surface(surface)),
        trial_count=8_000,
        seed=0,
        control_variate=False,
    ).price
)
```

## Command line

Synthetic market:

```bash
uv run python -m monte_carlo_option_engine price \
  --kind euro_call --S 100 --K 105 --T 0.5 --r 0.04 --q 0.01 --sigma 0.25 --n 100000
```

Yahoo snapshot (`--moneyness` sets strike as a multiple of spot):

```bash
uv run mcoe price --ticker AAPL --kind euro_call --moneyness 1.05 --T 0.5 --r 0.04 --n 100000
```

The command prints the kind, Monte Carlo price, 95% CI, stderr, and a
Black–Scholes line when a closed form exists under GBM. Barrier kinds need
`--B`; `--vol-source implied` interpolates the option chain (falls back to
historical). `--method sobol` turns antithetic off.

```bash
uv run mcoe price --kind euro_call --model heston \
  --S 100 --K 100 --T 0.5 --r 0.03 --kappa 1.5 --theta 0.04 --xi 0.5 --rho -0.5 --v0 0.04 --n 20000
```

Baskets stay on the Python API (`price_basket_call`).

## Streamlit UI

```bash
uv sync --extra ui --group dev
uv run --extra ui streamlit run scripts/streamlit_app.py
```

Sliders for model (GBM / Heston), spot, strike, vol, tenor, and path count; a
contract dropdown (`american_put`, `heston_call`, and the GBM kinds); sample
GBM paths; and a Monte Carlo vs closed-form table when one exists.

## Notebooks

- `nbs/ENGINE.ipynb` — synthetic MC vs BS, path plot, live ticker
- `nbs/01_convergence.ipynb` — European call price vs number of paths
- `nbs/02_variance_reduction.ipynb` — naive vs antithetic vs control-variate SEs

## Tests

CI runs `uv run pytest -m "not network"` on every push and pull request.
Live Yahoo tests are marked `network` and stay off by default.

## Greeks

| Method | Kinds | Process | Notes |
| --- | --- | --- | --- |
| `black_scholes_greeks` | euro call/put | GBM | Δ Γ ν θ ρ |
| `greeks(..., method="bump")` | any `price_mc` kind | same as `price_mc` | CRN finite differences |
| `greeks(..., method="pathwise")` | euro call/put (Δ ν); asian call/put and lookback call (Δ) | GBM | rejects digitals and barriers |
| `greeks(..., method="likelihood_ratio")` | terminal payoffs | GBM | use this for digitals |
| `heston_greeks_cf` | Heston European call | Heston CF | `vega` is ∂C/∂v0; `rho` is ∂C/∂ρ |

## Notes

- Live `sigma` from Yahoo is historical by default; `vol_source="implied"`
  interpolates the option chain and falls back to historical vol if needed.
- Arithmetic Asians average the simulated grid after `t = 0` (they exclude `S0`).
- Up-and-out barriers default to discrete monitoring; set
  `monitoring="continuous"` for the Broadie–Glasserman–Kou barrier shift.

## License

MIT. See [LICENSE](LICENSE).
