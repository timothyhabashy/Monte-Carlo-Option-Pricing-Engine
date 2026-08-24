# Monte Carlo Option Pricing Engine

A GBM Monte Carlo pricer for common equity options. The engine lives in the
`monte_carlo_option_engine` package; notebooks under `nbs/` are demo clients.

## Features

- Terminal draws for Europeans and cash-or-nothing digitals; path-dependent
  Asians, fixed-strike lookbacks, and the discrete knock-in/out barrier grid
- Batching, antithetic variates (pair-average SEs), and control variates
  (Black–Scholes for vanillas/digitals; geometric Asian for arithmetic Asians)
- Optional scrambled Sobol QMC (`method="sobol"`) with a Brownian bridge on paths
- Greeks: bump (CRN), pathwise Δ/ν for Europeans, likelihood ratio for digitals
- Barriers: discrete monitoring, or continuous via a BGK shift
- Longstaff–Schwartz American puts; correlated basket / best-of calls; Heston QE
- `PriceResult`: price, standard error, 95% CI, path count, optional seed
- Closed forms: European call/put, digital call/put, geometric Asian call/put
- `market_from_yfinance(..., vol_source="historical"|"implied")`
- CLI (`mcoe` / `python -m monte_carlo_option_engine`) and optional Streamlit UI

## Quick start

```bash
uv sync --group dev
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
Black–Scholes line when a closed form exists. Barrier kinds need `--B`;
`--vol-source implied` interpolates the option chain (falls back to historical).

American puts, Heston, and baskets stay on the Python API (`price_american_put`,
`price_heston_call`, `price_basket_call`).

## Streamlit UI

```bash
uv sync --extra ui --group dev
uv run --extra ui streamlit run scripts/streamlit_app.py
```

Sliders for spot, strike, vol, tenor, and path count; a contract dropdown;
sample GBM paths; and a Monte Carlo vs Black–Scholes table when a closed form
exists.

## Notebooks

- `nbs/ENGINE.ipynb` — synthetic MC vs BS, path plot, live ticker
- `nbs/01_convergence.ipynb` — European call price vs number of paths
- `nbs/02_variance_reduction.ipynb` — naive vs antithetic vs control-variate SEs

## Notes

- Live `sigma` from Yahoo is historical by default; `vol_source="implied"`
  interpolates the option chain and falls back to historical vol if needed.
- Arithmetic Asians average the simulated grid after `t = 0` (they exclude `S0`).
- Up-and-out barriers default to discrete monitoring; set
  `monitoring="continuous"` for the Broadie–Glasserman–Kou barrier shift.
