# Monte Carlo Option Pricing Engine

A small, self-contained Monte Carlo pricer for common equity options under GBM, built in a Jupyter notebook.

## Features
- Simulates **terminal prices** (fast) or **full paths** (for path-dependent options)
- Prices:
  - `euro_call`, `euro_put`
  - `digital_call`
  - `asian_call`
  - `up_and_out_call`
- **Batching** for large runs + optional **antithetic variates** (variance reduction for terminal-payoff options)
- Returns **price + standard error + 95% confidence interval**
- Optional **Black–Scholes** pricing for European call/put sanity checks
- `market_from_yfinance()` helper to build a market snapshot from live data (spot, dividend yield, historical vol)

## Quick start

```bash
uv venv
uv pip install -e .
```
Open `nbs/ENGINE.ipynb` to run the engine.

## NOTES:

A UV file is included but just incase, the required packages are:
numpy
matplotlib
pandas
dataclasses
yfinance
