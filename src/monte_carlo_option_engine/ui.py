"""Streamlit UI for the Monte Carlo pricer. Streamlit is an optional extra."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from monte_carlo_option_engine.black_scholes import black_scholes
from monte_carlo_option_engine.plotting import plot_paths
from monte_carlo_option_engine.pricer import price_mc
from monte_carlo_option_engine.types import (
    BARRIER_KINDS,
    CLOSED_FORM_KINDS,
    Contract,
    ContractKind,
    Market,
    Monitoring,
)


def main() -> None:
    """Render the pricing app. Call via ``streamlit run scripts/streamlit_app.py``."""

    import streamlit as st

    st.set_page_config(page_title="Monte Carlo Option Engine", layout="centered")
    st.title("Monte Carlo Option Engine")
    st.caption("GBM pricer with optional Black–Scholes comparison.")

    kinds = [k.value for k in ContractKind]
    kind_name = st.selectbox("Contract", kinds, index=kinds.index("euro_call"))
    kind = ContractKind(kind_name)

    col_a, col_b = st.columns(2)
    with col_a:
        spot = st.slider("Spot S", min_value=1.0, max_value=400.0, value=100.0, step=1.0)
        tenor = st.slider("Maturity T (years)", min_value=0.05, max_value=5.0, value=0.5, step=0.05)
        rate = st.slider("Rate r", min_value=0.0, max_value=0.2, value=0.04, step=0.005)
        div = st.slider("Dividend q", min_value=0.0, max_value=0.1, value=0.01, step=0.005)
    with col_b:
        strike = st.slider("Strike K", min_value=1.0, max_value=400.0, value=105.0, step=1.0)
        sigma = st.slider("Vol σ", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
        n_paths = st.slider("Paths", min_value=1_000, max_value=50_000, value=8_000, step=1_000)
        steps = st.slider("Steps", min_value=10, max_value=400, value=100, step=10)

    barrier = None
    if kind in BARRIER_KINDS:
        barrier = st.number_input("Barrier B", min_value=0.01, value=float(spot) * 1.3, step=1.0)
        monitoring_name = st.selectbox(
            "Monitoring", [m.value for m in Monitoring], index=0
        )
    else:
        monitoring_name = Monitoring.discrete.value

    market = Market(S=float(spot), T=float(tenor), r=float(rate), q=float(div), sigma=float(sigma))
    contract = Contract(
        K=float(strike),
        kind=kind,
        B=float(barrier) if barrier is not None else None,
        monitoring=monitoring_name,
    )

    if st.button("Price", type="primary"):
        result = price_mc(
            market,
            contract,
            steps=int(steps),
            trial_count=int(n_paths),
            seed=0,
        )
        rows = [
            {
                "source": "Monte Carlo",
                "price": result.price,
                "stderr": result.stderr,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
            }
        ]
        if kind in CLOSED_FORM_KINDS:
            bs = black_scholes(market, contract)
            rows.append(
                {
                    "source": "Black–Scholes",
                    "price": bs,
                    "stderr": 0.0,
                    "ci_low": bs,
                    "ci_high": bs,
                }
            )
        st.dataframe(rows, hide_index=True)
        st.write(
            f"{kind}  price={result.price:,.6f}  "
            f"95% CI [{result.ci_low:,.6f}, {result.ci_high:,.6f}]  "
            f"stderr={result.stderr:,.6f}"
        )

    fig, _ = plot_paths(
        market, steps=min(int(steps), 80), n_paths=8, rng=np.random.default_rng(0)
    )
    st.pyplot(fig)
    plt.close(fig)
