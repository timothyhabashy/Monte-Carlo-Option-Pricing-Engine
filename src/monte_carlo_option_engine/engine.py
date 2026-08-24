"""Backward-compatible facade for the core engine entry points."""

from monte_carlo_option_engine.gbm import simulate_paths, simulate_terminal
from monte_carlo_option_engine.pricer import price_mc, print_price

__all__ = [
    "price_mc",
    "print_price",
    "simulate_paths",
    "simulate_terminal",
]
