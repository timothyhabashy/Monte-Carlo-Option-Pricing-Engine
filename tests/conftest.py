import matplotlib

matplotlib.use("Agg")

import pytest

from monte_carlo_option_engine import Market


@pytest.fixture
def market() -> Market:
    return Market(S=100.0, T=0.5, r=0.04, q=0.01, sigma=0.25)


@pytest.fixture
def strike() -> float:
    return 105.0

