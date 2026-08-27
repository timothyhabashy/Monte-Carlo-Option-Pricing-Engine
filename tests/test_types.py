import pytest

from monte_carlo_option_engine import Contract, ContractKind, Market
from monte_carlo_option_engine.types import EARLY_EXERCISE_KINDS, PATH_KINDS, TERMINAL_KINDS


def test_version_is_string() -> None:
    from monte_carlo_option_engine import __version__

    assert isinstance(__version__, str)
    assert __version__ == "0.9.0"


def test_negative_spot_rejected() -> None:
    with pytest.raises(ValueError, match="S must be positive"):
        Market(S=-1.0, T=1.0, r=0.0, q=0.0, sigma=0.2)


def test_negative_tenor_rejected() -> None:
    with pytest.raises(ValueError, match="T must be non-negative"):
        Market(S=100.0, T=-0.1, r=0.0, q=0.0, sigma=0.2)


def test_nonpositive_sigma_rejected() -> None:
    with pytest.raises(ValueError, match="sigma must be positive"):
        Market(S=100.0, T=1.0, r=0.0, q=0.0, sigma=0.0)


def test_nonpositive_strike_rejected() -> None:
    with pytest.raises(ValueError, match="K must be positive"):
        Contract(K=0.0, kind=ContractKind.euro_call)


def test_unknown_kind_rejected() -> None:
    with pytest.raises(ValueError):
        Contract(K=100.0, kind="not_a_kind")


def test_kinds_are_partitioned() -> None:
    assert set(TERMINAL_KINDS | PATH_KINDS | EARLY_EXERCISE_KINDS) == set(ContractKind)
    assert TERMINAL_KINDS.isdisjoint(PATH_KINDS)
    assert TERMINAL_KINDS.isdisjoint(EARLY_EXERCISE_KINDS)
    assert PATH_KINDS.isdisjoint(EARLY_EXERCISE_KINDS)
