"""Market, contract, and pricing result types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ContractKind(StrEnum):
    """Supported payoff identifiers."""

    euro_call = "euro_call"
    euro_put = "euro_put"
    digital_call = "digital_call"
    digital_put = "digital_put"
    asian_call = "asian_call"
    asian_put = "asian_put"
    lookback_call = "lookback_call"
    up_and_out_call = "up_and_out_call"
    up_and_in_call = "up_and_in_call"
    down_and_out_call = "down_and_out_call"
    down_and_in_call = "down_and_in_call"
    up_and_out_put = "up_and_out_put"
    up_and_in_put = "up_and_in_put"
    down_and_out_put = "down_and_out_put"
    down_and_in_put = "down_and_in_put"


class Monitoring(StrEnum):
    """Barrier observation convention."""

    discrete = "discrete"
    continuous = "continuous"


TERMINAL_KINDS: frozenset[ContractKind] = frozenset(
    {
        ContractKind.euro_call,
        ContractKind.euro_put,
        ContractKind.digital_call,
        ContractKind.digital_put,
    }
)
PATH_KINDS: frozenset[ContractKind] = frozenset(
    {
        ContractKind.asian_call,
        ContractKind.asian_put,
        ContractKind.lookback_call,
        ContractKind.up_and_out_call,
        ContractKind.up_and_in_call,
        ContractKind.down_and_out_call,
        ContractKind.down_and_in_call,
        ContractKind.up_and_out_put,
        ContractKind.up_and_in_put,
        ContractKind.down_and_out_put,
        ContractKind.down_and_in_put,
    }
)
UP_BARRIER_KINDS: frozenset[ContractKind] = frozenset(
    {
        ContractKind.up_and_out_call,
        ContractKind.up_and_in_call,
        ContractKind.up_and_out_put,
        ContractKind.up_and_in_put,
    }
)
DOWN_BARRIER_KINDS: frozenset[ContractKind] = frozenset(
    {
        ContractKind.down_and_out_call,
        ContractKind.down_and_in_call,
        ContractKind.down_and_out_put,
        ContractKind.down_and_in_put,
    }
)
BARRIER_KINDS: frozenset[ContractKind] = UP_BARRIER_KINDS | DOWN_BARRIER_KINDS
ASIAN_KINDS: frozenset[ContractKind] = frozenset(
    {ContractKind.asian_call, ContractKind.asian_put}
)
CLOSED_FORM_KINDS: frozenset[ContractKind] = frozenset(
    {
        ContractKind.euro_call,
        ContractKind.euro_put,
        ContractKind.digital_call,
        ContractKind.digital_put,
    }
)


@dataclass(frozen=True)
class Market:
    """GBM market snapshot (spot, tenor, rates, vol)."""

    S: float
    T: float
    r: float
    q: float
    sigma: float

    def __post_init__(self) -> None:
        if self.S <= 0:
            raise ValueError("S must be positive")
        if self.T < 0:
            raise ValueError("T must be non-negative")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")


@dataclass(frozen=True)
class Contract:
    """Option contract: strike, kind, optional barrier and digital payout."""

    K: float
    kind: ContractKind | str
    B: float | None = None
    Q: float = 1.0
    monitoring: Monitoring | str = Monitoring.discrete

    def __post_init__(self) -> None:
        kind = (
            self.kind if isinstance(self.kind, ContractKind) else ContractKind(self.kind)
        )
        object.__setattr__(self, "kind", kind)
        monitoring = (
            self.monitoring
            if isinstance(self.monitoring, Monitoring)
            else Monitoring(self.monitoring)
        )
        object.__setattr__(self, "monitoring", monitoring)
        if self.K <= 0:
            raise ValueError("K must be positive")
        if kind in BARRIER_KINDS and self.B is None:
            raise ValueError(f"Barrier B required for {kind}")
        if self.B is not None and self.B <= 0:
            raise ValueError("B must be positive")


@dataclass(frozen=True)
class PriceResult:
    """Monte Carlo price with standard error and a normal 95% CI."""

    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int
    seed: int | None = None


@dataclass(frozen=True)
class GreeksResult:
    """Sensitivities. Unused fields are ``nan`` for a given method."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    stderr_delta: float = 0.0
    stderr_vega: float = 0.0
    method: str = "bump"
