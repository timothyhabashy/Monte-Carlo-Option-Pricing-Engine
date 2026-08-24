"""Command-line interface: ``mcoe price ...`` / ``python -m monte_carlo_option_engine``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from monte_carlo_option_engine.black_scholes import black_scholes
from monte_carlo_option_engine.marketdata import market_from_yfinance
from monte_carlo_option_engine.pricer import price_mc
from monte_carlo_option_engine.types import (
    BARRIER_KINDS,
    CLOSED_FORM_KINDS,
    Contract,
    ContractKind,
    Market,
    Monitoring,
    PriceResult,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcoe",
        description="Monte Carlo option pricer",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    price = sub.add_parser("price", help="Price a contract")
    price.add_argument("--kind", required=True, choices=[k.value for k in ContractKind])
    price.add_argument("--ticker", help="Yahoo ticker; fills S, q, and sigma")
    price.add_argument("--S", dest="spot", type=float, help="Spot (synthetic market)")
    price.add_argument("--K", dest="strike", type=float, help="Strike")
    price.add_argument(
        "--moneyness",
        type=float,
        help="Strike as a multiple of spot (used when --K is omitted)",
    )
    price.add_argument("--T", dest="tenor", type=float, required=True, help="Maturity in years")
    price.add_argument("--r", dest="rate", type=float, required=True, help="Risk-free rate")
    price.add_argument(
        "--q",
        dest="div",
        type=float,
        default=None,
        help="Dividend yield (default 0 synthetic; Yahoo yield with --ticker)",
    )
    price.add_argument("--sigma", type=float, help="Volatility (synthetic market)")
    price.add_argument("--B", dest="barrier", type=float, help="Barrier level")
    price.add_argument("--Q", dest="payout", type=float, default=1.0, help="Digital cash payout")
    price.add_argument(
        "--monitoring",
        choices=[m.value for m in Monitoring],
        default=Monitoring.discrete.value,
    )
    price.add_argument("--n", dest="n_paths", type=int, default=50_000)
    price.add_argument("--steps", type=int, default=200)
    price.add_argument("--seed", type=int, default=0)
    price.add_argument("--method", choices=("iid", "sobol"), default="iid")
    price.add_argument(
        "--vol-source",
        choices=("historical", "implied"),
        default="historical",
        help="Yahoo vol for --ticker",
    )
    price.add_argument("--no-antithetic", dest="antithetic", action="store_false")
    price.add_argument("--no-cv", dest="control_variate", action="store_false")
    price.set_defaults(antithetic=True, control_variate=True)
    return parser


def _format_result(contract: Contract, result: PriceResult, bs: float | None) -> str:
    def fmt(x: float) -> str:
        return f"{x:,.6f}"

    lines = [
        f"{str(contract.kind):<15} price={fmt(result.price)} | "
        f"95% CI [{fmt(result.ci_low)}, {fmt(result.ci_high)}] | "
        f"stderr={fmt(result.stderr)}"
    ]
    if bs is not None:
        lines.append(f"BS {contract.kind}: {fmt(bs)}")
    return "\n".join(lines)


def _build_market(args: argparse.Namespace) -> Market:
    if args.ticker:
        if args.spot is not None:
            raise ValueError("pass --ticker or --S, not both")
        strike = args.strike
        snap = market_from_yfinance(
            args.ticker,
            T=args.tenor,
            r=args.rate,
            vol_source=args.vol_source,
            strike=strike,
        )
        if (
            args.vol_source == "implied"
            and strike is None
            and args.moneyness is not None
        ):
            snap = market_from_yfinance(
                args.ticker,
                T=args.tenor,
                r=args.rate,
                vol_source="implied",
                strike=snap.S * args.moneyness,
            )
        q = snap.q if args.div is None else args.div
        sigma = snap.sigma if args.sigma is None else args.sigma
        return Market(S=snap.S, T=args.tenor, r=args.rate, q=q, sigma=sigma)
    if args.spot is None or args.sigma is None:
        raise ValueError("pass --S and --sigma, or --ticker")
    q = 0.0 if args.div is None else args.div
    return Market(S=args.spot, T=args.tenor, r=args.rate, q=q, sigma=args.sigma)


def _build_contract(args: argparse.Namespace, spot: float) -> Contract:
    kind = ContractKind(args.kind)
    if args.strike is not None:
        strike = args.strike
    elif args.moneyness is not None:
        strike = spot * args.moneyness
    else:
        raise ValueError("pass --K or --moneyness")
    if kind in BARRIER_KINDS:
        if args.barrier is None:
            raise ValueError(f"--B is required for {kind}")
        return Contract(
            K=strike,
            kind=kind,
            B=args.barrier,
            Q=args.payout,
            monitoring=args.monitoring,
        )
    return Contract(K=strike, kind=kind, Q=args.payout)


def price_from_args(
    args: argparse.Namespace,
) -> tuple[Contract, PriceResult, float | None]:
    market = _build_market(args)
    contract = _build_contract(args, market.S)
    result = price_mc(
        market,
        contract,
        steps=args.steps,
        trial_count=args.n_paths,
        antithetic=args.antithetic,
        control_variate=args.control_variate,
        method=args.method,
        seed=args.seed,
    )
    bs: float | None = None
    if ContractKind(contract.kind) in CLOSED_FORM_KINDS:
        bs = black_scholes(market, contract)
    return contract, result, bs


def cmd_price(args: argparse.Namespace) -> int:
    try:
        contract, result, bs = price_from_args(args)
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_result(contract, result, bs))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return int(code)
    if args.command == "price":
        return cmd_price(args)
    parser.print_help()
    return 2
