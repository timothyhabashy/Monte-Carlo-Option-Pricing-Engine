"""Build a ``Market`` snapshot from Yahoo Finance."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import interp1d

from monte_carlo_option_engine.surface import ImpliedVolSurface
from monte_carlo_option_engine.types import Market

VolSource = Literal["historical", "implied"]


def _close_series(hist: pd.DataFrame, ticker: str) -> pd.Series:
    close = hist["Close"]
    if isinstance(close, pd.DataFrame):
        close = close[ticker] if ticker in close.columns else close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        raise ValueError(f"No close prices for ticker: {ticker}")
    return close


def _historical_sigma(close: pd.Series, lookback_days: int) -> float:
    lookback = close.iloc[-lookback_days:]
    rets = np.log(lookback / lookback.shift(1)).dropna()
    if len(rets) < 2:
        raise ValueError("Not enough history to estimate volatility")
    sigma = float(rets.std(ddof=1) * np.sqrt(252))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Could not estimate a positive volatility")
    return sigma


def _implied_sigma(ticker_obj: object, tenor: float, spot: float, strike: float) -> float | None:
    expiries = getattr(ticker_obj, "options", None) or ()
    if not expiries:
        return None
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    target = now + pd.Timedelta(days=float(tenor) * 365.25)

    def _expiry_ts(label: str) -> pd.Timestamp:
        return pd.Timestamp(label).tz_localize(None)

    expiry = min(expiries, key=lambda label: abs(_expiry_ts(label) - target))
    chain = ticker_obj.option_chain(expiry)  # type: ignore[attr-defined]
    frames = []
    for frame in (getattr(chain, "calls", None), getattr(chain, "puts", None)):
        if frame is None or getattr(frame, "empty", True):
            continue
        frames.append(frame)
    if not frames:
        return None
    table = pd.concat(frames, ignore_index=True)
    if "strike" not in table.columns or "impliedVolatility" not in table.columns:
        return None
    table = table.dropna(subset=["strike", "impliedVolatility"])
    table = table[table["impliedVolatility"] > 0]
    if table.empty:
        return None
    grouped = table.groupby("strike", sort=True)["impliedVolatility"].mean()
    strikes = grouped.index.to_numpy(dtype=float)
    vols = grouped.to_numpy(dtype=float)
    if strikes.size == 1:
        return float(vols[0])
    interpolant = interp1d(
        strikes, vols, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    iv = float(interpolant(strike))
    if not np.isfinite(iv) or iv <= 0:
        return None
    return iv


def market_from_yfinance(
    ticker: str,
    T: float,
    r: float,
    lookback_days: int = 252,
    interval: str = "1d",
    vol_source: VolSource = "historical",
    strike: float | None = None,
) -> Market:
    """Spot, dividend yield, and volatility from yfinance.

    ``vol_source='historical'`` uses 252-day realized log-return vol.
    ``vol_source='implied'`` interpolates the option-chain implied vol at
    ``strike`` (ATM spot if omitted), then falls back to historical vol if
    the chain is empty. ``r`` is supplied by the caller.
    """

    if vol_source not in ("historical", "implied"):
        raise ValueError("vol_source must be 'historical' or 'implied'")

    hist = yf.download(ticker, period="2y", interval=interval, progress=False)
    if hist.empty:
        raise ValueError(f"No data for ticker: {ticker}")

    close = _close_series(hist, ticker)
    spot = float(close.iloc[-1])
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info if hasattr(ticker_obj, "info") else {}
    q = float((info or {}).get("dividendYield") or 0.0)

    sigma: float | None = None
    if vol_source == "implied":
        iv_strike = float(spot if strike is None else strike)
        try:
            sigma = _implied_sigma(ticker_obj, T, spot, iv_strike)
        except Exception:
            sigma = None
    if sigma is None:
        try:
            sigma = _historical_sigma(close, lookback_days)
        except ValueError as exc:
            raise ValueError(f"{exc} for {ticker}") from exc

    return Market(S=spot, T=T, r=r, q=q, sigma=sigma)


def _otm_iv_slice(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> tuple[np.ndarray, np.ndarray]:
    """OTM implied vols vs strike: puts below spot, calls above; average if both."""

    def _clean(frame: pd.DataFrame, side: str) -> pd.DataFrame:
        if frame is None or getattr(frame, "empty", True):
            return pd.DataFrame(columns=["strike", "impliedVolatility"])
        need = {"strike", "impliedVolatility"}
        if not need.issubset(frame.columns):
            return pd.DataFrame(columns=["strike", "impliedVolatility"])
        table = frame.dropna(subset=["strike", "impliedVolatility"]).copy()
        if "bid" in table.columns and "ask" in table.columns:
            table = table[(table["bid"] > 0) & (table["ask"] >= table["bid"])]
        table = table[
            (table["impliedVolatility"] > 0.01) & (table["impliedVolatility"] < 3.0)
        ]
        if side == "call":
            table = table[table["strike"] >= spot]
        else:
            table = table[table["strike"] <= spot]
        return table[["strike", "impliedVolatility"]]

    parts = [_clean(calls, "call"), _clean(puts, "put")]
    table = pd.concat(parts, ignore_index=True)
    if table.empty:
        return np.array([]), np.array([])
    grouped = table.groupby("strike", sort=True)["impliedVolatility"].mean()
    return grouped.index.to_numpy(dtype=float), grouped.to_numpy(dtype=float)


def surface_from_yfinance(
    ticker: str,
    r: float,
    *,
    max_expiries: int = 8,
    min_tenor: float = 5.0 / 365.0,
    interval: str = "1d",
) -> ImpliedVolSurface:
    """Implied-vol surface from listed Yahoo chains. No historical-vol fallback."""

    hist = yf.download(ticker, period="2y", interval=interval, progress=False)
    if hist.empty:
        raise ValueError(f"No data for ticker: {ticker}")
    close = _close_series(hist, ticker)
    spot = float(close.iloc[-1])
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info if hasattr(ticker_obj, "info") else {}
    q = float((info or {}).get("dividendYield") or 0.0)
    labels = list(getattr(ticker_obj, "options", None) or ())
    if not labels:
        raise ValueError(f"No option expiries for ticker: {ticker}")

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    slices: list[tuple[float, np.ndarray, np.ndarray]] = []
    for label in labels:
        expiry = pd.Timestamp(label).tz_localize(None)
        tenor = float((expiry - now).days) / 365.25
        if tenor < min_tenor:
            continue
        chain = ticker_obj.option_chain(label)
        strikes, vols = _otm_iv_slice(
            getattr(chain, "calls", None), getattr(chain, "puts", None), spot
        )
        if strikes.size < 2:
            continue
        slices.append((tenor, strikes, vols))
        if len(slices) >= max_expiries:
            break
    if len(slices) < 1:
        raise ValueError(f"Could not build an implied-vol surface for {ticker}")

    k_min = min(float(np.log(s[1].min() / spot)) for s in slices)
    k_max = max(float(np.log(s[1].max() / spot)) for s in slices)
    knots = np.linspace(k_min, k_max, 21)
    times = np.array([s[0] for s in slices], dtype=float)
    grid = np.empty((len(slices), knots.size), dtype=float)
    for i, (_tenor, strikes, vols) in enumerate(slices):
        log_k = np.log(strikes / spot)
        order = np.argsort(log_k)
        iv_on_knots = np.interp(knots, log_k[order], vols[order])
        grid[i] = np.clip(iv_on_knots, 1e-4, 3.0) ** 2 * times[i]
    return ImpliedVolSurface(
        S=spot, r=r, q=q, expiries=times, log_moneyness=knots, total_variance=grid
    )
