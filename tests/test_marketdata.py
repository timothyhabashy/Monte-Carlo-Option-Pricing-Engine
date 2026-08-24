from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from monte_carlo_option_engine import market_from_yfinance


def _close_history(n: int = 300, start: float = 100.0, end: float = 110.0) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = pd.Series(np.linspace(start, end, n), index=idx, name="Close")
    return pd.DataFrame({"Close": close})


@patch("monte_carlo_option_engine.marketdata.yf.Ticker")
@patch("monte_carlo_option_engine.marketdata.yf.download")
def test_market_from_yfinance_mocked(
    mock_download: MagicMock, mock_ticker_cls: MagicMock
) -> None:
    mock_download.return_value = _close_history()
    ticker = MagicMock()
    ticker.info = {"dividendYield": 0.012}
    mock_ticker_cls.return_value = ticker

    market = market_from_yfinance("AAPL", T=0.5, r=0.04)

    assert market.S == pytest.approx(110.0)
    assert market.T == 0.5
    assert market.r == 0.04
    assert market.q == pytest.approx(0.012)
    assert market.sigma > 0.0
    mock_download.assert_called_once()
    mock_ticker_cls.assert_called_once_with("AAPL")


@patch("monte_carlo_option_engine.marketdata.yf.Ticker")
@patch("monte_carlo_option_engine.marketdata.yf.download")
def test_missing_dividend_defaults_to_zero(
    mock_download: MagicMock, mock_ticker_cls: MagicMock
) -> None:
    mock_download.return_value = _close_history()
    ticker = MagicMock()
    ticker.info = {}
    mock_ticker_cls.return_value = ticker
    market = market_from_yfinance("AAPL", T=1.0, r=0.03)
    assert market.q == 0.0


@patch("monte_carlo_option_engine.marketdata.yf.download")
def test_empty_history_raises(mock_download: MagicMock) -> None:
    mock_download.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="No data"):
        market_from_yfinance("ZZZZ", T=0.5, r=0.04)


@pytest.mark.network
def test_live_aapl_snapshot() -> None:
    market = market_from_yfinance("AAPL", T=0.5, r=0.04)
    assert market.S > 0
    assert market.sigma > 0


@patch("monte_carlo_option_engine.marketdata.yf.Ticker")
@patch("monte_carlo_option_engine.marketdata.yf.download")
def test_implied_vol_interpolated_at_strike(
    mock_download: MagicMock, mock_ticker_cls: MagicMock
) -> None:
    mock_download.return_value = _close_history()
    ticker = MagicMock()
    ticker.info = {"dividendYield": 0.01}
    ticker.options = ("2026-09-18",)
    chain = MagicMock()
    chain.calls = pd.DataFrame(
        {"strike": [100.0, 110.0, 120.0], "impliedVolatility": [0.40, 0.30, 0.22]}
    )
    chain.puts = pd.DataFrame(
        {"strike": [100.0, 110.0, 120.0], "impliedVolatility": [0.40, 0.30, 0.22]}
    )
    ticker.option_chain.return_value = chain
    mock_ticker_cls.return_value = ticker

    market = market_from_yfinance(
        "AAPL", T=0.5, r=0.04, vol_source="implied", strike=110.0
    )
    assert market.sigma == pytest.approx(0.30)
    assert market.S == pytest.approx(110.0)


@patch("monte_carlo_option_engine.marketdata.yf.Ticker")
@patch("monte_carlo_option_engine.marketdata.yf.download")
def test_implied_vol_falls_back_to_historical(
    mock_download: MagicMock, mock_ticker_cls: MagicMock
) -> None:
    mock_download.return_value = _close_history()
    ticker = MagicMock()
    ticker.info = {"dividendYield": 0.0}
    ticker.options = ()
    mock_ticker_cls.return_value = ticker
    implied = market_from_yfinance("AAPL", T=0.5, r=0.04, vol_source="implied")
    historical = market_from_yfinance("AAPL", T=0.5, r=0.04, vol_source="historical")
    assert implied.sigma == pytest.approx(historical.sigma)
