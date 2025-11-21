"""Data download utilities."""
from __future__ import annotations
from datetime import datetime
from typing import Optional
import pandas as pd
import yfinance as yf
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def download_price_data(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    logger.info("Downloading %s data from %s to %s", symbol, start, end)
    data = yf.download(symbol, start=start, end=end, interval=interval)
    if data.empty:
        raise ValueError("No data downloaded from Yahoo Finance")
    data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    data.index.name = "timestamp"
    return data[["open", "high", "low", "close", "volume"]]
