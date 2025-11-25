"""Feature engineering without look-ahead bias."""
from __future__ import annotations
import pandas as pd


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI using backward-looking window."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    loss = loss.replace(0, 1e-9)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_keltner_channels(df: pd.DataFrame, ema_span: int = 20, atr_window: int = 10, multiplier: float = 1.5) -> pd.DataFrame:
    """Compute Keltner Channels using historical data only."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    ema = typical_price.ewm(span=ema_span, adjust=False).mean()
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(atr_window).mean()
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    return pd.DataFrame({"kc_middle": ema, "kc_upper": upper, "kc_lower": lower})
