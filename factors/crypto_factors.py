"""Backward-looking factor calculations for BTCUSDT perpetuals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def log_return(series: pd.Series, window: int) -> pd.Series:
    """Compute backward-looking log returns over a window."""
    return np.log(series / series.shift(window))


def funding_zscore(funding_rate: pd.Series, window: int = 30) -> pd.Series:
    rolling_mean = funding_rate.rolling(window).mean()
    rolling_std = funding_rate.rolling(window).std().replace(0, np.nan)
    return (funding_rate - rolling_mean) / rolling_std


def carry_features(funding_rate: pd.Series) -> pd.DataFrame:
    """Create carry/funding related factors using only past data."""
    data = pd.DataFrame({"funding_rate": funding_rate})
    data["funding_roll_mean"] = funding_rate.rolling(7).mean()
    data["funding_z"] = funding_zscore(funding_rate)
    return data


def momentum_features(close: pd.Series) -> pd.DataFrame:
    """Momentum style factors based on log-returns and moving averages."""
    mom_1d = log_return(close, 1)
    mom_7d = log_return(close, 7)
    mom_30d = log_return(close, 30)
    ma_50 = close.rolling(50).mean()
    ma_200 = close.rolling(200).mean()
    return pd.DataFrame(
        {
            "mom_1d": mom_1d,
            "mom_7d": mom_7d,
            "mom_30d": mom_30d,
            "close_ma50_diff": (close - ma_50) / close,
            "close_ma200_diff": (close - ma_200) / close,
        }
    )


def open_interest_features(open_interest: pd.Series) -> pd.DataFrame:
    diff = open_interest.diff()
    rolling = open_interest.rolling(30).mean()
    return pd.DataFrame(
        {
            "open_interest": open_interest,
            "oi_change": diff,
            "oi_norm": open_interest / rolling.replace(0, np.nan),
        }
    )


def compute_crypto_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble factor dataframe using available columns only."""
    factors: list[pd.DataFrame] = [momentum_features(df["close"])]
    if "funding_rate" in df.columns:
        factors.append(carry_features(df["funding_rate"]))
    else:
        zero_funding = pd.Series(0.0, index=df.index)
        factors.append(carry_features(zero_funding))

    if "open_interest" in df.columns:
        factors.append(open_interest_features(df["open_interest"]))
    else:
        zero_oi = pd.Series(0.0, index=df.index)
        factors.append(open_interest_features(zero_oi))

    combined = pd.concat(factors, axis=1)
    return combined.dropna()


@dataclass
class FactorSignal:
    """Rule-based factor signal mapped to [-1, 1] for BTC position sizing."""

    momentum_weight: float = 0.6
    carry_weight: float = 0.2
    oi_weight: float = 0.2

    def score(self, factor_row: Dict[str, float]) -> float:
        mom_score = np.tanh(
            (factor_row.get("mom_7d", 0.0) or 0.0) + 0.5 * (factor_row.get("mom_30d", 0.0) or 0.0)
        )
        carry_score = -np.tanh(factor_row.get("funding_z", 0.0) or 0.0)
        oi_score = np.tanh((factor_row.get("oi_change", 0.0) or 0.0))
        raw = self.momentum_weight * mom_score + self.carry_weight * carry_score + self.oi_weight * oi_score
        return float(np.clip(raw, -1.0, 1.0))

    def target_position(self, factor_row: Dict[str, float], max_position: float) -> float:
        return self.score(factor_row) * max_position
