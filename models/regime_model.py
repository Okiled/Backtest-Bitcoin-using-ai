"""Rule-based regime detection for BTCUSDT."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class RegimeDetector:
    """Deterministic regime classifier based on trend and volatility."""

    vol_threshold: float = 0.04
    ma_window: int = 200
    mom_window: int = 30

    def detect(self, prices: pd.Series) -> pd.Series:
        returns = prices.pct_change()
        vol = returns.rolling(20).std()
        ma = prices.rolling(self.ma_window).mean()
        mom = prices.pct_change(self.mom_window)

        regime = pd.Series(index=prices.index, dtype=int)
        regime[:] = 0
        regime[(vol > self.vol_threshold)] = 3
        regime[(prices > ma) & (mom > 0)] = 1
        regime[(prices < ma) & (mom < 0)] = 2
        regime[(vol > self.vol_threshold) & (mom < 0)] = 3
        return regime.fillna(0)
