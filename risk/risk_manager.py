"""Risk manager that enforces portfolio limits before execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtest.environment import TradeState
from backtest.strategy import Action


@dataclass
class RiskConfig:
    """Configuration for runtime risk limits."""

    max_leverage: float
    max_position_btc: float
    max_drawdown: float
    daily_loss_limit: float
    cvar_confidence: float
    target_vol: Optional[float] = None
    vol_lookback: int = 20
    position_change_limit: float = 0.2
    regime_position_scale: Dict[int, float] = field(default_factory=lambda: {2: 0.5, 3: 0.0})


@dataclass
class RiskManager:
    """Tracks equity and applies limits to strategy actions."""

    config: RiskConfig
    _initial_equity: float | None = None
    _equity_history: list[float] = field(default_factory=list)
    _daily_start_equity: dict[pd.Timestamp, float] = field(default_factory=dict)

    def reset(self, initial_equity: float) -> None:
        """Reset risk state for a fresh backtest run."""
        self._initial_equity = initial_equity
        self._equity_history = [initial_equity]
        self._daily_start_equity = {}

    def update(self, portfolio_state: TradeState, pnl_today: float, timestamp: pd.Timestamp) -> None:
        equity = portfolio_state.equity(portfolio_state.last_price)
        self._equity_history.append(equity)
        day = timestamp.normalize()
        self._daily_start_equity.setdefault(day, equity - pnl_today)

    def _current_drawdown(self, equity: float) -> float:
        peak = max(self._equity_history) if self._equity_history else equity
        return (equity - peak) / peak if peak else 0.0

    def _realized_vol(self) -> float:
        if len(self._equity_history) <= 1:
            return 0.0
        returns = np.diff(self._equity_history[-self.config.vol_lookback :]) / np.array(
            self._equity_history[-self.config.vol_lookback - 1 : -1]
        )
        return float(np.std(returns) * np.sqrt(252)) if returns.size > 0 else 0.0

    def _vol_scale(self) -> float:
        if self.config.target_vol is None:
            return 1.0
        realized_vol = self._realized_vol()
        if realized_vol <= 0:
            return 1.0
        return min(1.0, self.config.target_vol / realized_vol)

    def _daily_pnl(self, equity: float, timestamp: pd.Timestamp) -> float:
        day = timestamp.normalize()
        start = self._daily_start_equity.get(day, equity)
        return equity - start

    def _regime_scale(self, regime: int | None) -> float:
        if regime is None:
            return 1.0
        return self.config.regime_position_scale.get(regime, 1.0)

    def apply(self, raw_action: Action, portfolio_state: TradeState, price: float, timestamp: pd.Timestamp) -> Action:
        if self._initial_equity is None:
            self.reset(portfolio_state.equity(price))

        equity = portfolio_state.equity(price)
        drawdown = self._current_drawdown(equity)
        if drawdown <= -abs(self.config.max_drawdown):
            return Action(target_position=0.0, reduce_only=True)

        pnl_today = self._daily_pnl(equity, timestamp)
        if pnl_today <= -abs(self.config.daily_loss_limit):
            return Action(target_position=0.0, reduce_only=True)

        max_position_by_leverage = (equity * self.config.max_leverage) / price
        allowed_position = min(self.config.max_position_btc, max_position_by_leverage)

        vol_scaled = raw_action.target_position * self._vol_scale()
        regime_scaled = vol_scaled * self._regime_scale(portfolio_state.regime)
        clipped_position = float(np.clip(regime_scaled, -allowed_position, allowed_position))

        max_delta = allowed_position * self.config.position_change_limit
        desired_change = clipped_position - portfolio_state.holdings
        if abs(desired_change) > max_delta and max_delta > 0:
            clipped_position = portfolio_state.holdings + np.sign(desired_change) * max_delta

        portfolio_state.last_price = price  # type: ignore[attr-defined]
        return Action(target_position=clipped_position, reduce_only=raw_action.reduce_only)
