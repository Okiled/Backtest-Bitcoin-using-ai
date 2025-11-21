"""Trading environment ensuring no look-ahead."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from backtest.strategy import Action


@dataclass
class TradeState:
    """Holds the current portfolio state for the environment."""

    cash: float
    holdings: float
    step: int
    last_observation: np.ndarray | None = None
    regime: int | None = None
    last_price: float | None = None

    def equity(self, price: float) -> float:
        return float(self.cash + self.holdings * price)


class TradingEnvironment:
    """Simple long/flat environment with transaction costs and slippage."""

    def __init__(
        self,
        prices: pd.Series,
        features: np.ndarray,
        regimes: np.ndarray | None,
        transaction_cost: float,
        slippage: float,
        initial_cash: float,
    ) -> None:
        self.prices = prices.reset_index(drop=True)
        self.timestamps = prices.index
        self.features = features
        self.regimes = regimes
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.initial_cash = initial_cash
        self.state = TradeState(cash=initial_cash, holdings=0.0, step=0)

    def reset(self) -> Tuple[np.ndarray, TradeState]:
        self.state = TradeState(cash=self.initial_cash, holdings=0.0, step=0)
        observation = self._observation()
        self.state.last_observation = observation
        self.state.regime = self._current_regime()
        self.state.last_price = self.current_price()
        return observation, self.state

    def _observation(self) -> np.ndarray:
        return self.features[self.state.step]

    def current_price(self) -> float:
        return float(self.prices.iloc[self.state.step])

    def current_timestamp(self) -> pd.Timestamp:
        ts = self.timestamps[self.state.step]
        return pd.Timestamp(ts)

    def _current_regime(self) -> int | None:
        if self.regimes is None:
            return None
        return int(self.regimes[self.state.step])

    def _apply_trade(self, target_position: float, price: float) -> float:
        """Execute trade toward target position, returning trade cost."""

        trade_size = target_position - self.state.holdings
        if abs(trade_size) < 1e-12:
            return 0.0

        cost_multiplier = 1 + self.transaction_cost + self.slippage if trade_size > 0 else 1 - self.transaction_cost - self.slippage
        cash_change = trade_size * price * cost_multiplier
        self.state.cash -= cash_change
        self.state.holdings = target_position
        return float(abs(trade_size) * price * (self.transaction_cost + self.slippage))

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        price = self.current_price()
        done = self.state.step >= len(self.prices) - 2

        pre_trade_equity = self.state.equity(price)
        self._apply_trade(action.target_position, price)

        # advance time
        self.state.step += 1
        next_price = self.current_price()

        post_trade_equity = self.state.equity(next_price)
        reward = float(post_trade_equity - pre_trade_equity)

        observation = self._observation()
        self.state.last_observation = observation
        self.state.regime = self._current_regime()
        self.state.last_price = next_price

        info: Dict[str, object] = {
            "state": self.state,
            "timestamp": self.current_timestamp(),
            "price": next_price,
            "pnl": reward,
            "regime": self.state.regime,
        }
        return observation, reward, done, info
