"""Backtest runner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping

from backtest.environment import TradingEnvironment
from backtest.metrics import compute_performance_metrics
from backtest.strategy import Action, Strategy
from risk.risk_manager import RiskManager


@dataclass
class TradeRecord:
    step: int
    timestamp: object
    price: float
    target_position: float


@dataclass
class BacktestResult:
    equity_curve: list[float]
    trades: list[TradeRecord]
    metrics: Mapping[str, float]


def run_backtest(
    env: TradingEnvironment,
    strategy: Strategy,
    risk_manager: RiskManager | None,
    benchmark_equity: list[float] | None = None,
) -> BacktestResult:
    obs, state = env.reset()
    initial_price = env.current_price()
    if risk_manager:
        risk_manager.reset(state.equity(initial_price))

    equity_curve: List[float] = [state.equity(initial_price)]
    trades: List[TradeRecord] = []
    done = False

    while not done:
        raw_action: Action = strategy.step(state)
        filtered_action = (
            risk_manager.apply(raw_action, state, env.current_price(), env.current_timestamp())
            if risk_manager
            else raw_action
        )
        previous_holdings = state.holdings
        obs, reward, done, info = env.step(filtered_action)
        state = info["state"]  # type: ignore[assignment]
        current_equity = state.equity(env.current_price())
        if risk_manager:
            risk_manager.update(state, float(info["pnl"]), info["timestamp"])  # type: ignore[arg-type]

        equity_curve.append(current_equity)
        if filtered_action.target_position != previous_holdings:
            trades.append(
                TradeRecord(
                    step=state.step,
                    timestamp=env.current_timestamp(),
                    price=env.current_price(),
                    target_position=filtered_action.target_position,
                )
            )

    metrics = compute_performance_metrics(
        equity_curve=equity_curve,
        confidence_level=risk_manager.config.cvar_confidence,
        benchmark_equity=benchmark_equity,
    )

    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)
