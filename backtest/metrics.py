"""Performance metrics."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from risk.cvar import compute_cvar
from dataclasses import dataclass


def total_return(values: Sequence[float]) -> float:
    return float((values[-1] - values[0]) / values[0]) if values else float("nan")


def annualized_return(values: Sequence[float], periods_per_year: int = 252) -> float:
    if not values:
        return float("nan")
    r = total_return(values)
    years = len(values) / periods_per_year
    return float((1 + r) ** (1 / years) - 1) if years > 0 else float("nan")


def volatility(returns: Sequence[float], periods_per_year: int = 252) -> float:
    return float(np.std(returns) * np.sqrt(periods_per_year)) if len(returns) > 0 else float("nan")


def sharpe_ratio(returns: Sequence[float], risk_free_rate: float = 0.0) -> float:
    if len(returns) == 0:
        return float("nan")
    excess = np.array(returns) - risk_free_rate / len(returns)
    denom = np.std(excess)
    return float(np.mean(excess) / denom) if denom != 0 else float("nan")


def sortino_ratio(returns: Sequence[float], risk_free_rate: float = 0.0) -> float:
    if len(returns) == 0:
        return float("nan")
    excess = np.array(returns) - risk_free_rate / len(returns)
    downside = excess[excess < 0]
    downside_vol = np.std(downside)
    return float(np.mean(excess) / downside_vol) if downside_vol != 0 else float("nan")


def max_drawdown(values: Sequence[float]) -> float:
    if len(values) == 0:
        return float("nan")
    arr = np.array(values)
    cum_max = np.maximum.accumulate(arr)
    drawdowns = (arr - cum_max) / cum_max
    return float(drawdowns.min())


def calmar_ratio(values: Sequence[float], periods_per_year: int = 252) -> float:
    dd = max_drawdown(values)
    cagr = annualized_return(values, periods_per_year=periods_per_year)
    return float(cagr / abs(dd)) if dd not in (0, float("nan")) else float("nan")


def information_ratio(returns: Sequence[float], benchmark_returns: Sequence[float]) -> float:
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return float("nan")
    diff = np.array(returns) - np.array(benchmark_returns)
    denom = np.std(diff)
    return float(np.mean(diff) / denom) if denom != 0 else float("nan")


def compute_performance_metrics(
    equity_curve: Sequence[float],
    confidence_level: float,
    periods_per_year: int = 252,
    benchmark_equity: Sequence[float] | None = None,
) -> Mapping[str, float]:
    returns = (
        (np.array(equity_curve[1:]) - np.array(equity_curve[:-1])) / np.array(equity_curve[:-1])
        if len(equity_curve) > 1
        else np.array([])
    )
    benchmark_returns = (
        (np.array(benchmark_equity[1:]) - np.array(benchmark_equity[:-1])) / np.array(benchmark_equity[:-1])
        if benchmark_equity is not None and len(benchmark_equity) > 1
        else np.array([])
    )

    metrics: dict[str, float] = {
        "total_return": total_return(equity_curve),
        "cagr": annualized_return(equity_curve, periods_per_year=periods_per_year),
        "annualized_vol": volatility(returns, periods_per_year=periods_per_year),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar": calmar_ratio(equity_curve, periods_per_year=periods_per_year),
        "cvar": compute_cvar(pd.Series(returns if returns.size else np.array([0.0])), confidence_level).cvar,
    }
    if benchmark_equity is not None:
        metrics["information_ratio"] = information_ratio(returns, benchmark_returns)
    return metrics


@dataclass
class StrategyComparison:
    benchmark: Mapping[str, float]
    model: Mapping[str, float]
    deltas: Mapping[str, float]


def compare_strategies(bh_result: Mapping[str, float], model_result: Mapping[str, float]) -> StrategyComparison:
    """Return metric deltas between model and buy-and-hold."""

    keys = set(bh_result.keys()).union(model_result.keys())
    deltas = {k: model_result.get(k, np.nan) - bh_result.get(k, np.nan) for k in keys}
    return StrategyComparison(benchmark=bh_result, model=model_result, deltas=deltas)
