"""Conditional Value at Risk calculations."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CvarResult:
    level: float
    cvar: float


def compute_cvar(returns: pd.Series, confidence_level: float) -> CvarResult:
    """Compute CVaR using historical simulation."""
    quantile = returns.quantile(1 - confidence_level)
    tail_losses = returns[returns <= quantile]
    cvar = tail_losses.mean() if not tail_losses.empty else float("nan")
    return CvarResult(level=confidence_level, cvar=float(cvar))
