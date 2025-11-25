"""Position sizing utilities."""
from __future__ import annotations


def position_size(cash: float, risk_fraction: float, price: float) -> float:
    """Compute position size based on risk fraction of capital."""
    if price <= 0:
        return 0.0
    return (cash * risk_fraction) / price
