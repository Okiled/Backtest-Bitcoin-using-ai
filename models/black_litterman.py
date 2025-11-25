"""Minimal Black-Litterman estimation using prior return and subjective view."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BlackLittermanResult:
    prior_return: float
    posterior_return: float
    prior_vol: float
    posterior_vol: float
    optimal_weight: float


def run_black_litterman(returns: pd.Series, view_return: float = 0.0, tau: float = 0.05) -> BlackLittermanResult:
    """Compute a simple Black-Litterman posterior using a single-asset view."""
    mu = returns.mean()
    sigma = returns.std()
    prior_var = sigma ** 2
    adjusted_mu = (mu / tau + view_return) / (1 / tau + 1)
    posterior_var = prior_var * tau
    weight = adjusted_mu / (posterior_var + 1e-6)
    return BlackLittermanResult(prior_return=mu, posterior_return=adjusted_mu, prior_vol=sigma, posterior_vol=np.sqrt(posterior_var), optimal_weight=float(weight))
