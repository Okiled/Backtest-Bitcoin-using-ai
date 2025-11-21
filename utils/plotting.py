"""Plotting helpers (minimal placeholders)."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt


def plot_time_series(values: Sequence[float], title: str, path: Path) -> None:
    """Plot a simple time series and save to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(values)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
