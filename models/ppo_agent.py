"""Lightweight PPO-like agent placeholder."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PPOConfig:
    learning_rate: float
    gamma: float
    n_steps: int
    ent_coef: float
    vf_coef: float
    clip_range: float
    batch_size: int
    total_timesteps: int


class SimplePPOAgent:
    """A simplified PPO-style agent that learns mean action from training data."""

    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.action_mean = 0.0

    def train(self, rewards: np.ndarray, actions: np.ndarray) -> None:
        """Learn a simple mean action conditioned on past rewards (toy placeholder)."""
        if len(actions) == 0:
            return
        weighted = actions * (1 + rewards)
        self.action_mean = float(weighted.mean())

    def predict(self, observation: np.ndarray) -> Tuple[int, None]:
        """Return deterministic action based on learned mean sign."""
        action = 1 if self.action_mean >= 0 else 0
        return action, None
