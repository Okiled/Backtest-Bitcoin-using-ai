"""Strategy interface and baseline implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from backtest.environment import TradeState
from factors.crypto_factors import FactorSignal


@dataclass
class Action:
    """Represents a target BTC position in units of the asset."""

    target_position: float
    reduce_only: bool = False


class Strategy(Protocol):
    """Interface for trading strategies used by the backtester."""

    def reset(self) -> None:  # pragma: no cover - protocol
        """Reset any internal state before a new backtest run."""
        raise NotImplementedError

    def step(self, state: TradeState) -> Action:  # pragma: no cover - protocol
        """Return the next action given the current trade state."""
        raise NotImplementedError


class BuyAndHoldStrategy:
    """Buy one BTC on the first step and hold thereafter."""

    def __init__(self, units: float = 1.0) -> None:
        self.units = units
        self._entered = False

    def reset(self) -> None:
        self._entered = False

    def step(self, state: TradeState) -> Action:
        if not self._entered:
            self._entered = True
            return Action(target_position=self.units)
        return Action(target_position=state.holdings)


class AgentStrategy:
    """Wraps an agent with a predict(obs) API to conform to Strategy."""

    def __init__(self, agent, units: float = 1.0) -> None:
        self.agent = agent
        self.units = units

    def reset(self) -> None:
        """Agents typically manage their own reset; nothing to do here."""
        return None

    def step(self, state: TradeState) -> Action:
        observation = state.last_observation
        if observation is None:
            return Action(target_position=state.holdings)
        action, _ = self.agent.predict(observation)
        target = self.units if int(action) == 1 else 0.0
        return Action(target_position=target)


class FactorStrategy:
    """Rule-based factor strategy that maps factor scores to BTC positions."""

    def __init__(self, factor_signal: FactorSignal, feature_names: list[str], max_position: float = 1.0) -> None:
        self.factor_signal = factor_signal
        self.max_position = max_position
        self.feature_names = feature_names

    def reset(self) -> None:
        return None

    def step(self, state: TradeState) -> Action:
        if state.last_observation is None:
            return Action(target_position=0.0)
        factor_row = {name: float(val) for name, val in zip(self.feature_names, state.last_observation)}
        score = self.factor_signal.score(factor_row)
        target = score * self.max_position
        return Action(target_position=target)
