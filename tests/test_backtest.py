import numpy as np
import pandas as pd
from backtest.environment import TradingEnvironment
from backtest.executor import run_backtest
from backtest.strategy import Action, BuyAndHoldStrategy
from risk.risk_manager import RiskConfig, RiskManager

def test_environment_step():
    prices = pd.Series([100, 101, 102, 103, 104])
    features = np.random.rand(len(prices), 3)
    env = TradingEnvironment(prices=prices, features=features, transaction_cost=0.001, slippage=0.0, initial_cash=1000)
    obs, state = env.reset()
    next_obs, reward, done, state = env.step(Action(target_position=1.0))
    assert len(next_obs) == features.shape[1]
    assert isinstance(reward, float)


def test_backtest_runs():
    prices = pd.Series([100, 99, 101, 102, 103, 104, 105])
    features = np.random.rand(len(prices), 3)
    env = TradingEnvironment(prices=prices, features=features, transaction_cost=0.001, slippage=0.0, initial_cash=1000)
    strategy = BuyAndHoldStrategy()
    risk_manager = RiskManager(
        RiskConfig(
            max_leverage=2.0,
            max_position_btc=2.0,
            max_drawdown=0.5,
            daily_loss_limit=1000.0,
            cvar_confidence=0.95,
        )
    )
    result = run_backtest(env, strategy, risk_manager)
    assert len(result.equity_curve) > 1
