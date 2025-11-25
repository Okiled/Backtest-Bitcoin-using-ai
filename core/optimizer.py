"""Pipeline orchestrator for Bitcoin backtest."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml
import numpy as np
import torch
from data.downloader import download_price_data
from data.preprocessing import DataPreprocessor
from models.lstm_model import LSTMConfig, LSTMTrainer
from models.black_litterman import run_black_litterman
from models.ppo_agent import PPOConfig, SimplePPOAgent
from backtest.environment import TradingEnvironment
from backtest.executor import run_backtest
from backtest.metrics import compare_strategies, compute_performance_metrics
from backtest.strategy import AgentStrategy, BuyAndHoldStrategy, FactorStrategy
from risk.cvar import compute_cvar
from risk.risk_manager import RiskConfig, RiskManager
from utils.plotting import plot_time_series
from utils.logging_utils import get_logger
from factors.crypto_factors import FactorSignal

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    data: dict
    lstm: dict
    ppo: dict
    risk: dict
    backtest: dict

    @staticmethod
    def from_files(base_path: Path) -> "PipelineConfig":
        def load_yaml(name: str) -> dict:
            with open(base_path / "config" / name, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        return PipelineConfig(
            data=load_yaml("data_config.yaml"),
            lstm=load_yaml("lstm_config.yaml"),
            ppo=load_yaml("ppo_config.yaml"),
            risk=load_yaml("risk_config.yaml"),
            backtest=load_yaml("backtest_config.yaml"),
        )


class BitcoinPortfolioOptimizer:
    """High-level orchestrator ensuring no data leakage."""

    def __init__(self, config: PipelineConfig, output_dir: Path, device: torch.device) -> None:
        self.config = config
        self.output_dir = output_dir
        self.device = device
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        data_cfg = self.config.data
        price_df = download_price_data(data_cfg["symbol"], data_cfg["start_date"], data_cfg["end_date"], data_cfg["interval"])

        preprocessor = DataPreprocessor(train_ratio=data_cfg["train_ratio"], val_ratio=data_cfg["val_ratio"])
        splits, feature_cols, target_col = preprocessor.prepare(price_df)

        train_features = splits.train[list(feature_cols)].values
        val_features = splits.val[list(feature_cols)].values
        test_features = splits.test[list(feature_cols)].values
        train_targets = splits.train[target_col].values
        val_targets = splits.val[target_col].values
        test_targets = splits.test[target_col].values

        lstm_cfg = LSTMConfig(**self.config.lstm)
        trainer = LSTMTrainer(lstm_cfg, self.device)
        trainer.fit(train_features, val_features, train_targets, val_targets)
        preds = trainer.predict(test_features)

        bl_result = run_black_litterman(splits.train["return"].dropna())

        cvar_result = compute_cvar(splits.train["return"].dropna(), self.config.risk["cvar_confidence"])
        logger.info("CVaR: %s", cvar_result.cvar)

        # PPO agent uses training rewards to avoid test leakage
        ppo_cfg = PPOConfig(**self.config.ppo)
        agent = SimplePPOAgent(ppo_cfg)
        agent.train(rewards=train_targets, actions=np.ones_like(train_targets))

        env = TradingEnvironment(
            prices=splits.test["close"],
            features=test_features[lstm_cfg.look_back :],
            regimes=splits.test["regime"].values[lstm_cfg.look_back :],
            transaction_cost=self.config.backtest["transaction_cost"],
            slippage=self.config.backtest["slippage"],
            initial_cash=self.config.backtest["initial_cash"],
        )

        risk_cfg = RiskConfig(**self.config.risk)
        risk_manager_bh = RiskManager(config=risk_cfg)
        risk_manager_model = RiskManager(config=risk_cfg)
        risk_manager_factor = RiskManager(config=risk_cfg)

        buy_and_hold_strategy = BuyAndHoldStrategy()
        buy_and_hold_strategy.reset()
        buy_and_hold_result = run_backtest(env, buy_and_hold_strategy, risk_manager_bh)

        env.reset()
        model_strategy = AgentStrategy(agent=agent, units=1.0)
        model_strategy.reset()
        model_result = run_backtest(env, model_strategy, risk_manager_model, benchmark_equity=buy_and_hold_result.equity_curve)

        env.reset()
        factor_strategy = FactorStrategy(factor_signal=FactorSignal(), feature_names=list(feature_cols), max_position=1.0)
        factor_strategy.reset()
        factor_result = run_backtest(env, factor_strategy, risk_manager_factor, benchmark_equity=buy_and_hold_result.equity_curve)

        benchmark_metrics = compute_performance_metrics(
            equity_curve=buy_and_hold_result.equity_curve,
            confidence_level=risk_cfg.cvar_confidence,
        )
        model_metrics = compute_performance_metrics(
            equity_curve=model_result.equity_curve,
            confidence_level=risk_cfg.cvar_confidence,
            benchmark_equity=buy_and_hold_result.equity_curve,
        )
        factor_metrics = compute_performance_metrics(
            equity_curve=factor_result.equity_curve,
            confidence_level=risk_cfg.cvar_confidence,
            benchmark_equity=buy_and_hold_result.equity_curve,
        )

        model_vs_bh = compare_strategies(benchmark_metrics, model_metrics)
        factor_vs_bh = compare_strategies(benchmark_metrics, factor_metrics)

        plot_time_series(model_result.equity_curve, "Model Portfolio Value", self.output_dir / "results" / "portfolio.png")
        plot_time_series(buy_and_hold_result.equity_curve, "Buy & Hold Value", self.output_dir / "results" / "buy_hold.png")
        plot_time_series(factor_result.equity_curve, "Factor Portfolio Value", self.output_dir / "results" / "factor.png")

        logger.info("Buy & Hold metrics: %s", benchmark_metrics)
        logger.info("Model metrics vs benchmark: %s", model_metrics)
        logger.info("Factor metrics vs benchmark: %s", factor_metrics)
        logger.info("Model deltas vs B&H: %s", model_vs_bh.deltas)
        logger.info("Factor deltas vs B&H: %s", factor_vs_bh.deltas)
