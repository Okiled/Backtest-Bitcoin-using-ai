"""Data preprocessing with strict train/val/test separation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data.features import compute_rsi, compute_keltner_channels
from factors.crypto_factors import compute_crypto_factors
from models.regime_model import RegimeDetector


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class DataPreprocessor:
    """Prepare time-series data without leakage."""

    def __init__(self, train_ratio: float, val_ratio: float) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI and Keltner Channels using only past data."""
        df = df.copy()
        df["rsi"] = compute_rsi(df["close"])
        kc = compute_keltner_channels(df)
        df = pd.concat([df, kc], axis=1)
        return df.dropna()

    def add_factors_and_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute backward-looking factors and regimes without leakage."""
        factors = compute_crypto_factors(df)
        detector = RegimeDetector()
        regime = detector.detect(df["close"]).reindex(df.index)
        df = df.join(factors, how="inner")
        df["regime"] = regime
        return df.dropna()

    def split(self, df: pd.DataFrame) -> SplitData:
        """Chronological train/val/test split."""
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        return SplitData(train=train, val=val, test=test)

    def scale_features(self, splits: SplitData, feature_cols: Tuple[str, ...], target_col: str) -> SplitData:
        """Fit scalers on train only, transform val/test."""
        train = splits.train.copy()
        val = splits.val.copy()
        test = splits.test.copy()

        train[feature_cols] = self.feature_scaler.fit_transform(train[feature_cols])
        val[feature_cols] = self.feature_scaler.transform(val[feature_cols])
        test[feature_cols] = self.feature_scaler.transform(test[feature_cols])

        train[[target_col]] = self.target_scaler.fit_transform(train[[target_col]])
        val[[target_col]] = self.target_scaler.transform(val[[target_col]])
        test[[target_col]] = self.target_scaler.transform(test[[target_col]])
        return SplitData(train=train, val=val, test=test)

    def prepare(self, df: pd.DataFrame) -> Tuple[SplitData, Tuple[str, ...], str]:
        """Add indicators, split, and scale without leakage."""
        df = df.copy()
        df["return"] = df["close"].pct_change()
        df["target"] = df["close"].shift(-1) / df["close"] - 1
        df = self.add_indicators(df)
        df = self.add_factors_and_regime(df)

        factor_cols = (
            "mom_1d",
            "mom_7d",
            "mom_30d",
            "close_ma50_diff",
            "close_ma200_diff",
            "funding_rate",
            "funding_roll_mean",
            "funding_z",
            "open_interest",
            "oi_change",
            "oi_norm",
        )
        feature_cols = (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "kc_middle",
            "kc_upper",
            "kc_lower",
        ) + factor_cols
        splits = self.split(df)
        scaled = self.scale_features(splits, feature_cols=feature_cols, target_col="target")
        return scaled, feature_cols, "target"
