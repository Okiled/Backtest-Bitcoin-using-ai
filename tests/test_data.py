import pandas as pd
from data.preprocessing import DataPreprocessor

def test_scaler_no_leakage():
    df = pd.DataFrame({
        "open": range(10),
        "high": range(1, 11),
        "low": range(10),
        "close": range(1, 11),
        "volume": range(10)
    })
    pre = DataPreprocessor(train_ratio=0.6, val_ratio=0.2)
    splits, feature_cols, target_col = pre.prepare(df)
    train_means = splits.train[list(feature_cols)].mean()
    val_means = splits.val[list(feature_cols)].mean()
    assert not train_means.equals(val_means)
