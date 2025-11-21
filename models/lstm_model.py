"""PyTorch LSTM model with early stopping."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def create_sequences(data: np.ndarray, targets: np.ndarray, look_back: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(targets[i + look_back])
    return np.array(X), np.array(y)


class LSTMRegressor(nn.Module):
    """Simple LSTM regressor."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


@dataclass
class LSTMConfig:
    look_back: int
    hidden_size: int
    num_layers: int
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int
    weight_decay: float


class LSTMTrainer:
    """Trainer handling early stopping and validation."""

    def __init__(self, config: LSTMConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.model: LSTMRegressor | None = None

    def fit(self, train_features: np.ndarray, val_features: np.ndarray, train_targets: np.ndarray, val_targets: np.ndarray) -> LSTMRegressor:
        X_train, y_train = create_sequences(train_features, train_targets, self.config.look_back)
        X_val, y_val = create_sequences(val_features, val_targets, self.config.look_back)

        self.model = LSTMRegressor(
            input_size=train_features.shape[1],
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        loss_fn = nn.MSELoss()
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=self.config.batch_size, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb).squeeze()
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

            val_loss = self.evaluate(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        self.model.load_state_dict(best_state)
        return self.model

    def evaluate(self, loader: DataLoader) -> float:
        assert self.model is not None
        self.model.eval()
        loss_fn = nn.MSELoss()
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb).squeeze()
                losses.append(loss_fn(preds, yb).item())
        return float(np.mean(losses))

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert self.model is not None
        X, _ = create_sequences(features, features[:, 0], self.config.look_back)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy().squeeze()
        return preds
