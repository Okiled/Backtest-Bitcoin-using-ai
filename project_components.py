import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import os
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')
import matplotlib.ticker as mtick
from scipy import linalg
import pickle
import json
from pathlib import Path

class DataProcessor:
    def __init__(self):
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.lstm_pred_scaler = MinMaxScaler(feature_range=(0, 1))
        self.indicators = {}

    def fetch_yahoo_data(self, symbol='BTC-USD', interval='1d', start_date_str='2019-01-01', end_date_str='2025-01-01'):
        """Fetch Bitcoin data from Yahoo Finance"""
        try:
            data = yf.download(symbol, start=start_date_str, end=end_date_str, interval=interval)
            if data.empty:
                return None
            
            all_data = []
            for timestamp, row in data.iterrows():
                all_data.append({
                    'timestamp': timestamp, 
                    'open': float(row['Open']), 
                    'high': float(row['High']),
                    'low': float(row['Low']), 
                    'close': float(row['Close']), 
                    'volume': float(row['Volume'])
                })
            
            df = pd.DataFrame(all_data)
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def preprocess_data(self, df, lstm_predictions_pct_change=None):
        """Optimized preprocessing maintaining Indicators -> LSTM -> BL -> CVaR -> PPO pipeline"""
        if df is None or df.empty:
            return pd.DataFrame()
            
        processed_df = df.copy()
        processed_df = processed_df.fillna(method='ffill').fillna(method='bfill')
        
        # Core data preparation
        processed_df['close_unscaled'] = processed_df['close']
        processed_df['target_pct_change'] = processed_df['close'].pct_change().shift(-1)
        processed_df['daily_return'] = processed_df['close'].pct_change()
        
        # STEP 1: Calculate indicators ONCE (efficient)
        self._calculate_indicators(processed_df)
        
        # STEP 2: Join indicators efficiently
        if 'rsi' in self.indicators:
            processed_df = processed_df.join(self.indicators['rsi'])
        if 'keltner_channels' in self.indicators:
            processed_df = processed_df.join(self.indicators['keltner_channels'])
        
        # STEP 3: Add LSTM predictions for pipeline continuity
        if lstm_predictions_pct_change is not None:
            processed_df = processed_df.merge(
                lstm_predictions_pct_change, 
                left_index=True, 
                right_index=True, 
                how='left'
            )
            if 'lstm_pred_pct_change' in processed_df.columns:
                processed_df['lstm_pred_pct_change'].fillna(0, inplace=True)
                if not processed_df[['lstm_pred_pct_change']].isnull().all().all():
                    processed_df[['lstm_pred_pct_change']] = self.lstm_pred_scaler.fit_transform(
                        processed_df[['lstm_pred_pct_change']]
                    )
            else:
                processed_df['lstm_pred_pct_change'] = 0.0
        else:
            processed_df['lstm_pred_pct_change'] = 0.0
        
        # STEP 4: Prepare features for scaling (consolidated approach)
        feature_cols_to_scale = ['open', 'high', 'low', 'close', 'volume']
        
        # Add RSI features systematically
        if 'result_rsi' in processed_df.columns:
            feature_cols_to_scale.append('result_rsi')
        
        # Add Keltner Channel features systematically  
        kc_features = ['kc_upper', 'kc_middle', 'kc_lower']
        for col in kc_features:
            if col in processed_df.columns:
                feature_cols_to_scale.append(col)
        
        if 'kc_trend_strength' in processed_df.columns:
            feature_cols_to_scale.append('kc_trend_strength')
        
        # STEP 5: Convert categorical features to numeric (batch operation)
        categorical_mappings = {
            'kc_trend': {'Neutral': 0, 'Uptrend': 1, 'Downtrend': -1},
            'kc_trend_retest': {'No_Retest': 0, 'Uptrend_Retest': 1, 'Downtrend_Retest': -1},
            'result_rsi_status': {'Neutral': 0, 'Oversold': -1, 'Overbought': 1}
        }
        
        for col, mapping in categorical_mappings.items():
            if col in processed_df.columns:
                processed_df[f'{col}_numeric'] = processed_df[col].map(mapping)
                feature_cols_to_scale.append(f'{col}_numeric')
        
        # Convert boolean RSI features to numeric (batch operation)
        bool_rsi_features = ['result_rsi_double_bottom', 'result_rsi_bullish_div', 
                            'result_rsi_bearish_div', 'result_rsi_pivot_high', 'result_rsi_pivot_low']
        for col in bool_rsi_features:
            if col in processed_df.columns:
                processed_df[f'{col}_numeric'] = processed_df[col].astype(int)
                feature_cols_to_scale.append(f'{col}_numeric')
        
        # STEP 6: Clean data (consolidated)
        cols_to_dropna = feature_cols_to_scale + ['daily_return', 'target_pct_change', 'lstm_pred_pct_change']
        processed_df.dropna(subset=cols_to_dropna, inplace=True)
        
        if processed_df.empty:
            return processed_df
        
        # STEP 7: Scale features (single operation)
        if len(feature_cols_to_scale) > 0:
            processed_df[feature_cols_to_scale] = self.feature_scaler.fit_transform(processed_df[feature_cols_to_scale])
        
        # STEP 8: Scale target
        if not processed_df[['target_pct_change']].isnull().all().all():
            processed_df[['target_pct_change']] = self.target_scaler.fit_transform(processed_df[['target_pct_change']])
        
        # STEP 9: Final validation
        final_check_cols = feature_cols_to_scale + ['lstm_pred_pct_change']
        if len(final_check_cols) > 0 and processed_df[final_check_cols].isnull().any().any():
            processed_df.dropna(subset=final_check_cols, inplace=True)
        
        return processed_df

    def _calculate_indicators(self, df):
        """Calculate all technical indicators"""
        self.indicators['rsi'] = self.calculate_rsi_standard(df, window=14)
        self.indicators['keltner_channels'] = self.calculate_keltner_channels(df, length=36, mult=0.5, atr_length=10)
        return self.indicators

    def calculate_rsi_standard(self, df, window=14):
        """Calculate RSI with additional patterns"""
        close = df['close']
        delta = close.diff()
        
        if (delta == 0).all():
            rsi = pd.Series(50, index=close.index)
        else:
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            loss = loss.replace(0, 1e-9)
            rs = gain / loss
            
            rsi = 100 - (100 / (1 + rs))
            
        rsi.fillna(50, inplace=True)
        
        # RSI status
        rsi_status = pd.Series("Neutral", index=rsi.index)
        rsi_status[rsi > 70] = "Overbought"
        rsi_status[rsi < 30] = "Oversold"
        
        # Calculate RSI patterns
        rsi_pivot_high = self.find_pivot_high(rsi)
        rsi_pivot_low = self.find_pivot_low(rsi)
        rsi_double_bottom = self.detect_rsi_double_bottom(rsi)
        bullish_div, bearish_div = self.calculate_rsi_divergence(df, rsi)
        
        result_rsi_df = pd.DataFrame({
            'result_rsi': rsi,
            'result_rsi_status': rsi_status,
            'result_rsi_double_bottom': rsi_double_bottom,
            'result_rsi_bullish_div': bullish_div,
            'result_rsi_bearish_div': bearish_div,
            'result_rsi_pivot_high': rsi_pivot_high,
            'result_rsi_pivot_low': rsi_pivot_low
        })
        
        return result_rsi_df

    def find_pivot_high(self, data, left_bars=5, right_bars=5):
        """Find pivot high points"""
        result = pd.Series(False, index=data.index)
        
        if len(data) < left_bars + right_bars + 1:
            return result
            
        for i in range(left_bars, len(data) - right_bars):
            is_pivot = True
            current_value = data.iloc[i]
            
            # Check left side
            for j in range(1, left_bars + 1):
                if data.iloc[i - j] >= current_value:
                    is_pivot = False
                    break
                    
            # Check right side
            if is_pivot:
                for j in range(1, right_bars + 1):
                    if data.iloc[i + j] >= current_value:
                        is_pivot = False
                        break
                        
            result.iloc[i] = is_pivot
            
        return result

    def find_pivot_low(self, data, left_bars=5, right_bars=5):
        """Find pivot low points"""
        result = pd.Series(False, index=data.index)
        
        if len(data) < left_bars + right_bars + 1:
            return result
            
        for i in range(left_bars, len(data) - right_bars):
            is_pivot = True
            current_value = data.iloc[i]
            
            # Check left side
            for j in range(1, left_bars + 1):
                if data.iloc[i - j] <= current_value:
                    is_pivot = False
                    break
                    
            # Check right side
            if is_pivot:
                for j in range(1, right_bars + 1):
                    if data.iloc[i + j] <= current_value:
                        is_pivot = False
                        break
                        
            result.iloc[i] = is_pivot
            
        return result

    def detect_rsi_double_bottom(self, rsi, window_size=10, threshold=35):
        """Detect RSI double bottom pattern"""
        rsi_double_bottom = pd.Series(False, index=rsi.index)
        
        if len(rsi) < window_size + 1:
            return rsi_double_bottom
            
        for i in range(window_size, len(rsi)):
            segment = rsi[i-window_size:i+1]
            local_mins = []
            
            for j in range(1, len(segment)-1):
                if segment.iloc[j] < segment.iloc[j-1] and segment.iloc[j] < segment.iloc[j+1]:
                    local_mins.append((j, segment.iloc[j]))
                    
            if len(local_mins) >= 2:
                min_points = sorted(local_mins, key=lambda x: x[1])[:2]
                
                if len(min_points) == 2:
                    first_min, second_min = min_points[0], min_points[1]
                    
                    if second_min[1] > first_min[1] and first_min[1] < threshold:
                        rsi_double_bottom.iloc[i] = True
                        
        return rsi_double_bottom

    def calculate_rsi_divergence(self, df, rsi, lookback_left=5, lookback_right=5, range_lower=5, range_upper=60):
        """Calculate RSI divergence patterns"""
        rsi_pivot_high = self.find_pivot_high(rsi, lookback_left, lookback_right)
        rsi_pivot_low = self.find_pivot_low(rsi, lookback_left, lookback_right)
        
        bullish_div = pd.Series(False, index=rsi.index)
        bearish_div = pd.Series(False, index=rsi.index)
        
        if 'low' not in df.columns or 'high' not in df.columns:
            return bullish_div, bearish_div
            
        # Bullish divergence
        for i in range(lookback_left + lookback_right + 1, len(rsi)):
            if rsi_pivot_low.iloc[i - lookback_right]:
                j = i - 1
                while j >= lookback_left + lookback_right:
                    if rsi_pivot_low.iloc[j - lookback_right]:
                        bars_since = i - j
                        
                        if range_lower <= bars_since <= range_upper:
                            if rsi.iloc[i - lookback_right] > rsi.iloc[j - lookback_right]:
                                if df['low'].iloc[i - lookback_right] < df['low'].iloc[j - lookback_right]:
                                    bullish_div.iloc[i] = True
                        break
                    j -= 1
                    
        # Bearish divergence
        for i in range(lookback_left + lookback_right + 1, len(rsi)):
            if rsi_pivot_high.iloc[i - lookback_right]:
                j = i - 1
                while j >= lookback_left + lookback_right:
                    if rsi_pivot_high.iloc[j - lookback_right]:
                        bars_since = i - j
                        
                        if range_lower <= bars_since <= range_upper:
                            if rsi.iloc[i - lookback_right] < rsi.iloc[j - lookback_right]:
                                if df['high'].iloc[i - lookback_right] > df['high'].iloc[j - lookback_right]:
                                    bearish_div.iloc[i] = True
                        break
                    j -= 1
                    
        return bullish_div, bearish_div

    def calculate_ema(self, prices, window=12):
        """Calculate Exponential Moving Average"""
        if prices.empty:
            return pd.Series(index=prices.index)
            
        return prices.ewm(span=window, adjust=False).mean()

    def calculate_sma(self, prices, window=20):
        """Calculate Simple Moving Average"""
        if prices.empty:
            return pd.Series(index=prices.index)
            
        return prices.rolling(window=window).mean()

    def calculate_atr(self, high, low, close, length=14):
        """Calculate Average True Range"""
        if high.empty or low.empty or close.empty:
            return pd.Series(index=high.index)
            
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=length).mean()
        
        return atr

    def calculate_keltner_channels(self, df, length=36, mult=0.5, atr_length=10, use_ema=True):
        """Calculate Keltner Channels with trend analysis"""
        required_cols = ['close', 'high', 'low']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame(index=df.index)
            
        if use_ema:
            middle = self.calculate_ema(df['close'], length)
        else:
            middle = self.calculate_sma(df['close'], length)
            
        atr = self.calculate_atr(df['high'], df['low'], df['close'], atr_length)
        
        upper = middle + (mult * atr)
        lower = middle - (mult * atr)
        
        kc_df = pd.DataFrame({
            'kc_upper': upper,
            'kc_middle': middle,
            'kc_lower': lower
        })
        
        close = df['close']
        
        # Trend identification
        trend = pd.Series("Neutral", index=df.index)
        trend[(close > upper)] = "Uptrend"
        trend[(close < lower)] = "Downtrend"
        
        # Retest patterns
        uptrend_retest = (close.shift(1) > upper.shift(1)) & \
                        ((close <= upper) & (close >= middle)) & \
                        (trend.shift(1) == "Uptrend")
                        
        downtrend_retest = (close.shift(1) < lower.shift(1)) & \
                          ((close >= lower) & (close <= middle)) & \
                          (trend.shift(1) == "Downtrend")
                          
        trend_retest = pd.Series("No_Retest", index=df.index)
        trend_retest[uptrend_retest] = "Uptrend_Retest"
        trend_retest[downtrend_retest] = "Downtrend_Retest"
        
        # Trend strength
        band_width = upper - lower
        distance_from_middle = (close - middle) / band_width.replace(0, 1e-9)
        trend_strength = distance_from_middle.abs()
        
        kc_df['kc_trend'] = trend
        kc_df['kc_trend_retest'] = trend_retest
        kc_df['kc_trend_strength'] = trend_strength
        
        return kc_df

    def prepare_lstm_data(self, data, look_back=30):
        """Prepare data for LSTM training"""
        if data is None or data.empty:
            return np.array([]), np.array([])
            
        feature_columns = []
        
        # Base features
        base_features = ['open', 'high', 'low', 'close', 'volume']
        available_base = [col for col in base_features if col in data.columns]
        feature_columns.extend(available_base)
        
        # RSI features
        rsi_features = [col for col in data.columns if col.startswith('result_rsi') and col.endswith('_numeric')]
        if 'result_rsi' in data.columns:
            feature_columns.append('result_rsi')
        feature_columns.extend(rsi_features)
        
        # Keltner Channel features
        kc_features = [col for col in data.columns if col.startswith('kc_') and 
                       col not in ['kc_trend', 'kc_trend_retest']]
        feature_columns.extend(kc_features)
        
        if 'kc_trend_numeric' in data.columns:
            feature_columns.append('kc_trend_numeric')
        if 'kc_trend_retest_numeric' in data.columns:
            feature_columns.append('kc_trend_retest_numeric')
            
        if 'lstm_pred_pct_change' in data.columns:
            feature_columns.append('lstm_pred_pct_change')
            
        # Filter available columns
        available_columns = list(data.columns)
        feature_columns = [col for col in feature_columns if col in available_columns]
        
        if 'target_pct_change' not in data.columns:
            raise ValueError("Missing 'target_pct_change' column in data for LSTM.")
            
        if not feature_columns:
            raise ValueError("No feature columns available for LSTM after filtering.")
            
        features_df = data[feature_columns]
        features = features_df.values
        target = data['target_pct_change'].values
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - look_back): 
            X.append(features[i:(i + look_back)])
            y.append(target[i + look_back - 1])
            
        if not X: 
            return np.array([]), np.array([])
            
        X_array = np.array(X)
        y_array = np.array(y).reshape(-1, 1)
        
        return X_array, y_array

    def get_all_indicators(self):
        """Get all calculated indicators"""
        return self.indicators


class LSTMModel(nn.Module):
    """LSTM model for Bitcoin price prediction"""
    
    def __init__(self, input_size, hidden_size=100, num_layers=2, output_size=1, dropout_rate=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

    def train_pytorch_lstm(self, X_train_np, y_train_np, device, epochs=100, batch_size=32, 
                          learning_rate=0.0005, validation_split=0.2, 
                          model_save_path='./model/lstm_model.pth', patience=30):
        """Train LSTM model with improved validation"""
        
        # Input validation
        if X_train_np.size == 0 or y_train_np.size == 0:
            print("Warning: Empty training data provided")
            return self
            
        if len(X_train_np) != len(y_train_np):
            raise ValueError(f"Mismatch in training data: X has {len(X_train_np)} samples, y has {len(y_train_np)}")
            
        if len(X_train_np) < 10:
            print(f"Warning: Very small training set ({len(X_train_np)} samples)")
            
        # Check for NaN values
        if np.isnan(X_train_np).any() or np.isnan(y_train_np).any():
            print("Warning: NaN values detected in training data")
            # Remove NaN samples
            mask = ~(np.isnan(X_train_np).any(axis=(1,2)) | np.isnan(y_train_np).any(axis=1))
            X_train_np = X_train_np[mask]
            y_train_np = y_train_np[mask]
            
        self.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
        
        # Convert to tensors
        try:
            X_tensor = torch.from_numpy(X_train_np).float()
            y_tensor = torch.from_numpy(y_train_np).float()
        except Exception as e:
            print(f"Error converting data to tensors: {e}")
            return self
        
        dataset_size = len(X_tensor)
        if dataset_size == 0:
            print("No valid data for training")
            return self
            
        # Validation split with minimum size check
        val_size = max(1, int(validation_split * dataset_size))
        train_size = dataset_size - val_size
        
        if train_size <= 0:
            print("Warning: No data left for training after validation split")
            train_dataset = TensorDataset(X_tensor, y_tensor)
            val_loader = None
            train_size = dataset_size
        else:
            train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
            val_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])
            val_loader = DataLoader(val_dataset, batch_size=min(batch_size, val_size), shuffle=False)
            
        train_loader = DataLoader(train_dataset, batch_size=min(batch_size, train_size), shuffle=True)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        print(f"Training LSTM model: {train_size} train samples, {val_size if val_loader else 0} val samples")
        
        for epoch in range(epochs):
            self.train()
            train_loss_sum = 0
            train_batches = 0
            
            try:
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    optimizer.zero_grad()
                    outputs = self(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"NaN loss detected at epoch {epoch}")
                        continue
                        
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss_sum += loss.item()
                    train_batches += 1
                    
            except Exception as e:
                print(f"Training error at epoch {epoch}: {e}")
                continue
                
            avg_train_loss = train_loss_sum / max(train_batches, 1)
            
            # Validation
            if val_loader:
                self.eval()
                val_loss_sum = 0
                val_batches = 0
                
                with torch.no_grad():   
                    try:
                        for X_batch, y_batch in val_loader:
                            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                            outputs = self(X_batch)
                            loss = criterion(outputs, y_batch)
                            
                            if not torch.isnan(loss):
                                val_loss_sum += loss.item()
                                val_batches += 1
                    except Exception as e:
                        print(f"Validation error at epoch {epoch}: {e}")
                        
                avg_val_loss = val_loss_sum / max(val_batches, 1) if val_batches > 0 else float('inf')
                scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    
                    # Save model with error handling
                    try:
                        if not os.path.exists(os.path.dirname(model_save_path)):
                            os.makedirs(os.path.dirname(model_save_path))
                        torch.save(self.state_dict(), model_save_path)
                    except Exception as e:
                        print(f"Error saving model: {e}")
                        
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                        
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else: 
                if (epoch + 1) % 10 == 0:
                    try:
                        if not os.path.exists(os.path.dirname(model_save_path)):
                            os.makedirs(os.path.dirname(model_save_path))
                        torch.save(self.state_dict(), model_save_path)
                    except Exception as e:
                        print(f"Error saving model: {e}")
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")
                    
        # Load best model with error handling
        try:
            if os.path.exists(model_save_path):
                self.load_state_dict(torch.load(model_save_path, map_location=device))
                print(f"Best model loaded from {model_save_path}")
        except Exception as e:
            print(f"Error loading best model: {e}")
        
        return self

    def evaluate_pytorch_lstm(self, X_test_np, y_test_np, device, target_scaler, batch_size=32):
        """Evaluate LSTM model"""
        if len(X_test_np) == 0 or len(y_test_np) == 0:
            return {'rmse_pct_change': float('nan'), 'mae_pct_change': float('nan')}, np.array([])
            
        self.to(device)
        self.eval()
        
        X_test_tensor = torch.from_numpy(X_test_np).float()
        dataset = TensorDataset(X_test_tensor, torch.zeros(len(X_test_tensor)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds_scaled_list = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(device)
                outputs = self(X_batch)
                all_preds_scaled_list.append(outputs.cpu().numpy())
                
        if not all_preds_scaled_list:
            return {'rmse_pct_change': float('nan'), 'mae_pct_change': float('nan')}, np.array([])
            
        y_pred_scaled_pct_change = np.concatenate(all_preds_scaled_list)
        y_pred_unscaled_pct_change = target_scaler.inverse_transform(y_pred_scaled_pct_change)
        y_test_unscaled_pct_change = target_scaler.inverse_transform(y_test_np)
        
        rmse = np.sqrt(mean_squared_error(y_test_unscaled_pct_change, y_pred_unscaled_pct_change))
        mae = mean_absolute_error(y_test_unscaled_pct_change, y_pred_unscaled_pct_change)
        
        return {'rmse_pct_change': rmse, 'mae_pct_change': mae}, y_pred_scaled_pct_change

    def predict_pytorch_lstm(self, X_np, device, batch_size=32):
        """Make predictions with LSTM model"""
        if len(X_np) == 0:
            return np.array([])
            
        self.to(device)
        self.eval()
        
        X_tensor = torch.from_numpy(X_np).float()
        dataset = TensorDataset(X_tensor, torch.zeros(len(X_tensor)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds_list = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(device)
                outputs = self(X_batch)
                all_preds_list.append(outputs.cpu().numpy())
                
        if not all_preds_list:
            return np.array([])
            
        return np.concatenate(all_preds_list)


class BlackLitterman:
    """Optimized Black-Litterman portfolio optimization for Bitcoin spot trading"""
    
    def __init__(self, returns_series, lstm_predictions=None, risk_aversion=2.5, 
                 tau=0.05, confidence_level=0.75, view_impact=0.4):
        
        self.returns_series = returns_series
        self.lstm_predictions = lstm_predictions
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.confidence_level = confidence_level
        self.view_impact = view_impact
        
        self.market_weights = None
        self.cov_matrix = None
        self.prior_returns = None
        self.posterior_returns = None
        self.posterior_cov = None
        
    def calculate_market_capitalization_weights(self):
        """Calculate market cap weights (single asset = 100%)"""
        self.market_weights = np.array([1.0])
        return self.market_weights
    
    def calculate_covariance_matrix(self, window=90):
        """Calculate covariance matrix with improved validation"""
        if self.returns_series is None:
            raise ValueError("Returns series is required to calculate covariance matrix")
            
        if len(self.returns_series) < 30:
            raise ValueError("At least 30 data points required for reliable covariance estimation")
        
        returns_data = self.returns_series.tail(min(window, len(self.returns_series))).dropna()
        
        if len(returns_data) == 0:
            raise ValueError("No valid returns data after cleaning")
        
        # Use exponentially weighted variance with validation
        alpha = 0.94
        try:
            variance = returns_data.ewm(alpha=alpha, adjust=False).var().iloc[-1]
        except Exception:
            # Fallback to simple variance
            variance = returns_data.var()
        
        # Ensure minimum variance for numerical stability
        min_variance = 0.0001
        variance = max(variance, min_variance)
        
        if np.isnan(variance) or variance <= 0:
            variance = min_variance
            
        self.cov_matrix = np.array([[variance]])
        return self.cov_matrix
    
    def calculate_implied_returns(self):
        """Calculate implied equilibrium returns"""
        if self.market_weights is None:
            self.calculate_market_capitalization_weights()
        
        if self.cov_matrix is None:
            self.calculate_covariance_matrix()
        
        self.prior_returns = self.risk_aversion * np.dot(self.cov_matrix, self.market_weights)
        
        # Clip to reasonable range
        max_daily_return = 0.005
        min_daily_return = 0.0001
        self.prior_returns = np.array([np.clip(self.prior_returns[0], min_daily_return, max_daily_return)])
        
        return self.prior_returns
    
    def prepare_views_and_uncertainty(self):
        """Optimized LSTM views preparation for Black-Litterman (Bitcoin spot focus)"""
        if self.lstm_predictions is None or len(self.lstm_predictions) < 10:
            return None, None, None
        
        # Picking matrix for single asset (Bitcoin)
        P = np.array([[1.0]])
        
        # OPTIMASI 1: Exponentially weighted LSTM views (more recent = more important)
        prediction_window = min(20, len(self.lstm_predictions))
        recent_predictions = self.lstm_predictions.tail(prediction_window)
        
        # Create exponential weights (recent predictions get higher weight)
        weights = np.exp(np.linspace(-1, 0, prediction_window))
        weights = weights / weights.sum()
        
        # Weighted average of predictions
        weighted_prediction = np.average(recent_predictions.values, weights=weights)
        
        # OPTIMASI 2: Dynamic clipping based on recent Bitcoin volatility
        recent_vol = self.returns_series.tail(30).std() if len(self.returns_series) >= 30 else 0.03
        
        # Bitcoin-specific bounds (more aggressive than traditional assets)
        max_daily_return = min(0.012, recent_vol * 4)  # Bitcoin can move 1.2% daily
        min_daily_return = max(-0.010, -recent_vol * 4)  # Limit downside to 1%
        
        Q = np.array([np.clip(weighted_prediction, min_daily_return, max_daily_return)])
        
        # OPTIMASI 3: Adaptive uncertainty based on LSTM prediction confidence
        prediction_std = recent_predictions.std()
        prediction_range = recent_predictions.quantile(0.75) - recent_predictions.quantile(0.25)
        
        # Base uncertainty with confidence adjustment
        base_uncertainty = (prediction_std**2 + (prediction_range**2)/4) / self.confidence_level
        
        # OPTIMASI 4: Bitcoin volatility adjustment
        volatility_factor = max(0.5, min(2.0, recent_vol / 0.03))  # Normalize around 3% daily vol
        adjusted_uncertainty = base_uncertainty * volatility_factor
        
        # Ensure minimum uncertainty for numerical stability
        min_uncertainty = 0.3 * np.var(self.returns_series.tail(60)) if len(self.returns_series) >= 60 else 0.0001
        adjusted_uncertainty = max(adjusted_uncertainty, min_uncertainty)
        
        # OPTIMASI 5: View deviation penalty (if LSTM disagrees strongly with market)
        if self.prior_returns is not None:
            view_deviation = abs(Q[0] - self.prior_returns[0])
            if view_deviation > 0.005:  # 0.5% daily deviation threshold
                uncertainty_penalty = 1 + np.tanh(view_deviation * 20) * 1.5
                adjusted_uncertainty *= uncertainty_penalty
        
        omega = np.array([[adjusted_uncertainty]])
        
        return P, Q, omega
    
    def calculate_posterior_returns(self):
        """Optimized posterior return calculation for Bitcoin"""
        if self.prior_returns is None:
            self.calculate_implied_returns()
        
        P, Q, omega = self.prepare_views_and_uncertainty()
        
        if P is None or Q is None or omega is None:
            self.posterior_returns = self.prior_returns.copy()
            self.posterior_cov = self.cov_matrix.copy()
            return self.posterior_returns, self.posterior_cov
        
        # OPTIMASI 1: Numerical stability improvements
        tau_cov = self.tau * self.cov_matrix
        tau_cov_reg = tau_cov + np.eye(1) * 1e-10  # Small regularization
        
        try:
            tau_cov_inv = np.linalg.inv(tau_cov_reg)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            tau_cov_inv = np.linalg.pinv(tau_cov_reg)
        
        omega_reg = omega + np.eye(1) * 1e-10
        try:
            omega_inv = np.linalg.inv(omega_reg)
        except np.linalg.LinAlgError:
            omega_inv = np.linalg.pinv(omega_reg)
        
        # OPTIMASI 2: Stable Black-Litterman calculation
        try:
            posterior_precision = tau_cov_inv + np.dot(P.T, np.dot(omega_inv, P))
            posterior_cov_matrix = np.linalg.inv(posterior_precision)
            
            prior_weighted = np.dot(tau_cov_inv, self.prior_returns)
            views_weighted = np.dot(P.T, np.dot(omega_inv, Q))
            
            bl_posterior = np.dot(posterior_cov_matrix, prior_weighted + views_weighted)
            
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to simple blending if numerical issues arise
            bl_posterior = 0.7 * self.prior_returns + 0.3 * Q.flatten()
            posterior_cov_matrix = self.cov_matrix * 0.9
        
        # OPTIMASI 3: Adaptive view impact based on Bitcoin volatility
        recent_vol = np.std(self.returns_series.tail(30)) if len(self.returns_series) >= 30 else 0.03
        
        if recent_vol > 0.05:  # High volatility - trust views less
            adaptive_view_impact = self.view_impact * 0.7
        elif recent_vol < 0.02:  # Low volatility - trust views more
            adaptive_view_impact = min(self.view_impact * 1.3, 0.6)
        else:
            adaptive_view_impact = self.view_impact
        
        # Blend with adaptive impact
        self.posterior_returns = ((1 - adaptive_view_impact) * self.prior_returns + 
                                 adaptive_view_impact * bl_posterior)
        
        self.posterior_cov = posterior_cov_matrix
        
        # OPTIMASI 4: Ensure reasonable volatility bounds for Bitcoin
        min_vol_ratio = 0.6  # Don't reduce volatility too much
        max_vol_ratio = 1.4   # Don't increase volatility too much
        
        prior_vol = np.sqrt(self.cov_matrix[0, 0])
        posterior_vol = np.sqrt(self.posterior_cov[0, 0])
        
        if posterior_vol < min_vol_ratio * prior_vol:
            scaling_factor = (min_vol_ratio * prior_vol / posterior_vol)**2
            self.posterior_cov = self.posterior_cov * scaling_factor
        elif posterior_vol > max_vol_ratio * prior_vol:
            scaling_factor = (max_vol_ratio * prior_vol / posterior_vol)**2
            self.posterior_cov = self.posterior_cov * scaling_factor
        
        return self.posterior_returns, self.posterior_cov
    
    def calculate_optimal_weights(self):
        """Optimized weight calculation for Bitcoin spot trading"""
        if self.posterior_returns is None:
            self.calculate_posterior_returns()
        
        # OPTIMASI 1: Dynamic allocation bounds based on market conditions
        recent_vol = np.std(self.returns_series.tail(30)) if len(self.returns_series) >= 30 else 0.03
        
        if recent_vol > 0.06:  # High volatility - more conservative
            min_allocation = 0.0
            max_allocation = 0.8  # Max 80% in high vol
        elif recent_vol < 0.02:  # Low volatility - more aggressive
            min_allocation = 0.2  # Min 20% in low vol
            max_allocation = 1.0
        else:  # Normal volatility
            min_allocation = 0.0
            max_allocation = 1.0
        
        # OPTIMASI 2: Risk-adjusted allocation
        if self.posterior_cov is not None and np.abs(self.posterior_cov[0, 0]) >= 1e-8:
            # Mean-variance optimization with regularization
            posterior_cov_reg = self.posterior_cov + np.eye(1) * 1e-8
            try:
                inv_cov = np.linalg.inv(posterior_cov_reg)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(posterior_cov_reg)
            
            # OPTIMASI 3: Adaptive risk aversion based on Bitcoin characteristics
            # Bitcoin is more volatile, so adjust risk aversion dynamically
            btc_risk_aversion = self.risk_aversion * (1 + recent_vol * 5)
            
            optimal_weight = np.dot(inv_cov, self.posterior_returns) / btc_risk_aversion
            
            # OPTIMASI 4: Smooth clipping to avoid extreme allocations
            raw_weight = optimal_weight[0]
            if raw_weight < min_allocation:
                smooth_weight = min_allocation + 0.1 * (raw_weight - min_allocation)
            elif raw_weight > max_allocation:
                smooth_weight = max_allocation - 0.1 * (raw_weight - max_allocation)
            else:
                smooth_weight = raw_weight
                
            final_weight = np.clip(smooth_weight, min_allocation, max_allocation)
            optimal_weight = np.array([final_weight])
            
        else:
            # OPTIMASI 5: Fallback allocation based on return expectations
            if self.posterior_returns[0] > 0.003:  # Strong positive expectation
                optimal_weight = np.array([max_allocation * 0.8])
            elif self.posterior_returns[0] > 0.001:  # Moderate positive expectation
                optimal_weight = np.array([max_allocation * 0.5])
            elif self.posterior_returns[0] < -0.002:  # Negative expectation
                optimal_weight = np.array([min_allocation])
            else:  # Neutral expectation
                optimal_weight = np.array([(min_allocation + max_allocation) / 2])
        
        return optimal_weight
    
    def get_bl_results(self):
        """Get comprehensive Black-Litterman results"""
        market_weights = self.calculate_market_capitalization_weights()
        cov_matrix = self.calculate_covariance_matrix()
        prior_returns = self.calculate_implied_returns()
        posterior_returns, posterior_cov = self.calculate_posterior_returns()
        optimal_weights = self.calculate_optimal_weights()
        
        P, Q, omega = self.prepare_views_and_uncertainty()
        views_summary = {
            'has_views': P is not None and Q is not None,
            'lstm_view': Q[0] if Q is not None else None,
            'view_uncertainty': omega[0, 0] if omega is not None else None,
            'confidence_level': self.confidence_level,
            'view_impact': self.view_impact
        }
        
        scaling_factor = self._determine_scaling_factor()
        
        prior_volatility = np.sqrt(cov_matrix[0, 0] * scaling_factor)
        posterior_volatility = np.sqrt(posterior_cov[0, 0] * scaling_factor) if posterior_cov is not None else prior_volatility
        
        annualized_prior_returns = prior_returns * scaling_factor
        annualized_posterior_returns = posterior_returns * scaling_factor if posterior_returns is not None else annualized_prior_returns
        
        # Risk-free rate
        risk_free_rate = 0.02
        daily_rf = risk_free_rate / scaling_factor
        
        # Sharpe ratios
        prior_sharpe = (prior_returns[0] - daily_rf) / np.sqrt(cov_matrix[0, 0]) if cov_matrix[0, 0] > 0 else 0
        posterior_sharpe = (posterior_returns[0] - daily_rf) / np.sqrt(posterior_cov[0, 0]) if posterior_cov is not None and posterior_cov[0, 0] > 0 else 0
        
        prior_sharpe_annualized = prior_sharpe * np.sqrt(scaling_factor)
        posterior_sharpe_annualized = posterior_sharpe * np.sqrt(scaling_factor)
        
        # Model quality metrics
        prior_mean = prior_returns[0]
        posterior_mean = posterior_returns[0]
        view_relative_impact = abs((posterior_mean - prior_mean) / prior_mean) if abs(prior_mean) > 1e-10 else 0
        
        model_quality = {
            'prior_sharpe': prior_sharpe_annualized,
            'posterior_sharpe': posterior_sharpe_annualized,
            'view_relative_impact': view_relative_impact,
            'volatility_reduction_pct': (prior_volatility - posterior_volatility) / prior_volatility * 100 if prior_volatility > 0 else 0,
        }
        
        return {
            'market_weights': market_weights,
            'prior_returns': prior_returns,
            'posterior_returns': posterior_returns,
            'posterior_covariance': posterior_cov,
            'optimal_weights': optimal_weights,
            'views_summary': views_summary,
            'prior_volatility_annualized': prior_volatility,
            'posterior_volatility_annualized': posterior_volatility,
            'annualized_prior_returns': annualized_prior_returns,
            'annualized_posterior_returns': annualized_posterior_returns,
            'return_adjustment': posterior_returns[0] - prior_returns[0] if posterior_returns is not None else 0,
            'risk_adjustment': posterior_volatility - prior_volatility,
            'scaling_factor': scaling_factor,
            'model_quality': model_quality,
            'risk_aversion': self.risk_aversion
        }
    
    def _determine_scaling_factor(self):
        """Determine annualization factor"""
        if hasattr(self.returns_series, 'index'):
            if hasattr(self.returns_series.index, 'freq'):
                freq = self.returns_series.index.freq
                if freq == 'D' or freq is None:
                    return 365
                elif freq == 'B':
                    return 365
                elif freq == 'W':
                    return 52
                elif freq == 'M':
                    return 12
                else:
                    return 365
            else:
                if len(self.returns_series) > 1:
                    try:
                        avg_days = (self.returns_series.index[-1] - self.returns_series.index[0]).days / (len(self.returns_series) - 1)
                        if avg_days < 2:
                            return 365
                        elif avg_days < 8:
                            return 52
                        elif avg_days < 32:
                            return 12
                        else:
                            return 365
                    except:
                        return 365
                else:
                    return 365
        else:
            return 365
    
    def get_adjusted_returns_series(self):
        """Get returns series adjusted by Black-Litterman"""
        if self.posterior_returns is None:
            self.calculate_posterior_returns()
        
        adjustment_factor = self.posterior_returns[0] - self.prior_returns[0] if self.posterior_returns is not None else 0
        
        adjusted_returns = self.returns_series.copy()
        if len(adjusted_returns) >= 60:
            recent_returns = adjusted_returns.tail(60)
            
            # Apply gradual adjustment
            weights = np.linspace(0.1, 1.0, 60)
            adjustment = adjustment_factor * weights * 0.3
            adjusted_recent = recent_returns.values + adjustment
            
            adjusted_returns.iloc[-60:] = adjusted_recent
        
        return adjusted_returns


class CVaR:
    """Optimized Conditional Value at Risk with dynamic weighting for Bitcoin spot"""
    
    def __init__(self, returns_series, confidence_level=0.90, indicator_data=None, 
                 rsi_weight=0.3, kc_weight=0.3, use_enhanced_features=True,
                 bl_results=None):
        self.returns_series = returns_series
        self.confidence_level = confidence_level
        self.indicator_data = indicator_data
        self.rsi_weight = rsi_weight
        self.kc_weight = kc_weight
        self.base_weight = 1.0 - rsi_weight - kc_weight
        self.use_enhanced_features = use_enhanced_features
        self.bl_results = bl_results

    def calculate_var(self):
        """Calculate Value at Risk"""
        if self.returns_series is None or self.returns_series.empty:
            return np.nan
            
        var = self.returns_series.quantile(1 - self.confidence_level)
        return var

    def calculate_cvar(self):
        """Optimized CVaR calculation with dynamic component weighting for Bitcoin spot"""
        if self.returns_series is None or self.returns_series.empty:
            return np.nan
            
        var = self.calculate_var()
        if np.isnan(var):
            return np.nan
            
        traditional_cvar = self.returns_series[self.returns_series <= var].mean()
        
        if self.indicator_data is None and self.bl_results is None:
            return traditional_cvar
        
        # OPTIMASI 1: Dynamic weight allocation based on market regime
        market_regime = self._determine_market_regime()
        weights = self._get_dynamic_weights(market_regime)
        
        # Get component adjustments
        rsi_adjustment = self.calculate_rsi_risk_adjustment()
        kc_adjustment = self.calculate_kc_risk_adjustment()
        bl_adjustment = self.calculate_bl_risk_adjustment()
        
        # OPTIMASI 2: Component reliability scoring
        rsi_reliability = self._calculate_rsi_reliability()
        kc_reliability = self._calculate_kc_reliability()
        bl_reliability = self._calculate_bl_reliability()
        
        # Adjust weights by reliability
        total_reliability = rsi_reliability + kc_reliability + bl_reliability
        if total_reliability > 0:
            reliability_adjusted_weights = {
                'rsi': weights['rsi'] * (rsi_reliability / total_reliability),
                'kc': weights['kc'] * (kc_reliability / total_reliability), 
                'bl': weights['bl'] * (bl_reliability / total_reliability),
                'base': weights['base']
            }
        else:
            reliability_adjusted_weights = weights
        
        # OPTIMASI 3: Weighted adjustment calculation
        weighted_adjustment = (
            reliability_adjusted_weights['base'] * 1.0 + 
            reliability_adjusted_weights['rsi'] * rsi_adjustment + 
            reliability_adjusted_weights['kc'] * kc_adjustment +
            reliability_adjusted_weights['bl'] * bl_adjustment
        )
        
        # OPTIMASI 4: Smooth adjustment to avoid extreme values
        max_adjustment = 2.5  # Maximum 250% of traditional CVaR
        min_adjustment = 0.4  # Minimum 40% of traditional CVaR
        
        smooth_adjustment = np.tanh(weighted_adjustment - 1) * 0.5 + 1
        final_adjustment = np.clip(smooth_adjustment, min_adjustment, max_adjustment)
        
        adjusted_cvar = traditional_cvar * final_adjustment
        
        return adjusted_cvar

    def _determine_market_regime(self):
        """Determine current market regime for dynamic weighting"""
        if len(self.returns_series) < 30:
            return 'normal'
        
        recent_returns = self.returns_series.tail(30)
        recent_vol = recent_returns.std()
        recent_trend = recent_returns.mean()
        
        # OPTIMASI: Bitcoin-specific regime classification
        if recent_vol > 0.06:  # High volatility (6%+ daily)
            if recent_trend > 0.005:
                return 'bull_volatile'  # Bull market with high volatility
            elif recent_trend < -0.005:
                return 'bear_volatile'  # Bear market with high volatility
            else:
                return 'choppy'  # High volatility, no clear trend
        elif recent_vol < 0.02:  # Low volatility (2%- daily)
            if recent_trend > 0.002:
                return 'bull_stable'  # Stable bull market
            elif recent_trend < -0.002:
                return 'bear_stable'  # Stable bear market
            else:
                return 'sideways'  # Low volatility sideways market
        else:
            return 'normal'  # Normal market conditions

    def _get_dynamic_weights(self, market_regime):
        """Get dynamic component weights based on market regime"""
        
        # OPTIMASI: Regime-specific weight allocation for Bitcoin
        weight_schemes = {
            'bull_volatile': {
                'base': 0.15, 'rsi': 0.25, 'kc': 0.35, 'bl': 0.25
                # In bull volatile: trust Keltner Channels more (trend following)
            },
            'bear_volatile': {
                'base': 0.15, 'rsi': 0.40, 'kc': 0.25, 'bl': 0.20
                # In bear volatile: trust RSI more (oversold signals)
            },
            'bull_stable': {
                'base': 0.20, 'rsi': 0.20, 'kc': 0.25, 'bl': 0.35
                # In bull stable: trust Black-Litterman more (LSTM predictions)
            },
            'bear_stable': {
                'base': 0.25, 'rsi': 0.30, 'kc': 0.20, 'bl': 0.25
                # In bear stable: balanced approach with RSI emphasis
            },
            'choppy': {
                'base': 0.30, 'rsi': 0.30, 'kc': 0.20, 'bl': 0.20
                # In choppy: trust base model more (conservative)
            },
            'sideways': {
                'base': 0.20, 'rsi': 0.35, 'kc': 0.15, 'bl': 0.30
                # In sideways: RSI for range-bound trading
            },
            'normal': {
                'base': 0.20, 'rsi': 0.25, 'kc': 0.25, 'bl': 0.30
                # Normal: balanced allocation
            }
        }
        
        return weight_schemes.get(market_regime, weight_schemes['normal'])

    def calculate_rsi_risk_adjustment(self):
        """Calculate RSI-based risk adjustment"""
        if self.indicator_data is None or 'rsi' not in self.indicator_data:
            return 1.0
            
        rsi_df = self.indicator_data['rsi']
        if 'result_rsi' not in rsi_df.columns:
            return 1.0
            
        recent_rsi = rsi_df['result_rsi'].iloc[-1]
        
        # Base RSI adjustment
        if recent_rsi > 70:
            base_adjustment = 1.2 + (recent_rsi - 70) / 30 * 0.3
        elif recent_rsi < 30:
            base_adjustment = 1.2 + (30 - recent_rsi) / 30 * 0.3
        else:
            base_adjustment = 1.0 - 0.3 * (1 - abs(recent_rsi - 50) / 20)
            
        # Enhanced features
        if self.use_enhanced_features:
            rsi_factor = 1.0
            
            if 'result_rsi_bullish_div' in rsi_df.columns and rsi_df['result_rsi_bullish_div'].iloc[-1]:
                rsi_factor *= 0.85
                
            if 'result_rsi_bearish_div' in rsi_df.columns and rsi_df['result_rsi_bearish_div'].iloc[-1]:
                rsi_factor *= 1.25
                
            if 'result_rsi_double_bottom' in rsi_df.columns and rsi_df['result_rsi_double_bottom'].iloc[-1]:
                rsi_factor *= 0.8
                
            if 'result_rsi_pivot_high' in rsi_df.columns and rsi_df['result_rsi_pivot_high'].iloc[-1]:
                rsi_factor *= 1.15
                
            if 'result_rsi_pivot_low' in rsi_df.columns and rsi_df['result_rsi_pivot_low'].iloc[-1]:
                rsi_factor *= 0.85
                
            return base_adjustment * rsi_factor
            
        return base_adjustment

    def calculate_kc_risk_adjustment(self):
        """Calculate Keltner Channel risk adjustment"""
        if self.indicator_data is None or 'keltner_channels' not in self.indicator_data:
            return 1.0
            
        kc_df = self.indicator_data['keltner_channels']
        
        required_cols = ['kc_trend', 'kc_trend_strength']
        if not all(col in kc_df.columns for col in required_cols):
            return 1.0
            
        recent_trend = kc_df['kc_trend'].iloc[-1]
        recent_strength = kc_df['kc_trend_strength'].iloc[-1]
        
        has_retest = 'kc_trend_retest' in kc_df.columns
        if has_retest:
            recent_retest = kc_df['kc_trend_retest'].iloc[-1]
        else:
            recent_retest = "No_Retest"
            
        # Base trend adjustment
        if recent_trend == "Uptrend":
            base_adj = 0.9 - (recent_strength * 0.2)
            base_adj = max(0.6, base_adj)
        elif recent_trend == "Downtrend":
            base_adj = 1.1 + (recent_strength * 0.3)
            base_adj = min(1.8, base_adj)
        else:
            base_adj = 1.0
            
        # Retest adjustment
        retest_adj = 1.0
        if recent_retest == "Uptrend_Retest":
            retest_adj = 1.15
        elif recent_retest == "Downtrend_Retest":
            retest_adj = 0.95
            
        return base_adj * retest_adj

    def calculate_bl_risk_adjustment(self):
        """Calculate Black-Litterman risk adjustment"""
        if self.bl_results is None:
            return 1.0
        
        return_adjustment = self.bl_results.get('return_adjustment', 0)
        risk_adjustment = self.bl_results.get('risk_adjustment', 0)
        views_summary = self.bl_results.get('views_summary', {})
        
        # Return-based adjustment
        if return_adjustment > 0.01:
            base_adj = 0.8 - min(return_adjustment * 5, 0.3)
        elif return_adjustment < -0.01:
            base_adj = 1.3 + min(abs(return_adjustment) * 5, 0.5)
        else:
            base_adj = 1.0
        
        # Confidence adjustment
        if views_summary.get('has_views', False):
            confidence = views_summary.get('confidence_level', 0.5)
            confidence_adj = 0.8 + (confidence * 0.4)
            base_adj = 1.0 + (base_adj - 1.0) * confidence_adj
        
        # Risk adjustment
        if risk_adjustment > 0:
            risk_adj = 1.1 + min(risk_adjustment * 2, 0.3)
        elif risk_adjustment < 0:
            risk_adj = 0.9 - min(abs(risk_adjustment) * 2, 0.2)
        else:
            risk_adj = 1.0
        
        final_adjustment = base_adj * risk_adj
        
        return max(0.5, min(final_adjustment, 2.0))

    def _calculate_rsi_reliability(self):
        """Calculate RSI component reliability score"""
        if self.indicator_data is None or 'rsi' not in self.indicator_data:
            return 0.0
        
        rsi_df = self.indicator_data['rsi']
        if 'result_rsi' not in rsi_df.columns or len(rsi_df) < 10:
            return 0.0
        
        recent_rsi = rsi_df['result_rsi'].tail(10)
        
        # OPTIMASI: RSI reliability based on signal strength and consistency
        reliability_score = 0.5  # Base reliability
        
        # Check for strong signals
        current_rsi = recent_rsi.iloc[-1]
        if current_rsi < 25 or current_rsi > 75:  # Very strong oversold/overbought
            reliability_score += 0.3
        elif current_rsi < 35 or current_rsi > 65:  # Moderate oversold/overbought
            reliability_score += 0.15
        
        # Check for divergence patterns (high reliability indicators)
        if 'result_rsi_bullish_div' in rsi_df.columns and rsi_df['result_rsi_bullish_div'].iloc[-1]:
            reliability_score += 0.25
        if 'result_rsi_bearish_div' in rsi_df.columns and rsi_df['result_rsi_bearish_div'].iloc[-1]:
            reliability_score += 0.25
        if 'result_rsi_double_bottom' in rsi_df.columns and rsi_df['result_rsi_double_bottom'].iloc[-1]:
            reliability_score += 0.2
        
        # Penalize erratic RSI behavior
        rsi_volatility = recent_rsi.std()
        if rsi_volatility > 15:  # Very volatile RSI
            reliability_score *= 0.8
        
        return min(1.0, reliability_score)

    def _calculate_kc_reliability(self):
        """Calculate Keltner Channel component reliability score"""
        if self.indicator_data is None or 'keltner_channels' not in self.indicator_data:
            return 0.0
        
        kc_df = self.indicator_data['keltner_channels']
        required_cols = ['kc_trend', 'kc_trend_strength']
        if not all(col in kc_df.columns for col in required_cols) or len(kc_df) < 5:
            return 0.0
        
        # OPTIMASI: KC reliability based on trend strength and consistency
        reliability_score = 0.5  # Base reliability
        
        current_trend = kc_df['kc_trend'].iloc[-1]
        current_strength = kc_df['kc_trend_strength'].iloc[-1]
        
        # Strong trend = higher reliability
        if current_strength > 0.8:
            reliability_score += 0.3
        elif current_strength > 0.5:
            reliability_score += 0.15
        
        # Trend consistency over recent periods
        recent_trends = kc_df['kc_trend'].tail(5)
        trend_consistency = (recent_trends == current_trend).sum() / len(recent_trends)
        reliability_score += trend_consistency * 0.2
        
        # Retest patterns add reliability
        if 'kc_trend_retest' in kc_df.columns:
            recent_retest = kc_df['kc_trend_retest'].iloc[-1]
            if recent_retest in ['Uptrend_Retest', 'Downtrend_Retest']:
                reliability_score += 0.15
        
        return min(1.0, reliability_score)

    def _calculate_bl_reliability(self):
        """Calculate Black-Litterman component reliability score"""
        if self.bl_results is None:
            return 0.0
        
        # OPTIMASI: BL reliability based on view strength and model quality
        reliability_score = 0.5  # Base reliability
        
        views_summary = self.bl_results.get('views_summary', {})
        
        # Views availability and confidence
        if views_summary.get('has_views', False):
            confidence = views_summary.get('confidence_level', 0.5)
            reliability_score += confidence * 0.3
            
            # View impact strength
            view_impact = views_summary.get('view_impact', 0)
            reliability_score += view_impact * 0.2
        else:
            reliability_score *= 0.6  # Penalize lack of views
        
        # Model quality metrics
        model_quality = self.bl_results.get('model_quality', {})
        posterior_sharpe = model_quality.get('posterior_sharpe', 0)
        
        if posterior_sharpe > 1.0:  # Good risk-adjusted returns
            reliability_score += 0.2
        elif posterior_sharpe > 0.5:
            reliability_score += 0.1
        elif posterior_sharpe < 0:
            reliability_score *= 0.8  # Penalize negative Sharpe
        
        # Return adjustment magnitude (too extreme = less reliable)
        return_adj = abs(self.bl_results.get('return_adjustment', 0) * 365 * 100)
        if return_adj > 50:  # Very large annual adjustment (>50%)
            reliability_score *= 0.7
        elif return_adj > 25:  # Large adjustment (>25%)
            reliability_score *= 0.85
        
        return min(1.0, reliability_score)

    def get_risk_components(self):
        """Get detailed risk component analysis"""
        var = self.calculate_var()
        if np.isnan(var):
            return {"error": "Insufficient data to calculate risk components"}
            
        traditional_cvar = self.returns_series[self.returns_series <= var].mean()
        
        # Get dynamic weights and adjustments
        market_regime = self._determine_market_regime()
        weights = self._get_dynamic_weights(market_regime)
        
        rsi_adjustment = self.calculate_rsi_risk_adjustment()
        kc_adjustment = self.calculate_kc_risk_adjustment()
        bl_adjustment = self.calculate_bl_risk_adjustment()
        
        # Calculate reliability scores
        rsi_reliability = self._calculate_rsi_reliability()
        kc_reliability = self._calculate_kc_reliability()
        bl_reliability = self._calculate_bl_reliability()
        
        # Adjust weights by reliability
        total_reliability = rsi_reliability + kc_reliability + bl_reliability
        if total_reliability > 0:
            reliability_adjusted_weights = {
                'rsi': weights['rsi'] * (rsi_reliability / total_reliability),
                'kc': weights['kc'] * (kc_reliability / total_reliability), 
                'bl': weights['bl'] * (bl_reliability / total_reliability),
                'base': weights['base']
            }
        else:
            reliability_adjusted_weights = weights
        
        # Calculate weighted adjustment
        weighted_adjustment = (
            reliability_adjusted_weights['base'] * 1.0 + 
            reliability_adjusted_weights['rsi'] * rsi_adjustment + 
            reliability_adjusted_weights['kc'] * kc_adjustment +
            reliability_adjusted_weights['bl'] * bl_adjustment
        )
        
        # Apply smooth adjustment
        max_adjustment = 2.5  
        min_adjustment = 0.4  
        smooth_adjustment = np.tanh(weighted_adjustment - 1) * 0.5 + 1
        final_adjustment = np.clip(smooth_adjustment, min_adjustment, max_adjustment)
        adjusted_cvar = traditional_cvar * final_adjustment
        
        # Component contributions
        base_contribution = traditional_cvar * reliability_adjusted_weights['base']
        rsi_contribution = traditional_cvar * reliability_adjusted_weights['rsi'] * rsi_adjustment
        kc_contribution = traditional_cvar * reliability_adjusted_weights['kc'] * kc_adjustment
        bl_contribution = traditional_cvar * reliability_adjusted_weights['bl'] * bl_adjustment
        
        risk_components = {
            "traditional_cvar": traditional_cvar,
            "adjusted_cvar": adjusted_cvar,
            "var": var,
            "market_regime": market_regime,
            "rsi_adjustment": rsi_adjustment,
            "kc_adjustment": kc_adjustment,
            "bl_adjustment": bl_adjustment,
            "weighted_adjustment": weighted_adjustment,
            "final_adjustment": final_adjustment,
            "base_contribution": base_contribution,
            "rsi_contribution": rsi_contribution,
            "kc_contribution": kc_contribution,
            "bl_contribution": bl_contribution,
            "weight_distribution": reliability_adjusted_weights,
            "reliability_scores": {
                "rsi": rsi_reliability,
                "kc": kc_reliability,
                "bl": bl_reliability
            }
        }
        
        # Add detailed indicator information
        if self.indicator_data is not None and self.use_enhanced_features:
            if 'rsi' in self.indicator_data:
                rsi_df = self.indicator_data['rsi']
                rsi_components = {}
                
                for col in ['result_rsi_status', 'result_rsi_bullish_div', 
                            'result_rsi_bearish_div', 'result_rsi_double_bottom']:
                    if col in rsi_df.columns:
                        rsi_components[col] = rsi_df[col].iloc[-1]
                        
                risk_components["rsi_details"] = rsi_components
                
            if 'keltner_channels' in self.indicator_data:
                kc_df = self.indicator_data['keltner_channels']
                kc_components = {}
                
                for col in ['kc_trend', 'kc_trend_retest', 'kc_trend_strength']:
                    if col in kc_df.columns:
                        kc_components[col] = kc_df[col].iloc[-1]
                        
                risk_components["kc_details"] = kc_components
        
        # Add Black-Litterman details
        if self.bl_results is not None:
            risk_components["bl_details"] = {
                "return_adjustment": self.bl_results.get('return_adjustment', 0),
                "risk_adjustment": self.bl_results.get('risk_adjustment', 0),
                "views_summary": self.bl_results.get('views_summary', {}),
                "posterior_volatility": self.bl_results.get('posterior_volatility_annualized', 0)
            }
                
        return risk_components


class BitcoinTradingEnv(gym.Env):
    """Optimized Bitcoin trading environment for PPO reinforcement learning"""
    
    def __init__(self, df, initial_balance=100000, 
                 maker_fee=0.001, taker_fee=0.003, 
                 slippage_factor=0.001, 
                 window_size=30, 
                 profit_reward_weight=1.0, cvar_penalty_weight=0.1, hold_penalty=-0.0001,
                 min_transaction_pct=0.1, max_hold_duration=20):
        super(BitcoinTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_factor = slippage_factor
        self.window_size = window_size
        self.profit_reward_weight = profit_reward_weight
        self.cvar_penalty_weight = cvar_penalty_weight
        self.hold_penalty = hold_penalty
        self.min_transaction_pct = min_transaction_pct
        self.max_hold_duration = max_hold_duration
        
        # Action space: continuous allocation percentage [0, 1]
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # Prepare feature columns
        base_features = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in df.columns]
        rsi_features = ['result_rsi'] if 'result_rsi' in df.columns else []
        kc_features = [col for col in df.columns if col.startswith('kc_') and col not in ['kc_trend', 'kc_trend_retest']]
        lstm_feature = ['lstm_pred_pct_change'] if 'lstm_pred_pct_change' in df.columns else []
        self.ppo_feature_cols = base_features + rsi_features + kc_features + lstm_feature
        
        if 'close_unscaled' not in self.df.columns:
            raise ValueError("Required column 'close_unscaled' not found in DataFrame")
            
        # Observation space
        num_df_features = len(self.ppo_feature_cols)
        self.observation_shape = (window_size, num_df_features + 3)  # +3 for balance, btc_value, prev_action
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=self.observation_shape, 
            dtype=np.float32
        )
        
        # State tracking
        self.returns_window = deque(maxlen=30)
        self.portfolio_values = []
        self.btc_buy_hold_values = []
        self.transaction_history = []
        self.market_state = {}
        self.trading_reasons = {}
        self.cooldown_period = 3
        
        # Performance tracking
        self.recent_trade_success_rate = 0.5
        
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.window_size
        
        if len(self.df) <= self.current_step:
            raise ValueError(f"DataFrame too short for window_size. Need at least {self.window_size + 1} rows, got {len(self.df)}")
            
        # Reset portfolio state
        self.balance = self.initial_balance
        self.btc_held = 0
        self.net_worth_history = [self.initial_balance]
        self.previous_action = 0.5
        self.last_buy_price = 0
        self.hold_steps_count = 0
        self.prev_net_worth = self.initial_balance
        self.portfolio_values = [self.initial_balance]
        
        # Initialize buy & hold comparison
        initial_btc_price = self.df.iloc[self.window_size]['close_unscaled']
        self.initial_btc_amount = self.initial_balance / initial_btc_price
        self.btc_buy_hold_values = [self.initial_balance]
        
        # Wallet history tracking
        self.wallet_history = [{
            'step': self.current_step,
            'timestamp': self.df.index[self.current_step] if self.current_step < len(self.df.index) else None,
            'portfolio_value': self.initial_balance,
            'usd_balance': self.initial_balance,
            'btc_held': 0,
            'btc_value_usd': 0,
            'usd_ratio': 1.0,
            'btc_ratio': 0.0
        }]
        
        # Reset tracking variables
        self.consecutive_same_actions = 0
        self.action_history = []
        self.reward_history = []
        self.transaction_history = []
        self.trading_reasons = {}
        self.cooldown_counter = 0
        self.last_trade_step = 0
        
        return self._get_observation()

    def _get_observation(self):
        """Get current environment observation"""
        if self.current_step < self.window_size or self.current_step >= len(self.df):
            return np.zeros(self.observation_shape, dtype=np.float32)
            
        # Get window of market data
        frame = self.df.iloc[self.current_step - self.window_size : self.current_step]
        
        if len(frame) != self.window_size:
            padding_needed = self.window_size - len(frame)
            padding = np.zeros((padding_needed, len(self.ppo_feature_cols)))
            obs_data = np.vstack((padding, frame[self.ppo_feature_cols].values))
        else:
            obs_data = frame[self.ppo_feature_cols].values
            
        # Add portfolio information
        norm_balance = self.balance / self.initial_balance
        current_price_for_obs = self.df.iloc[self.current_step - 1]['close_unscaled'] if self.current_step > 0 else 0
        norm_btc_value = (self.btc_held * current_price_for_obs) / self.initial_balance
        norm_prev_action = self.previous_action
        
        # Create account info matrix
        account_info = np.array([[norm_balance, norm_btc_value, norm_prev_action]] * self.window_size)
        combined_features = np.hstack((obs_data, account_info)).astype(np.float32)
        
        # Ensure correct shape
        if combined_features.shape[1] > self.observation_shape[1]:
            combined_features = combined_features[:, :self.observation_shape[1]]
        else:
            padding = np.zeros((combined_features.shape[0], 
                            self.observation_shape[1] - combined_features.shape[1]))
            combined_features = np.hstack((combined_features, padding))
            
        return combined_features

    def _get_current_price(self):
        """Get current Bitcoin price"""
        if self.current_step < len(self.df):
            return self.df.iloc[self.current_step]['close_unscaled']
        return 0
    
    def update_market_state(self):
        """Optimized market state update for portfolio optimization"""
        if self.current_step >= len(self.df):
            return
            
        current_row = self.df.iloc[self.current_step]
        
        # RSI state
        rsi_value = current_row.get('result_rsi', 50)
        self.market_state['rsi'] = {
            'value': rsi_value,
            'status': self._get_rsi_status(rsi_value),
            'bullish_div': current_row.get('result_rsi_bullish_div', False),
            'bearish_div': current_row.get('result_rsi_bearish_div', False),
            'double_bottom': current_row.get('result_rsi_double_bottom', False),
            'pivot_high': current_row.get('result_rsi_pivot_high', False),
            'pivot_low': current_row.get('result_rsi_pivot_low', False)
        }
        
        # Keltner Channel state
        self.market_state['keltner'] = {
            'trend': current_row.get('kc_trend', 'Neutral'),
            'retest': current_row.get('kc_trend_retest', 'No_Retest'),
            'trend_strength': current_row.get('kc_trend_strength', 0.0),
            'upper': current_row.get('kc_upper', 0.0),
            'middle': current_row.get('kc_middle', 0.0),
            'lower': current_row.get('kc_lower', 0.0)
        }
        
        # LSTM prediction state
        self.market_state['lstm_prediction'] = {
            'pct_change': current_row.get('lstm_pred_pct_change', 0.0)
        }
        
        # Price action state
        self.market_state['price_action'] = {
            'price': current_row['close_unscaled'],
            'daily_return': current_row.get('daily_return', 0.0)
        }
        
        # Portfolio state
        current_price = current_row['close_unscaled']
        total_value = self.balance + (self.btc_held * current_price)
        self.market_state['portfolio'] = {
            'balance': self.balance,
            'btc_held': self.btc_held,
            'net_worth': total_value,
            'cash_ratio': self.balance / total_value if total_value > 0 else 0
        }
        
        # Volatility metrics for risk assessment
        if len(self.returns_window) > 5:
            self.market_state['volatility'] = {
                'recent_vol': np.std(self.returns_window),
                'cvar': self._calculate_current_cvar()
            }
        else:
            self.market_state['volatility'] = {
                'recent_vol': 0.0,
                'cvar': 0.0
            }
    
    def _calculate_current_cvar(self):
        """Calculate current CVaR from returns window"""
        if len(self.returns_window) < 10:
            return 0.0
            
        returns_array = np.array(self.returns_window)
        threshold = np.percentile(returns_array, 10)
        below_threshold = returns_array[returns_array <= threshold]
        
        return below_threshold.mean() if len(below_threshold) > 0 else 0.0
    
    def _get_rsi_status(self, rsi_value):
        """Get RSI status string"""
        if rsi_value > 70:
            return "Overbought"
        elif rsi_value < 30:
            return "Oversold"
        else:
            return "Neutral"
    
    def can_execute_buy(self):
        """Check if buy order can be executed"""
        portfolio_value = self.balance + (self.btc_held * self._get_current_price())
        min_transaction_value = portfolio_value * self.min_transaction_pct
        
        return self.balance >= min_transaction_value and self.cooldown_counter == 0
    
    def can_execute_sell(self):
        """Check if sell order can be executed"""
        current_price = self._get_current_price()
        portfolio_value = self.balance + (self.btc_held * current_price)
        min_transaction_value = portfolio_value * self.min_transaction_pct
        btc_value = self.btc_held * current_price
        
        return self.btc_held > 0 and btc_value >= min_transaction_value and self.cooldown_counter == 0

    def calculate_trade_decision(self, action_value):
        """Optimized PPO trading decision using full pipeline: Indicators->LSTM->BL->CVaR"""
        self.update_market_state()
        
        trade_reasons = []
        decision = "HOLD"
        position_size = 0.0
        
        # Check cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            decision = "HOLD"
            trade_reasons.append(f"Trade cooldown: {self.cooldown_counter} steps remaining")
            return decision, position_size, trade_reasons
            
        current_price = self._get_current_price()
        
        # OPTIMASI 1: Dynamic signal calculation with confidence weighting
        signal_components = self._calculate_signal_components()
        signal_weights = self._calculate_dynamic_signal_weights()
        
        # Calculate weighted overall signal
        overall_signal = 0
        for component, signal in signal_components.items():
            weighted_signal = signal * signal_weights[component]
            overall_signal += weighted_signal
            
            if abs(weighted_signal) > 0.1:  # Only report significant signals
                trade_reasons.append(f"{component.upper()}: {signal:.2f} (weight: {signal_weights[component]:.2f})")
        
        # OPTIMASI 2: Risk-adjusted position sizing
        risk_factor = self._calculate_current_risk_factor()
        base_position_size = self._calculate_base_position_size(overall_signal)
        risk_adjusted_size = base_position_size * (1 / risk_factor)
        
        # OPTIMASI 3: Portfolio optimization decision logic
        portfolio_state = self.market_state['portfolio']
        current_allocation = portfolio_state['btc_held'] * current_price / portfolio_state['net_worth'] if portfolio_state['net_worth'] > 0 else 0
        
        # Trading decisions with portfolio optimization
        if overall_signal < -0.5 and current_allocation > 0.1:  # Sell signal with meaningful position
            decision = "SELL"
            
            # OPTIMASI: Graduated selling based on signal strength
            if overall_signal < -1.0:
                position_size = min(1.0, risk_adjusted_size)  # Strong sell
            elif overall_signal < -0.75:
                position_size = min(0.7, risk_adjusted_size)  # Moderate sell
            else:
                position_size = min(0.4, risk_adjusted_size)  # Light sell
                
            trade_reasons.append(f"SELL decision: signal={overall_signal:.2f}, size={position_size:.2f}")
            
            if not self.can_execute_sell():
                decision = "HOLD"
                trade_reasons = ["Cannot execute sell - insufficient BTC or minimum not met"]
                position_size = 0.0
                
        elif overall_signal > 0.5 and current_allocation < 0.9:  # Buy signal with room to increase
            decision = "BUY"
            
            # OPTIMASI: Graduated buying based on signal strength and available cash
            available_cash_ratio = portfolio_state['cash_ratio']
            
            if overall_signal > 1.0:
                target_buy_ratio = min(0.8, available_cash_ratio * 0.9)  # Strong buy
            elif overall_signal > 0:
                if current_allocation > 0.9:
                    trade_reasons.append(f"Bullish signals but already fully allocated ({current_allocation:.1%})")
                else:
                    trade_reasons.append(f"Mild bullish signals (score: {overall_signal:.2f}) - threshold not met")
            else:
                if current_allocation < 0.1:
                    trade_reasons.append(f"Bearish signals but minimal position to sell ({current_allocation:.1%})")
                else:
                    trade_reasons.append(f"Mild bearish signals (score: {overall_signal:.2f}) - threshold not met")
        
        # OPTIMASI 4: Add context information
        if decision != "HOLD":
            trade_reasons.append(f"Risk factor: {risk_factor:.2f}")
            trade_reasons.append(f"Current allocation: {current_allocation:.1%}")
            trade_reasons.append(f"Available cash: {portfolio_state['cash_ratio']:.1%}")
        
        # Store comprehensive trading rationale
        self.trading_reasons[self.current_step] = {
            "decision": decision,
            "position_size": position_size,
            "reasons": trade_reasons,
            "signal_components": signal_components,
            "signal_weights": signal_weights,
            "overall_signal": overall_signal,
            "risk_factor": risk_factor,
            "current_allocation": current_allocation,
            "market_state": {
                "price": current_price,
                "portfolio_value": portfolio_state['net_worth']
            }
        }
        
        return decision, position_size, trade_reasons

    def _calculate_signal_components(self):
        """Calculate individual signal components from pipeline"""
        signals = {}
        
        # COMPONENT 1: RSI signals (from Indicators stage)
        signals['rsi'] = self._get_rsi_signals()
        
        # COMPONENT 2: Keltner Channel signals (from Indicators stage)  
        signals['keltner'] = self._get_keltner_signals()
        
        # COMPONENT 3: LSTM prediction signals (from LSTM stage)
        signals['lstm'] = self._get_lstm_signals()
        
        # COMPONENT 4: Black-Litterman signals (from BL stage)
        signals['black_litterman'] = self._get_bl_signals()
        
        # COMPONENT 5: CVaR risk signals (from CVaR stage)
        signals['cvar_risk'] = self._get_cvar_signals()
        
        # COMPONENT 6: Portfolio optimization signals
        signals['portfolio'] = self._get_portfolio_signals()
        
        return signals

    def _calculate_dynamic_signal_weights(self):
        """Calculate dynamic weights for signal components based on current conditions"""
        
        # OPTIMASI: Base weights for Bitcoin spot trading
        base_weights = {
            'rsi': 0.15,
            'keltner': 0.15, 
            'lstm': 0.25,
            'black_litterman': 0.20,
            'cvar_risk': 0.10,
            'portfolio': 0.15
        }
        
        # OPTIMASI: Adjust weights based on market conditions
        market_vol = self.market_state.get('volatility', {}).get('recent_vol', 0.03)
        
        if market_vol > 0.06:  # High volatility - trust risk management more
            base_weights['cvar_risk'] *= 1.5
            base_weights['lstm'] *= 0.8  # LSTM less reliable in high vol
            base_weights['portfolio'] *= 1.2  # Portfolio management more important
            
        elif market_vol < 0.02:  # Low volatility - trust predictions more
            base_weights['lstm'] *= 1.3
            base_weights['black_litterman'] *= 1.2
            base_weights['cvar_risk'] *= 0.7
            
        # OPTIMASI: Adjust based on recent performance
        if hasattr(self, 'recent_trade_success_rate'):
            if self.recent_trade_success_rate > 0.7:  # Good recent performance
                # Trust the system more, emphasize prediction components
                base_weights['lstm'] *= 1.2
                base_weights['black_litterman'] *= 1.1
            elif self.recent_trade_success_rate < 0.4:  # Poor recent performance
                # Be more conservative, emphasize risk management
                base_weights['cvar_risk'] *= 1.4
                base_weights['portfolio'] *= 1.3
                base_weights['lstm'] *= 0.8
        
        # Normalize weights to sum to 1
        total_weight = sum(base_weights.values())
        normalized_weights = {k: v/total_weight for k, v in base_weights.items()}
        
        return normalized_weights

    def _get_rsi_signals(self):
        """Extract RSI signals from market state"""
        rsi_signal = 0
        
        if 'rsi' not in self.market_state:
            return rsi_signal
            
        rsi_data = self.market_state['rsi']
        rsi_value = rsi_data['value']
        
        # RSI level signals
        if rsi_value < 25:
            rsi_signal += 1.2  # Very oversold
        elif rsi_value < 35:
            rsi_signal += 0.6  # Oversold
        elif rsi_value > 75:
            rsi_signal -= 1.2  # Very overbought
        elif rsi_value > 65:
            rsi_signal -= 0.6  # Overbought
        
        # Pattern signals (high confidence)
        if rsi_data.get('bullish_div', False):
            rsi_signal += 0.8
        if rsi_data.get('bearish_div', False):
            rsi_signal -= 0.8
        if rsi_data.get('double_bottom', False):
            rsi_signal += 0.6
        
        return np.clip(rsi_signal, -2.0, 2.0)

    def _get_keltner_signals(self):
        """Extract Keltner Channel signals from market state"""
        kc_signal = 0
        
        if 'keltner' not in self.market_state:
            return kc_signal
            
        kc_data = self.market_state['keltner']
        trend = kc_data['trend']
        strength = kc_data['trend_strength']
        retest = kc_data['retest']
        
        # Trend signals
        if trend == "Uptrend":
            kc_signal += 0.5 + (strength * 0.5)
        elif trend == "Downtrend":
            kc_signal -= 0.5 + (strength * 0.5)
        
        # Retest signals (high probability setups)
        if retest == "Uptrend_Retest":
            kc_signal += 0.8
        elif retest == "Downtrend_Retest":
            kc_signal -= 0.8
        
        return np.clip(kc_signal, -2.0, 2.0)

    def _get_lstm_signals(self):
        """Extract LSTM prediction signals from market state"""
        lstm_signal = 0
        
        if 'lstm_prediction' not in self.market_state:
            return lstm_signal
            
        pred_change = self.market_state['lstm_prediction']['pct_change']
        
        # Scale prediction to signal strength
        if pred_change > 0.02:  # >2% predicted gain
            lstm_signal = 1.5
        elif pred_change > 0.01:  # >1% predicted gain
            lstm_signal = 1.0
        elif pred_change > 0.005:  # >0.5% predicted gain
            lstm_signal = 0.5
        elif pred_change < -0.02:  # >2% predicted loss
            lstm_signal = -1.5
        elif pred_change < -0.01:  # >1% predicted loss
            lstm_signal = -1.0
        elif pred_change < -0.005:  # >0.5% predicted loss
            lstm_signal = -0.5
        
        return lstm_signal

    def _get_bl_signals(self):
        """Extract Black-Litterman signals"""
        # This would be derived from the BL optimal allocation vs current allocation
        bl_signal = 0
        
        # Implementation would compare current portfolio allocation 
        # with BL optimal allocation to generate rebalancing signals
        
        return bl_signal

    def _get_cvar_signals(self):
        """Extract CVaR risk signals"""
        cvar_signal = 0
        
        if 'volatility' in self.market_state:
            current_cvar = self.market_state['volatility']['cvar']
            
            # If CVaR indicates high risk, signal to reduce position
            if current_cvar < -0.05:  # High tail risk
                cvar_signal = -0.8
            elif current_cvar < -0.03:  # Moderate tail risk
                cvar_signal = -0.4
            elif current_cvar > -0.01:  # Low tail risk
                cvar_signal = 0.3
        
        return cvar_signal

    def _get_portfolio_signals(self):
        """Extract portfolio optimization signals"""
        portfolio_signal = 0
        
        # Profit taking logic
        if self.last_buy_price > 0:
            current_price = self._get_current_price()
            profit_pct = (current_price / self.last_buy_price) - 1
            
            if profit_pct > 0.15:  # >15% profit
                portfolio_signal = -1.0
            elif profit_pct > 0.08:  # >8% profit
                portfolio_signal = -0.5
            elif profit_pct < -0.08:  # >8% loss
                portfolio_signal = -0.8  # Cut losses
        
        # Hold duration penalty
        if self.hold_steps_count > self.max_hold_duration:
            if self.btc_held > 0:
                portfolio_signal -= 0.3
            elif self.balance > 0:
                portfolio_signal += 0.3
        
        return portfolio_signal

    def _calculate_current_risk_factor(self):
        """Calculate current risk factor for position sizing"""
        base_risk = 1.0
        
        # Volatility adjustment
        if 'volatility' in self.market_state:
            vol = self.market_state['volatility']['recent_vol']
            if vol > 0.06:  # High volatility
                base_risk *= 1.5
            elif vol > 0.04:  # Moderate volatility
                base_risk *= 1.2
            elif vol < 0.02:  # Low volatility
                base_risk *= 0.8
        
        # CVaR adjustment
        if 'volatility' in self.market_state:
            cvar = self.market_state['volatility']['cvar']
            if cvar < -0.05:  # High tail risk
                base_risk *= 1.3
            elif cvar < -0.03:  # Moderate tail risk
                base_risk *= 1.1
        
        return max(0.5, min(base_risk, 2.5))

    def _calculate_base_position_size(self, signal_strength):
        """Calculate base position size based on signal strength"""
        abs_signal = abs(signal_strength)
        
        if abs_signal > 1.5:
            return 0.8  # Strong signal
        elif abs_signal > 1.0:
            return 0.6  # Moderate signal
        elif abs_signal > 0.7:
            return 0.4  # Mild signal
        else:
            return 0.2  # Weak signal

    def _execute_buy(self, buy_percentage, current_price):
        """Execute buy order"""
        portfolio_value = self.balance + (self.btc_held * current_price)
        min_transaction_value = portfolio_value * self.min_transaction_pct
        
        if self.balance < min_transaction_value:
            self.hold_steps_count += 1
            return False
            
        amount_to_spend = self.balance * buy_percentage
        
        if amount_to_spend < min_transaction_value:
            amount_to_spend = min_transaction_value
            
        # Apply slippage and fees
        execution_price = current_price * (1 + self.slippage_factor)
        fee = amount_to_spend * self.taker_fee
        amount_to_spend_on_btc = amount_to_spend - fee
        
        btc_to_buy = amount_to_spend_on_btc / execution_price
        self.btc_held += btc_to_buy
        self.balance -= amount_to_spend
        
        if self.balance < 0:
            self.balance = 0
            
        # Update average buy price
        if self.btc_held - btc_to_buy > 0 and self.last_buy_price > 0:
            total_btc = self.btc_held
            prev_btc = total_btc - btc_to_buy
            self.last_buy_price = ((prev_btc * self.last_buy_price) + (btc_to_buy * execution_price)) / total_btc
        else:
            self.last_buy_price = execution_price
            
        self.hold_steps_count = 0
        return True

    def _execute_sell(self, sell_percentage, current_price):
        """Execute sell order"""
        realized_profit = 0
        
        if self.btc_held <= 0:
            self.hold_steps_count += 1
            return realized_profit, False
            
        portfolio_value = self.balance + (self.btc_held * current_price)
        min_transaction_value = portfolio_value * self.min_transaction_pct
        btc_sold_amount = self.btc_held * sell_percentage
        
        # Apply slippage and fees
        execution_price = current_price * (1 - self.slippage_factor)
        sell_value_gross = btc_sold_amount * execution_price
        
        if sell_value_gross < min_transaction_value:
            self.hold_steps_count += 1
            return realized_profit, False
            
        fee = sell_value_gross * self.maker_fee
        sell_value_net = sell_value_gross - fee
        
        self.balance += sell_value_net
        self.btc_held -= btc_sold_amount
        
        # Calculate realized profit
        if self.last_buy_price > 0:
            cost_basis = btc_sold_amount * self.last_buy_price
            realized_profit = sell_value_net - cost_basis
            
        self.hold_steps_count = 0
        return realized_profit, True

    def step(self, action):
        """Execute one environment step"""
        if self.current_step >= len(self.df):
            return self._get_observation(), 0, True, {'error': 'Step out of bounds - end of data'}
            
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            
        # Process action
        action_value = float(action[0]) if hasattr(action, "shape") else float(action)
        action_value = max(0.0, min(action_value, 1.0))
        
        self.action_history.append(action_value)
        
        # Get trading decision
        decision, position_size, reasons = self.calculate_trade_decision(action_value)
        
        current_price = self.df.iloc[self.current_step]['close_unscaled']
        prev_net_worth = self.balance + (self.btc_held * current_price)
        
        # Update buy & hold comparison
        btc_buy_hold_value = self.initial_btc_amount * current_price
        self.btc_buy_hold_values.append(btc_buy_hold_value)
        
        # Execute trading decision
        realized_profit_this_step = 0
        trade_executed = False
        
        if decision == "SELL" and position_size > 0:
            btc_amount_before = self.btc_held
            realized_profit_this_step, sell_executed = self._execute_sell(position_size, current_price)
            
            if sell_executed:
                trade_executed = True
                btc_sold = btc_amount_before - self.btc_held
                
                self.transaction_history.append({
                    "step": self.current_step,
                    "timestamp": self.df.index[self.current_step],
                    "type": "SELL",
                    "amount_pct": position_size * 100,
                    "price": current_price,
                    "btc_amount": btc_sold,
                    "usd_value": btc_sold * current_price,
                    "profit": realized_profit_this_step,
                    "reasons": reasons
                })
                
                self.cooldown_counter = self.cooldown_period
                self.last_trade_step = self.current_step
                
        elif decision == "BUY" and position_size > 0:
            balance_before = self.balance
            buy_executed = self._execute_buy(position_size, current_price)
            
            if buy_executed:
                trade_executed = True
                cash_spent = balance_before - self.balance
                
                self.transaction_history.append({
                    "step": self.current_step,
                    "timestamp": self.df.index[self.current_step],
                    "type": "BUY",
                    "amount_pct": position_size * 100,
                    "price": current_price,
                    "btc_amount": cash_spent / (current_price * (1 + self.taker_fee)),
                    "usd_value": cash_spent,
                    "reasons": reasons
                })
                
                self.cooldown_counter = self.cooldown_period
                self.last_trade_step = self.current_step
                
        else:
            self.hold_steps_count += 1
            
        # Update portfolio tracking
        current_net_worth = self.balance + (self.btc_held * current_price)
        self.net_worth_history.append(current_net_worth)
        self.portfolio_values.append(current_net_worth)
        
        self.wallet_history.append({
            'step': self.current_step,
            'timestamp': self.df.index[self.current_step] if self.current_step < len(self.df.index) else None,
            'portfolio_value': current_net_worth,
            'usd_balance': self.balance,
            'btc_held': self.btc_held,
            'btc_value_usd': self.btc_held * current_price,
            'usd_ratio': (self.balance / current_net_worth) if current_net_worth > 0 else 0,
            'btc_ratio': ((self.btc_held * current_price) / current_net_worth) if current_net_worth > 0 else 0
        })
        
        # Calculate returns and rewards
        pct_change = (current_net_worth - prev_net_worth) / max(prev_net_worth, 1e-6)
        self.returns_window.append(pct_change)
        
        # Reward components
        portfolio_change_reward = pct_change
        realized_trade_bonus = (realized_profit_this_step / max(prev_net_worth, 1e-6)) * self.profit_reward_weight
        
        # CVaR penalty
        cvar_penalty = 0
        if len(self.returns_window) >= 10:
            returns_array = np.array(self.returns_window)
            cvar_threshold = np.percentile(returns_array, 10)
            below_threshold = returns_array[returns_array <= cvar_threshold]
            cvar = below_threshold.mean() if len(below_threshold) > 0 else 0
            cvar_penalty = abs(cvar) * self.cvar_penalty_weight if cvar < 0 else 0
            
        # Hold penalty
        hold_penalty = 0
        if self.hold_steps_count > self.max_hold_duration:
            hold_penalty = self.hold_penalty * (self.hold_steps_count - self.max_hold_duration)
            
        # Trade profit bonus
        trade_profit_bonus = 0
        if realized_profit_this_step > 0:
            trade_profit_bonus = 0.5 * (realized_profit_this_step / max(prev_net_worth, 1e-6))
            
        # Outperformance reward
        btc_hold_pct_change = (btc_buy_hold_value - self.btc_buy_hold_values[-2]) / self.btc_buy_hold_values[-2] if len(self.btc_buy_hold_values) > 1 else 0
        outperformance_reward = 0
        if pct_change > btc_hold_pct_change:
            outperformance_reward = 0.2 * (pct_change - btc_hold_pct_change)
            
        # Final reward calculation
        weighted_portfolio_change = portfolio_change_reward * 0.8
        weighted_hold_penalty = hold_penalty * 1.5
        
        reward = weighted_portfolio_change + realized_trade_bonus + trade_profit_bonus - cvar_penalty + weighted_hold_penalty + outperformance_reward
        reward = np.clip(reward, -1.0, 1.0)
        
        self.reward_history.append(reward)
        self.prev_net_worth = current_net_worth
        self.current_step += 1
        
        # Check termination conditions
        done = self.current_step >= len(self.df) - 1 or current_net_worth < 0.1 * self.initial_balance
        
        # Prepare info dictionary
        info = {
            'portfolio_value': current_net_worth, 
            'balance': self.balance,
            'btc_held': self.btc_held, 
            'current_price': current_price,
            'btc_buy_hold_value': btc_buy_hold_value,
            'trade_decision': {
                'action': decision,
                'position_size': position_size,
                'reasons': reasons,
            },
            'reward_debug': {
                'portfolio_chg': portfolio_change_reward,
                'trade_bonus': realized_trade_bonus,
                'cvar_pen': cvar_penalty,
                'hold_pen': hold_penalty,
                'outperformance': outperformance_reward
            }
        }
        
        return self._get_observation(), reward, done, info

    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        initial_value = self.initial_balance
        final_value = self.net_worth_history[-1] if self.net_worth_history else self.initial_balance
        
        # Calculate returns
        returns = []
        for i in range(1, len(self.net_worth_history)):
            ret = (self.net_worth_history[i] - self.net_worth_history[i-1]) / self.net_worth_history[i-1]
            returns.append(ret)
            
        # Buy & hold comparison
        final_buy_hold = self.btc_buy_hold_values[-1] if self.btc_buy_hold_values else self.initial_balance
        buy_hold_return = ((final_buy_hold - self.initial_balance) / self.initial_balance) * 100
        
        # Trade statistics
        buy_count = sum(1 for t in self.transaction_history if t['type'] == 'BUY')
        sell_count = sum(1 for t in self.transaction_history if t['type'] == 'SELL')
        
        # Sharpe ratio calculation
        sharpe_ratio = 0
        if len(returns) > 30:
            risk_free_rate = 0.02 / 365
            excess_returns = np.array(returns) - risk_free_rate
            if np.std(returns) > 0:
                sharpe_ratio = (np.mean(excess_returns) / np.std(returns)) * np.sqrt(365)
        
        return {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return': (final_value - initial_value) / initial_value * 100,
            'buy_hold_return': buy_hold_return,
            'outperformance': ((final_value - initial_value) / initial_value * 100) - buy_hold_return,
            'n_trades': len(self.transaction_history),
            'buy_count': buy_count,
            'sell_count': sell_count,
            'btc_held': self.btc_held,
            'cash_balance': self.balance,
            'episode_length': len(self.action_history),
            'portfolio_values': self.net_worth_history,
            'btc_buy_hold_values': self.btc_buy_hold_values,
            'returns': returns,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': calculate_max_drawdown(self.net_worth_history),
            'transactions': self.transaction_history
        }

    def get_transaction_history(self):
        """Get transaction history"""
        return self.transaction_history

    def get_trading_reasons(self):
        """Get trading decision reasons"""
        return self.trading_reasons

    def plot_wallet_composition(self, save_path='./results/wallet_composition.png', interactive=False):
        """Plot wallet composition over time"""
        if not hasattr(self, 'wallet_history') or not self.wallet_history:
            return
            
        steps = [entry['step'] for entry in self.wallet_history]
        timestamps = [entry['timestamp'] for entry in self.wallet_history]
        portfolio_values = [entry['portfolio_value'] for entry in self.wallet_history]
        usd_balances = [entry['usd_balance'] for entry in self.wallet_history]
        btc_values = [entry['btc_value_usd'] for entry in self.wallet_history]
        usd_ratios = [entry['usd_ratio'] * 100 for entry in self.wallet_history]
        btc_ratios = [entry['btc_ratio'] * 100 for entry in self.wallet_history]
        
        plt.style.use('default')
        
        fig = plt.figure(figsize=(16, 12), dpi=100)
        gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1], width_ratios=[2, 1], 
                      hspace=0.3, wspace=0.3)
        
        # Main portfolio evolution plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.fill_between(timestamps, 0, usd_balances, alpha=0.6, color='#27ae60', label='USD Balance')
        ax1.fill_between(timestamps, usd_balances, portfolio_values, alpha=0.6, color='#f39c12', label='BTC Value')
        ax1.plot(timestamps, portfolio_values, color='#2c3e50', linewidth=3, label='Total Portfolio', alpha=0.9)
        
        # Mark buy/sell transactions
        buy_times = []
        buy_values = []
        sell_times = []
        sell_values = []
        
        for transaction in self.transaction_history:
            if transaction['timestamp'] in timestamps:
                if transaction['type'] == 'BUY':
                    buy_times.append(transaction['timestamp'])
                    buy_values.append(transaction['usd_value'])
                elif transaction['type'] == 'SELL':
                    sell_times.append(transaction['timestamp'])
                    sell_values.append(transaction['usd_value'])
        
        if buy_times:
            ax1.scatter(buy_times, buy_values, marker='^', color='#16a085', s=120, 
                       zorder=5, alpha=0.8, edgecolors='white', linewidth=2, label='Buy Orders')
        if sell_times:
            ax1.scatter(sell_times, sell_values, marker='v', color='#e74c3c', s=120, 
                       zorder=5, alpha=0.8, edgecolors='white', linewidth=2, label='Sell Orders')
        
        ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=11)
        ax1.set_title('Portfolio Value Evolution & Trading Activity', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Value (USD)', fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        
        for spine in ax1.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#34495e')
        
        # Asset allocation over time
        ax2 = fig.add_subplot(gs[1, 0])
        width = 0.6
        x_pos = np.arange(len(timestamps[::20]))
        
        ax2.bar(x_pos, [usd_ratios[i] for i in range(0, len(usd_ratios), 20)], 
               width, color='#27ae60', alpha=0.7, label='USD %')
        ax2.bar(x_pos, [btc_ratios[i] for i in range(0, len(btc_ratios), 20)], 
               width, bottom=[usd_ratios[i] for i in range(0, len(usd_ratios), 20)], 
               color='#f39c12', alpha=0.7, label='BTC %')
        
        ax2.set_xticks(x_pos[::5])
        ax2.set_xticklabels([timestamps[i].strftime('%Y-%m') for i in range(0, len(timestamps), 100)], rotation=45)
        ax2.set_title('Asset Allocation Over Time', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Allocation (%)', fontsize=11)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.4)
        ax2.set_ylim(0, 100)
        
        # Final portfolio composition pie chart
        ax3 = fig.add_subplot(gs[1, 1])
        final_usd_ratio = usd_ratios[-1] if usd_ratios else 0
        final_btc_ratio = btc_ratios[-1] if btc_ratios else 0
        
        sizes = [final_usd_ratio, final_btc_ratio]
        labels = [f'USD\n{final_usd_ratio:.1f}%', f'BTC\n{final_btc_ratio:.1f}%']
        colors = ['#27ae60', '#f39c12']
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, explode=explode,
                                          autopct='%1.1f%%', startangle=90, shadow=True,
                                          textprops={'fontsize': 10, 'fontweight': 'bold'})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax3.set_title('Final Portfolio\nComposition', fontsize=12, fontweight='bold')
        
        # Trading activity timeline
        ax4 = fig.add_subplot(gs[2, :])
        
        trade_dates = [t['timestamp'] for t in self.transaction_history]
        trade_types = [1 if t['type'] == 'BUY' else -1 for t in self.transaction_history]
        trade_amounts = [t['amount_pct'] for t in self.transaction_history]
        
        colors = ['#16a085' if t == 1 else '#e74c3c' for t in trade_types]
        
        bars = ax4.bar(range(len(trade_dates)), [t * a for t, a in zip(trade_types, trade_amounts)], 
                      color=colors, alpha=0.7, width=0.8)
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        ax4.set_title('Trading Activity Timeline', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Transaction Size (%)', fontsize=11)
        ax4.set_xlabel('Trade Number', fontsize=11)
        ax4.grid(True, linestyle=':', alpha=0.4)
        
        if len(trade_dates) > 0:
            ax4.set_xticks(range(0, len(trade_dates), max(1, len(trade_dates)//10)))
            ax4.set_xticklabels([f"T{i+1}" for i in range(0, len(trade_dates), max(1, len(trade_dates)//10))])
        
        fig.suptitle('Bitcoin Trading Strategy - Comprehensive Portfolio Analysis', 
                    fontsize=18, fontweight='bold', y=0.96)
        
        # Performance summary text
        if hasattr(self, 'net_worth_history') and len(self.net_worth_history) > 0:
            initial = self.initial_balance
            final = self.net_worth_history[-1]
            returns = ((final - initial) / initial) * 100
            
            if hasattr(self, 'btc_buy_hold_values') and len(self.btc_buy_hold_values) > 0:
                bh_return = ((self.btc_buy_hold_values[-1] - initial) / initial) * 100
                outperformance = returns - bh_return
                
                performance_text = (f"Strategy Return: {returns:.2f}% | "
                                  f"Buy & Hold: {bh_return:.2f}% | "
                                  f"Alpha: {outperformance:+.2f}% | "
                                  f"Total Trades: {len(self.transaction_history)}")
            else:
                performance_text = f"Strategy Return: {returns:.2f}% | Total Trades: {len(self.transaction_history)}"
                
            fig.text(0.5, 0.02, performance_text, ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='#ecf0f1', alpha=0.8, edgecolor='#bdc3c7'))
        
        plt.tight_layout(rect=[0, 0.04, 1, 0.94])
        
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        if interactive: 
            plt.show()
        else: 
            plt.close(fig)


def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown percentage"""
    if not portfolio_values or len(portfolio_values) < 2:
        return 0.0
        
    peak = portfolio_values[0]
    max_drawdown = 0.0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
            
        if peak > 0:
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
    return max_drawdown * 100

def plot_lstm_predictions(predictions_df, actual_prices_series, lstm_metrics=None, 
                         save_path='./results/lstm_predictions.png', interactive=False):
    """Plot LSTM predictions vs actual prices - simplified version"""
    if predictions_df.empty or actual_prices_series.empty:
        return
        
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Single plot instead of subplots
    fig, ax = plt.subplots(1, 1, figsize=(15, 8), dpi=100)
    
    # Price comparison plot
    ax.plot(actual_prices_series.index, actual_prices_series.values, 
           color='#2c3e50', label='Actual Bitcoin Price', linewidth=3, alpha=0.8)
    
    # Calculate predicted prices
    predicted_prices = [actual_prices_series.iloc[0]] 
    for i in range(len(predictions_df['Predicted_Pct_Change'])-1): 
        prev_actual_price = actual_prices_series.iloc[i] 
        next_predicted_price = prev_actual_price * (1 + predictions_df['Predicted_Pct_Change'].iloc[i])
        predicted_prices.append(next_predicted_price)
        
    plot_dates_pred = predictions_df.index[:len(predicted_prices)]
    
    ax.plot(plot_dates_pred, predicted_prices, color='#e74c3c', linestyle='--', 
           label='LSTM Predicted Price', linewidth=3, alpha=0.8)
    
    ax.fill_between(actual_prices_series.index, 0, actual_prices_series.values, 
                    alpha=0.1, color='#2c3e50')
    
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=16)
    ax.set_title('Bitcoin Price Prediction - LSTM Model Performance', fontsize=20, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    
    # Format tanggal di bawah dengan ukuran font yang lebih besar
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    
    # Add metrics text (optional)
    if lstm_metrics and 'rmse_pct_change' in lstm_metrics and 'mae_pct_change' in lstm_metrics:
        rmse = lstm_metrics['rmse_pct_change'] 
        mae = lstm_metrics['mae_pct_change']
        metric_text = f'Model Performance: RMSE = {rmse:.4f}, MAE = {mae:.4f}'
        
        fig.text(0.5, 0.02, metric_text, ha='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#ecf0f1', alpha=0.9, edgecolor='#bdc3c7'))
    
    plt.tight_layout()
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if interactive: 
        plt.show()
    else: 
        plt.close(fig)

def plot_black_litterman_results(bl_results, save_path='./results/black_litterman_analysis.png', interactive=False):
    """Plot comprehensive Black-Litterman analysis"""
    if bl_results is None:
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig = plt.figure(figsize=(18, 12), dpi=100)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.2], width_ratios=[1, 1, 1], 
                  hspace=0.35, wspace=0.25)
    
    # Expected returns comparison
    ax1 = fig.add_subplot(gs[0, 0])
    if bl_results.get('prior_returns') is not None and bl_results.get('posterior_returns') is not None:
        prior_return = bl_results['prior_returns'][0] * 365 * 100
        posterior_return = bl_results['posterior_returns'][0] * 365 * 100
        
        categories = ['Market Prior', 'BL Posterior']
        values = [prior_return, posterior_return]
        colors = ['#3498db', '#e74c3c']
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.annotate(f'{val:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax1.set_title('Expected Annual Returns', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Expected Return (%)', fontsize=12)
        ax1.grid(axis='y', linestyle=':', alpha=0.6)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=1)
    
    # Volatility comparison
    ax2 = fig.add_subplot(gs[0, 1])
    if bl_results.get('prior_volatility_annualized') is not None and bl_results.get('posterior_volatility_annualized') is not None:
        prior_vol = bl_results['prior_volatility_annualized'] * 100
        posterior_vol = bl_results['posterior_volatility_annualized'] * 100
        
        categories = ['Prior Risk', 'Posterior Risk']
        values = [prior_vol, posterior_vol]
        colors = ['#9b59b6', '#2ecc71']
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.annotate(f'{val:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2.set_title('Annual Volatility', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Volatility (%)', fontsize=12)
        ax2.grid(axis='y', linestyle=':', alpha=0.6)
    
    # LSTM views integration
    ax3 = fig.add_subplot(gs[0, 2])
    views_summary = bl_results.get('views_summary', {})
    
    if views_summary.get('has_views', False):
        confidence = views_summary.get('confidence_level', 0) * 100
        view_impact = views_summary.get('view_impact', 0) * 100
        
        metrics = ['Confidence\n(%)', 'View Impact\n(%)']
        values = [confidence, view_impact]
        colors = ['#27ae60', '#e67e22']
        
        bars = ax3.bar(metrics, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax3.set_title('LSTM Views Integration', fontweight='bold', fontsize=14)
        ax3.set_ylabel('Value', fontsize=12)
        ax3.grid(axis='y', linestyle=':', alpha=0.6)
        plt.setp(ax3.get_xticklabels(), fontsize=9)
    else:
        ax3.text(0.5, 0.5, "No LSTM Views\nIncorporated", ha='center', va='center',
                transform=ax3.transAxes, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.8))
        ax3.set_title('Views Status', fontweight='bold', fontsize=14)
    
    # Black-Litterman adjustments
    ax4 = fig.add_subplot(gs[1, 0])
    return_adj = bl_results.get('return_adjustment', 0) * 365 * 100
    risk_adj = bl_results.get('risk_adjustment', 0) * 100
    
    adjustments = ['Return\nAdjustment', 'Risk\nAdjustment']
    adj_values = [return_adj, risk_adj]
    adj_colors = ['#3498db' if val >= 0 else '#e74c3c' for val in adj_values]
    
    bars = ax4.bar(adjustments, adj_values, color=adj_colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, adj_values):
        height = bar.get_height()
        ax4.annotate(f'{val:+.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=11, fontweight='bold')
    
    ax4.set_title('Black-Litterman Adjustments', fontweight='bold', fontsize=14)
    ax4.set_ylabel('Adjustment (%)', fontsize=12)
    ax4.grid(axis='y', linestyle=':', alpha=0.6)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=1)
    
    # Portfolio allocation
    ax5 = fig.add_subplot(gs[1, 1])
    optimal_weights = bl_results.get('optimal_weights', np.array([1.0]))
    market_weights = bl_results.get('market_weights', np.array([1.0]))
    
    weight_categories = ['Market\nWeight', 'Optimal\nWeight']
    weight_values = [market_weights[0] * 100, optimal_weights[0] * 100]
    weight_colors = ['#95a5a6', '#f39c12']
    
    bars = ax5.bar(weight_categories, weight_values, color=weight_colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, weight_values):
        height = bar.get_height()
        ax5.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax5.set_title('Portfolio Allocation', fontweight='bold', fontsize=14)
    ax5.set_ylabel('Weight (%)', fontsize=12)
    ax5.set_ylim(0, 110)
    ax5.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Risk-adjusted returns (Sharpe ratios)
    ax6 = fig.add_subplot(gs[1, 2])
    model_quality = bl_results.get('model_quality', {})
    prior_sharpe = model_quality.get('prior_sharpe', 0)
    posterior_sharpe = model_quality.get('posterior_sharpe', 0)
    
    sharpe_categories = ['Prior\nSharpe', 'Posterior\nSharpe']
    sharpe_values = [prior_sharpe, posterior_sharpe]
    sharpe_colors = ['#8e44ad', '#16a085']
    
    bars = ax6.bar(sharpe_categories, sharpe_values, color=sharpe_colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, sharpe_values):
        height = bar.get_height()
        ax6.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax6.set_title('Risk-Adjusted Returns', fontweight='bold', fontsize=14)
    ax6.set_ylabel('Sharpe Ratio', fontsize=12)
    ax6.grid(axis='y', linestyle=':', alpha=0.6)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=1)
    
    # Comprehensive summary text
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary_text = "BLACK-LITTERMAN MODEL ANALYSIS SUMMARY\n"
    summary_text += "=" * 80 + "\n\n"
    
    if bl_results.get('prior_returns') is not None:
        prior_ret = bl_results['prior_returns'][0] * 365 * 100
        summary_text += f"Market Implied Return (Annual): {prior_ret:.2f}%\n"
    
    if bl_results.get('posterior_returns') is not None:
        posterior_ret = bl_results['posterior_returns'][0] * 365 * 100
        summary_text += f"Black-Litterman Adjusted Return (Annual): {posterior_ret:.2f}%\n"
    
    summary_text += f"Net Return Adjustment: {return_adj:+.2f}% annually\n"
    summary_text += f"Risk Adjustment: {risk_adj:+.2f}%\n\n"
    
    if views_summary.get('has_views', False):
        summary_text += "LSTM MARKET VIEWS:\n"
        summary_text += f"• Confidence in LSTM Predictions: {confidence:.0f}%\n"
        summary_text += f"• View Impact on Final Allocation: {view_impact:.1f}%\n\n"
    else:
        summary_text += "LSTM MARKET VIEWS: No views incorporated\n\n"
    
    summary_text += "MODEL PARAMETERS & RESULTS:\n"
    summary_text += f"• Risk Aversion Parameter: {bl_results.get('risk_aversion', 2.5)}\n"
    summary_text += f"• Prior Volatility (Annual): {bl_results.get('prior_volatility_annualized', 0)*100:.1f}%\n"
    summary_text += f"• Posterior Volatility (Annual): {bl_results.get('posterior_volatility_annualized', 0)*100:.1f}%\n"
    summary_text += f"• Optimal Portfolio Weight: {optimal_weights[0]*100:.1f}%\n\n"
    
    summary_text += "INTERPRETATION & RECOMMENDATIONS:\n"
    if return_adj > 1:
        summary_text += "• LSTM suggests significantly higher returns than market implies\n"
        summary_text += "• Consider increasing Bitcoin allocation\n"
        summary_text += "• Monitor LSTM prediction accuracy closely\n"
    elif return_adj < -1:
        summary_text += "• LSTM suggests lower returns than market expects\n"
        summary_text += "• Consider defensive positioning\n"
        summary_text += "• Review market conditions and model assumptions\n"
    else:
        summary_text += "• LSTM views align reasonably with market expectations\n"
        summary_text += "• Current allocation appears appropriate\n"
        summary_text += "• Continue monitoring for significant view changes\n"
    
    props = dict(boxstyle='round,pad=1', facecolor='#f8f9fa', alpha=0.95, edgecolor='#bdc3c7', linewidth=1.5)
    ax7.text(0.02, 0.98, summary_text, transform=ax7.transAxes, fontsize=10,
           verticalalignment='top', bbox=props, family='monospace')
    
    fig.suptitle('Black-Litterman Portfolio Optimization Analysis', fontsize=20, fontweight='bold', y=0.98)
    
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.99, 0.01, f"Generated: {current_date}", ha='right', fontsize=9, style='italic', alpha=0.7)
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if interactive:
        plt.show()
    else:
        plt.close(fig)


def plot_cvar_analysis(risk_components, confidence_level=0.90, save_path='./results/cvar_analysis.png', interactive=False):
    """Plot comprehensive CVaR analysis"""
    if not risk_components or "error" in risk_components:
        print("Cannot create CVaR plot due to insufficient data")
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig = plt.figure(figsize=(18, 14), dpi=100)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.2], width_ratios=[1, 1, 1], 
                  hspace=0.35, wspace=0.25)
    
    # Risk measures comparison
    ax1 = fig.add_subplot(gs[0, 0])
    traditional_cvar = risk_components.get('traditional_cvar', 0)
    adjusted_cvar = risk_components.get('adjusted_cvar', 0)
    var_value = risk_components.get('var', 0)
    
    categories = ['VaR\n(Value at Risk)', 'Traditional\nCVaR', 'Enhanced\nCVaR']
    values = [var_value, traditional_cvar, adjusted_cvar]
    colors = ['#f39c12', '#e74c3c', '#c0392b']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=10, fontweight='bold')
    
    ax1.set_title(f'Risk Measures ({confidence_level*100:.0f}% Confidence)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Risk Value', fontsize=12)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=1)
    
    # Risk component adjustments
    ax2 = fig.add_subplot(gs[0, 1])
    rsi_adj = risk_components.get('rsi_adjustment', 1.0)
    kc_adj = risk_components.get('kc_adjustment', 1.0)
    bl_adj = risk_components.get('bl_adjustment', 1.0)
    
    adjustments = ['RSI\nAdjustment', 'Keltner\nAdjustment', 'Black-Litterman\nAdjustment']
    adj_values = [(rsi_adj - 1) * 100, (kc_adj - 1) * 100, (bl_adj - 1) * 100]
    adj_colors = ['#3498db' if val >= 0 else '#e74c3c' for val in adj_values]
    
    bars = ax2.bar(adjustments, adj_values, color=adj_colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, adj_values):
        height = bar.get_height()
        ax2.annotate(f'{val:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=10, fontweight='bold')
    
    ax2.set_title('Risk Component Adjustments', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Adjustment (%)', fontsize=12)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=1)
    plt.setp(ax2.get_xticklabels(), rotation=25, ha="right", fontsize=10)
    
    # Component weights pie chart
    ax3 = fig.add_subplot(gs[0, 2])
    weight_dist = risk_components.get('weight_distribution', {})
    
    if weight_dist:
        components = ['RSI', 'Keltner\nChannels', 'Black-\nLitterman', 'Base\nModel']
        weights = [
            weight_dist.get('rsi', 0) * 100,
            weight_dist.get('kc', 0) * 100,
            weight_dist.get('bl', 0) * 100,
            weight_dist.get('base', 0) * 100
        ]
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#95a5a6']
        
        wedges, texts, autotexts = ax3.pie(weights, labels=components, colors=colors, autopct='%1.1f%%', 
                                          startangle=90, shadow=True, explode=(0.02, 0.02, 0.02, 0.02))
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        for text in texts:
            text.set_fontsize(9)
            text.set_fontweight('bold')
        
        ax3.set_title('Component Weights', fontweight='bold', fontsize=14)
    else:
        ax3.text(0.5, 0.5, "No Weight\nDistribution\nAvailable", ha='center', va='center',
                transform=ax3.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.8))
        ax3.set_title('Component Weights', fontweight='bold', fontsize=14)
    
    # Risk contributions by component
    ax4 = fig.add_subplot(gs[1, 0])
    base_contrib = risk_components.get('base_contribution', 0)
    rsi_contrib = risk_components.get('rsi_contribution', 0)
    kc_contrib = risk_components.get('kc_contribution', 0)
    bl_contrib = risk_components.get('bl_contribution', 0)
    
    contributions = ['Base', 'RSI', 'Keltner', 'Black-Litterman']
    contrib_values = [base_contrib, rsi_contrib, kc_contrib, bl_contrib]
    contrib_colors = ['#95a5a6', '#3498db', '#2ecc71', '#9b59b6']
    
    bars = ax4.bar(contributions, contrib_values, color=contrib_colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, contrib_values):
        height = bar.get_height()
        ax4.annotate(f'{val:.5f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=9, fontweight='bold')
    
    ax4.set_title('Risk Contributions by Component', fontweight='bold', fontsize=14)
    ax4.set_ylabel('Contribution Value', fontsize=12)
    ax4.grid(axis='y', linestyle=':', alpha=0.6)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=1)
    plt.setp(ax4.get_xticklabels(), rotation=25, ha="right", fontsize=10)
    
    # Technical indicators status
    ax5 = fig.add_subplot(gs[1, 1])
    rsi_details = risk_components.get('rsi_details', {})
    kc_details = risk_components.get('kc_details', {})
    
    indicator_summary = "TECHNICAL INDICATORS STATUS:\n\n"
    
    if rsi_details:
        rsi_status = rsi_details.get('result_rsi_status', 'N/A')
        indicator_summary += f"RSI Status: {rsi_status}\n"
        
        if rsi_details.get('result_rsi_bullish_div', False):
            indicator_summary += "• Bullish Divergence Detected ✓\n"
        if rsi_details.get('result_rsi_bearish_div', False):
            indicator_summary += "• Bearish Divergence Detected ⚠\n"
        if rsi_details.get('result_rsi_double_bottom', False):
            indicator_summary += "• Double Bottom Pattern ✓\n"
    
    if kc_details:
        kc_trend = kc_details.get('kc_trend', 'N/A')
        kc_retest = kc_details.get('kc_trend_retest', 'N/A')
        kc_strength = kc_details.get('kc_trend_strength', 0)
        
        indicator_summary += f"\nKeltner Channel Analysis:\n"
        indicator_summary += f"• Trend: {kc_trend}\n"
        indicator_summary += f"• Retest Status: {kc_retest}\n"
        indicator_summary += f"• Trend Strength: {kc_strength:.3f}\n"
    
    if not rsi_details and not kc_details:
        indicator_summary = "No detailed indicator\ninformation available"
    
    ax5.text(0.05, 0.95, indicator_summary, transform=ax5.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', alpha=0.9, edgecolor='#bdc3c7'))
    ax5.set_title('Technical Analysis Details', fontweight='bold', fontsize=14)
    ax5.axis('off')
    
    # Black-Litterman integration details
    ax6 = fig.add_subplot(gs[1, 2])
    bl_details = risk_components.get('bl_details', {})
    
    bl_summary = "BLACK-LITTERMAN INTEGRATION:\n\n"
    
    if bl_details:
        return_adj = bl_details.get('return_adjustment', 0) * 100
        risk_adj = bl_details.get('risk_adjustment', 0) * 100
        views_summary = bl_details.get('views_summary', {})
        
        bl_summary += f"Return Adjustment: {return_adj:+.3f}%\n"
        bl_summary += f"Risk Adjustment: {risk_adj:+.3f}%\n\n"
        
        if views_summary.get('has_views', False):
            bl_summary += "LSTM Views Status: ACTIVE\n"
            confidence = views_summary.get('confidence_level', 0) * 100
            bl_summary += f"View Confidence: {confidence:.0f}%\n"
            view_impact = views_summary.get('view_impact', 0) * 100
            bl_summary += f"Impact on Risk: {view_impact:.1f}%\n"
        else:
            bl_summary += "LSTM Views Status: INACTIVE\n"
    else:
        bl_summary = "Black-Litterman\nintegration not available"
    
    ax6.text(0.05, 0.95, bl_summary, transform=ax6.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', alpha=0.9, edgecolor='#bdc3c7'))
    ax6.set_title('Black-Litterman Integration', fontweight='bold', fontsize=14)
    ax6.axis('off')
    
    # Comprehensive analysis summary
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary_text = "CONDITIONAL VALUE AT RISK (CVaR) COMPREHENSIVE ANALYSIS\n"
    summary_text += "=" * 90 + "\n\n"
    
    summary_text += f"RISK MEASUREMENT FRAMEWORK (Confidence Level: {confidence_level*100:.0f}%):\n"
    summary_text += f"• Value at Risk (VaR): {var_value:.6f}\n"
    summary_text += f"• Traditional CVaR: {traditional_cvar:.6f}\n"
    summary_text += f"• Enhanced CVaR (Multi-Factor): {adjusted_cvar:.6f}\n\n"
    
    # Risk change calculation and interpretation
    summary_text += f"RISK ADJUSTMENT ANALYSIS:\n"
    
    if traditional_cvar != 0:
        # For CVaR: more negative = worse risk
        absolute_risk_change = abs(adjusted_cvar) - abs(traditional_cvar)
        relative_risk_change = (absolute_risk_change / abs(traditional_cvar)) * 100
        
        summary_text += f"• Absolute Risk Change: {absolute_risk_change:+.6f}\n"
        summary_text += f"• Relative Risk Change: {relative_risk_change:+.2f}%\n"
    else:
        relative_risk_change = 0
        summary_text += f"• Risk Change: Cannot calculate (traditional CVaR is zero)\n"
    
    summary_text += f"• RSI Component Adjustment: {(rsi_adj - 1)*100:+.1f}%\n"
    summary_text += f"• Keltner Channel Adjustment: {(kc_adj - 1)*100:+.1f}%\n"
    summary_text += f"• Black-Litterman Adjustment: {(bl_adj - 1)*100:+.1f}%\n\n"
    
    summary_text += f"COMPONENT WEIGHT DISTRIBUTION:\n"
    if weight_dist:
        summary_text += f"• Base Model: {weight_dist.get('base', 0)*100:.1f}%\n"
        summary_text += f"• RSI Indicator: {weight_dist.get('rsi', 0)*100:.1f}%\n"
        summary_text += f"• Keltner Channels: {weight_dist.get('kc', 0)*100:.1f}%\n"
        summary_text += f"• Black-Litterman: {weight_dist.get('bl', 0)*100:.1f}%\n\n"
    
    # Market regime information
    market_regime = risk_components.get('market_regime', 'normal')
    summary_text += f"MARKET REGIME ANALYSIS:\n"
    summary_text += f"• Current Regime: {market_regime.replace('_', ' ').title()}\n"
    
    # Reliability scores
    reliability_scores = risk_components.get('reliability_scores', {})
    if reliability_scores:
        summary_text += f"• RSI Reliability Score: {reliability_scores.get('rsi', 0):.2f}\n"
        summary_text += f"• Keltner Channel Reliability: {reliability_scores.get('kc', 0):.2f}\n"
        summary_text += f"• Black-Litterman Reliability: {reliability_scores.get('bl', 0):.2f}\n\n"
    
    # Risk interpretation logic
    summary_text += f"RISK INTERPRETATION & RECOMMENDATIONS:\n"
    
    if traditional_cvar < 0 and adjusted_cvar < 0:
        # More negative = worse risk
        if abs(adjusted_cvar) > abs(traditional_cvar) * 1.1:  # 10% worse
            summary_text += "• SIGNIFICANTLY HIGHER RISK ENVIRONMENT:\n"
            summary_text += "  - Enhanced CVaR indicates deteriorated risk conditions\n"
            summary_text += "  - Consider reducing position sizes\n"
            summary_text += "  - Implement tighter stop-losses\n"
            summary_text += "  - Increase portfolio hedging\n"
            summary_text += "  - Monitor market conditions closely\n"
        elif abs(adjusted_cvar) < abs(traditional_cvar) * 0.9:  # 10% better
            summary_text += "• LOWER RISK ENVIRONMENT DETECTED:\n"
            summary_text += "  - Multi-factor model suggests improved risk conditions\n"
            summary_text += "  - Opportunity for increased allocation\n"
            summary_text += "  - Consider leveraging favorable conditions\n"
            summary_text += "  - Maintain risk management discipline\n"
            summary_text += "  - Prepare for potential regime change\n"
        else:
            summary_text += "• STABLE RISK ENVIRONMENT:\n"
            summary_text += "  - Enhanced and traditional CVaR show similar risk levels\n"
            summary_text += "  - Current risk levels appear normal\n"
            summary_text += "  - Maintain existing strategy\n"
            summary_text += "  - Continue regular monitoring\n"
            summary_text += "  - Stay alert for significant changes\n"
    else:
        summary_text += "• RISK ASSESSMENT INCONCLUSIVE:\n"
        summary_text += "  - Unusual CVaR values detected\n"
        summary_text += "  - Recommend manual review of risk calculations\n"
        summary_text += "  - Monitor data quality and model inputs\n"
        summary_text += "  - Consider recalibrating risk parameters\n"
    
    # Detailed risk analysis
    summary_text += f"\nDETAILED RISK ANALYSIS:\n"
    if traditional_cvar < 0 and adjusted_cvar < 0:
        risk_severity_traditional = "High" if abs(traditional_cvar) > 0.05 else "Moderate" if abs(traditional_cvar) > 0.02 else "Low"
        risk_severity_enhanced = "High" if abs(adjusted_cvar) > 0.05 else "Moderate" if abs(adjusted_cvar) > 0.02 else "Low"
        
        summary_text += f"• Traditional CVaR Risk Level: {risk_severity_traditional} ({abs(traditional_cvar):.4f})\n"
        summary_text += f"• Enhanced CVaR Risk Level: {risk_severity_enhanced} ({abs(adjusted_cvar):.4f})\n"
        
        if abs(adjusted_cvar) > abs(traditional_cvar):
            summary_text += f"• Multi-factor analysis reveals {((abs(adjusted_cvar)/abs(traditional_cvar))-1)*100:.1f}% higher risk than traditional measure\n"
        else:
            summary_text += f"• Multi-factor analysis suggests {((abs(traditional_cvar)/abs(adjusted_cvar))-1)*100:.1f}% lower risk than traditional measure\n"
    
    summary_text += f"\n" + "=" * 90 + "\n"
    summary_text += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary_text += "Multi-Factor CVaR Model: RSI + Keltner Channels + Black-Litterman Integration\n"
    summary_text += "Note: For CVaR, more negative values indicate higher tail risk"
    
    props = dict(boxstyle='round,pad=1', facecolor='#f8f9fa', alpha=0.95, edgecolor='#bdc3c7', linewidth=1.5)
    ax7.text(0.02, 0.98, summary_text, transform=ax7.transAxes, fontsize=10,
           verticalalignment='top', bbox=props, family='monospace')
    
    fig.suptitle('Enhanced Conditional Value at Risk (CVaR) Analysis\nMulti-Factor Risk Assessment Framework', 
                fontsize=20, fontweight='bold', y=0.98)
    
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.99, 0.01, f"Generated: {current_date}", ha='right', fontsize=9, style='italic', alpha=0.7)
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if interactive:
        plt.show()
    else:
        plt.close(fig)

