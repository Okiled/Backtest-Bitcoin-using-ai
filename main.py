import numpy as np
import pandas as pd
import os
import torch
import random
import argparse
import matplotlib
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import time
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

from project_components import (
    DataProcessor,
    LSTMModel,
    BlackLitterman,
    CVaR,
    BitcoinTradingEnv,
    calculate_max_drawdown,
    plot_lstm_predictions,
    plot_black_litterman_results,
    plot_integrated_results,
    plot_cvar_analysis
)

warnings.filterwarnings('ignore')

def create_bitcoin_csv_for_model(output_dir=None):
    """Create Bitcoin CSV data for model storage and future use"""
    try:
        import yfinance as yf
        
        print("\n🔄 Creating Bitcoin CSV for model storage...")
        
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "model")
        
        csv_filename = f"bitcoin_ohlcv_data_{datetime.now().strftime('%Y%m%d')}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        os.makedirs(output_dir, exist_ok=True)
        
        data_processor = DataProcessor()
        
        df = data_processor.fetch_yahoo_data(
            symbol='BTC-USD',
            interval='1d',
            start_date_str='2019-01-01',
            end_date_str='2025-01-01'
        )
        
        if df is None or df.empty:
            print("❌ Failed to fetch Bitcoin data for CSV!")
            return None
        
        df_csv = df.reset_index()
        df_csv.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        df_csv['close_price'] = df_csv['close']
        
        df_csv = df_csv[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_price']]
        
        df_csv.to_csv(csv_path, index=False)
        
        print(f"✅ Bitcoin CSV created successfully!")
        print(f"📁 Location: {csv_path}")
        print(f"📊 Records: {len(df_csv):,} rows")
        print(f"📅 Period: {df_csv['timestamp'].min()} to {df_csv['timestamp'].max()}")
        print(f"💰 Price range: ${df_csv['close_price'].min():,.2f} - ${df_csv['close_price'].max():,.2f}")
        print(f"💾 File size: {os.path.getsize(csv_path) / 1024:.1f} KB")
        
        return csv_path
        
    except Exception as e:
        print(f"❌ Error creating Bitcoin CSV: {str(e)}")
        return None

def plot_ppo_results(test_data_index, portfolio_values, btc_prices_at_step, actions_taken,
                     performance_metrics, btc_buy_hold_values=None, save_path='./results/ppo_results.png', interactive=False):
    """Plot PPO trading results"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 10), dpi=100)
    
    # Create grid layout
    gs = plt.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # Main performance comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    if len(portfolio_values) > 1 and btc_buy_hold_values and len(btc_buy_hold_values) > 1:
        time_steps = range(len(portfolio_values))
        
        ax1.plot(time_steps, portfolio_values, color='#2c3e50', linewidth=3, 
                label='PPO Strategy Portfolio', alpha=0.9)
        ax1.plot(time_steps, btc_buy_hold_values, color='#f39c12', linewidth=3, 
                label='Buy & Hold Benchmark', alpha=0.8, linestyle='--')
        
        ax1.fill_between(time_steps, portfolio_values, alpha=0.2, color='#2c3e50')
        ax1.fill_between(time_steps, btc_buy_hold_values, alpha=0.1, color='#f39c12')
        
        ax1.set_title('Portfolio Performance: PPO Strategy vs Buy & Hold', fontweight='bold', fontsize=16)
        ax1.set_ylabel('Portfolio Value (USD)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    
    # Performance metrics bar chart
    ax2 = fig.add_subplot(gs[1, 0])
    
    total_return = performance_metrics.get('total_return', 0)
    buy_hold_return = performance_metrics.get('buy_hold_return', 0)
    outperformance = performance_metrics.get('outperformance', 0)
    sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
    max_drawdown = performance_metrics.get('max_drawdown', 0)
    
    metrics_names = ['Strategy\nReturn (%)', 'Buy & Hold\nReturn (%)', 'Alpha\n(%)', 'Sharpe\nRatio', 'Max DD\n(%)']
    metrics_values = [total_return, buy_hold_return, outperformance, sharpe_ratio, max_drawdown]
    
    colors = ['#3498db', '#f39c12', '#2ecc71' if outperformance > 0 else '#e74c3c', '#16a085', '#e74c3c']
    
    bars = ax2.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 * max(abs(max(metrics_values)), abs(min(metrics_values)))), 
               f"{val:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_title('Performance Metrics', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=1)
    
    # Trading statistics
    ax3 = fig.add_subplot(gs[1, 1])
    
    buy_count = performance_metrics.get('buy_count', 0)
    sell_count = performance_metrics.get('sell_count', 0)
    total_trades = performance_metrics.get('num_trades', 0)
    
    trade_labels = ['Buy\nOrders', 'Sell\nOrders']
    trade_counts = [buy_count, sell_count]
    trade_colors = ['#27ae60', '#e74c3c']
    
    if sum(trade_counts) > 0:
        wedges, texts, autotexts = ax3.pie(trade_counts, labels=trade_labels, colors=trade_colors, 
                                          autopct='%1.0f', startangle=90, shadow=True)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    ax3.set_title(f'Trading Activity\n({total_trades} total trades)', fontweight='bold', fontsize=12)
    
    # Add performance summary
    if len(test_data_index) > 0:
        start_date = test_data_index[0].strftime("%Y-%m-%d") if hasattr(test_data_index[0], 'strftime') else "Start"
        end_date = test_data_index[-1].strftime("%Y-%m-%d") if hasattr(test_data_index[-1], 'strftime') else "End"
        date_range = f"Backtest Period: {start_date} to {end_date}"
        
        performance_text = (f"{date_range} | "
                          f"Strategy: {total_return:.2f}% | "
                          f"Buy & Hold: {buy_hold_return:.2f}% | "
                          f"Alpha: {outperformance:+.2f}%")
        
        fig.text(0.5, 0.02, performance_text, ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#ecf0f1', alpha=0.8, edgecolor='#bdc3c7'))
    
    plt.suptitle('PPO Reinforcement Learning Trading Strategy Results', 
                fontsize=18, fontweight='bold', y=0.95)
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if interactive: 
        plt.show()
    else: 
        plt.close(fig)

class BitcoinPortfolioOptimizer:
    """
    Complete Bitcoin Portfolio Optimization System for Spot Trading
    Pipeline: Data Processing → LSTM Prediction → Black-Litterman → Enhanced CVaR → PPO Trading
    """
    
    def __init__(self, base_output_dir, interactive=False, use_gpu=False, 
                 min_transaction_pct=0.1, max_hold_duration=20):
        self.base_output_dir = os.path.abspath(base_output_dir)
        print(f"Initializing Bitcoin Portfolio Optimizer for Spot Trading")
        print(f"All outputs will be saved to: {self.base_output_dir}")

        # Create directory structure
        self.model_dir = os.path.join(self.base_output_dir, 'model')
        self.results_dir = os.path.join(self.base_output_dir, 'results')
        self.data_predictions_dir = os.path.join(self.base_output_dir, 'data', 'predictions')
        self.logs_dir = os.path.join(self.base_output_dir, 'logs', 'ppo_tensorboard')
        
        # Check existing outputs
        self._check_existing_outputs()

        # Initialize components
        self.data_processor = DataProcessor()
        self.pytorch_lstm_model = None
        self.bl_model = None
        self.cvar_model = None
        self.ppo_model = None
        self.processed_data = None
        self.lstm_predictions = None
        self.bl_results = None
        
        # Data configuration
        self.common_data = None
        self.start_date = '2019-01-01'
        self.end_date = '2025-01-01'

        # Model paths
        self.lstm_model_save_path = os.path.join(self.model_dir, 'lstm_model.pth')
        self.ppo_model_save_path = os.path.join(self.model_dir, 'ppo_model.zip')
        
        # Configuration
        self.interactive = interactive
        self.min_transaction_pct = min_transaction_pct
        self.max_hold_duration = max_hold_duration
        
        # Device configuration
        if use_gpu and torch.cuda.is_available(): 
            self.device = torch.device("cuda")
            print(f"PyTorch using GPU: {torch.cuda.get_device_name(0)}")
        else: 
            self.device = torch.device("cpu")
            print("PyTorch using CPU")
        
        if not interactive: 
            matplotlib.use('Agg')
    
    def _check_existing_outputs(self):
        """Check for existing output files"""
        print(f"\n🔍 Checking existing outputs...")
        print("=" * 50)
        
        expected_files = {
            'data/predictions/lstm_predicted_pct_changes.csv': 'LSTM Predictions CSV',
            'data/predictions/black_litterman_results.csv': 'Black-Litterman Results CSV', 
            'model/lstm_model.pth': 'LSTM Model',
            'model/ppo_model.zip': 'PPO Model',
            'results/lstm_predictions.png': 'LSTM Predictions Plot',
            'results/black_litterman_analysis.png': 'Black-Litterman Plot',
            'results/cvar_analysis.png': 'CVaR Analysis Plot',
            'results/ppo_results.png': 'PPO Results Plot',
            'results/integrated_strategy_results.png': 'Integrated Results Plot',
            'results/wallet_composition.png': 'Wallet Composition Plot',
            'results/performance_metrics.json': 'Performance Metrics JSON',
            'results/transaction_history.csv': 'Transaction History CSV'
        }
        
        self.existing_files = {}
        self.missing_files = {}
        
        # Check for Bitcoin CSV
        model_dir = self.model_dir
        bitcoin_csv_exists = False
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                if filename.startswith('bitcoin_ohlcv_data') and filename.endswith('.csv'):
                    csv_path = os.path.join(model_dir, filename)
                    if os.path.exists(csv_path):
                        size = os.path.getsize(csv_path) / 1024
                        print(f"✅ Bitcoin OHLCV Data: {size:.1f} KB")
                        bitcoin_csv_exists = True
                        break
        
        if not bitcoin_csv_exists:
            print(f"❌ Bitcoin OHLCV Data: Missing")
            self.missing_files['bitcoin_csv'] = True
        else:
            self.existing_files['bitcoin_csv'] = True
        
        # Check other expected files
        for rel_path, description in expected_files.items():
            full_path = os.path.join(self.base_output_dir, rel_path)
            if os.path.exists(full_path):
                size = os.path.getsize(full_path) / 1024
                print(f"✅ {description}: {size:.1f} KB")
                self.existing_files[rel_path] = full_path
            else:
                print(f"❌ {description}: Missing")
                self.missing_files[rel_path] = full_path
        
        existing_count = len(self.existing_files)
        missing_count = len(self.missing_files)
        print(f"\n📊 Summary: {existing_count} files exist, {missing_count} missing")
        print("=" * 50)

    def _load_or_create_dir(self, dir_path):
        """Create directory if it doesn't exist"""
        if not os.path.exists(dir_path): 
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

    def fetch_yahoo_data(self, start_date=None, end_date=None):
        """Step 1: Fetch Bitcoin data from Yahoo Finance"""
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
            
        print(f"\nStep 1: Fetching Yahoo Finance Data from {start_date} to {end_date}...")
        self._load_or_create_dir(self.data_predictions_dir)
        
        self.start_date = start_date
        self.end_date = end_date
        
        raw_data = self.data_processor.fetch_yahoo_data(
            symbol='BTC-USD',
            interval='1d',
            start_date_str=start_date, 
            end_date_str=end_date
        )
        
        if raw_data is None: 
            raise RuntimeError("Failed to fetch data from Yahoo Finance.")
        
        self.common_data = raw_data.copy()
        
        print(f"✅ Data fetched successfully: {raw_data.shape}")
        print(f"📅 Date range: {raw_data.index.min()} to {raw_data.index.max()}")
        print(f"💰 Price range: ${raw_data['close'].min():,.2f} - ${raw_data['close'].max():,.2f}")
        
        return raw_data

    def preprocess_data(self, raw_data=None, lstm_predictions_for_ppo=None):
        """Step 2: Preprocess data with technical indicators (RSI + Keltner Channels only)"""
        print("\nStep 2: Preprocessing Data with Technical Indicators...")
        
        if raw_data is None:
            if self.common_data is None:
                self.common_data = self.fetch_yahoo_data()
            raw_data = self.common_data.copy()
        
        self.processed_data = self.data_processor.preprocess_data(
            raw_data, 
            lstm_predictions_pct_change=lstm_predictions_for_ppo
        )
        
        if self.processed_data.empty: 
            raise RuntimeError("Data preprocessing resulted in an empty DataFrame.")
        
        print(f"✅ Data preprocessed successfully: {self.processed_data.shape}")
        print(f"📊 Technical indicators calculated: RSI, Keltner Channels")
        print(f"📅 Processed data range: {self.processed_data.index.min()} to {self.processed_data.index.max()}")
        
        return self.processed_data
    
    def feature_engineering(self):
        """Step 3: Feature engineering and validation"""
        print("\nStep 3: Performing Feature Engineering...")
        
        # Essential features for spot trading (no EMA)
        essential_features = [
            'open', 'high', 'low', 'close', 'volume', 'daily_return', 'target_pct_change',
            'result_rsi', 'kc_upper', 'kc_middle', 'kc_lower'
        ]
        
        missing_features = [feat for feat in essential_features if feat not in self.processed_data.columns]
        
        if missing_features:
            print(f"⚠️ Warning: Missing essential features: {missing_features}")
        else:
            print("✅ All essential features verified and available for model training.")
        
        # Handle NaN values
        nan_counts = self.processed_data.isna().sum()
        if nan_counts.sum() > 0:
            print(f"🔧 Handling {nan_counts.sum()} NaN values...")
            self.processed_data.fillna(method='ffill', inplace=True)
            self.processed_data.fillna(method='bfill', inplace=True)
            print("✅ NaN values resolved using forward and backward filling.")
            
        return self.processed_data

    def check_technical_indicators(self):
        """Step 4: Verify technical indicators (RSI & Keltner Channels only)"""
        print("\nStep 4: Verifying Technical Indicators...")
        
        indicators = self.data_processor.get_all_indicators()
        
        # Check RSI
        if 'rsi' in indicators and len(indicators['rsi']) > 0:
            rsi_df = indicators['rsi']
            print(f"✅ RSI indicator: {len(rsi_df)} data points")
            print(f"📊 RSI columns: {list(rsi_df.columns)}")
            
            if len(rsi_df) > 100:
                sample_date = rsi_df.index[100]
                rsi_value = rsi_df.loc[sample_date, 'result_rsi'] if 'result_rsi' in rsi_df.columns else 'N/A'
                rsi_status = rsi_df.loc[sample_date, 'result_rsi_status'] if 'result_rsi_status' in rsi_df.columns else 'N/A'
                print(f"📈 Sample RSI at {sample_date}: {rsi_value} ({rsi_status})")
        else:
            print("⚠️ Warning: RSI indicator not found or empty.")
            
        # Check Keltner Channels
        if 'keltner_channels' in indicators and len(indicators['keltner_channels']) > 0:
            kc_df = indicators['keltner_channels']
            print(f"✅ Keltner Channels: {len(kc_df)} data points")
            print(f"📊 KC columns: {list(kc_df.columns)}")
            
            if len(kc_df) > 100:
                sample_date = kc_df.index[100]
                kc_trend = kc_df.loc[sample_date, 'kc_trend'] if 'kc_trend' in kc_df.columns else 'N/A'
                print(f"📈 Sample KC trend at {sample_date}: {kc_trend}")
        else:
            print("⚠️ Warning: Keltner Channel indicator not found or empty.")
            
        return indicators

    def train_or_load_lstm_model(self, look_back=60, train_ratio=0.8, force_retrain=False, 
                            learning_rate=0.0005, validation_split=0.2, batch_size=32, 
                            epochs=100, patience=15, hidden_size=100, num_layers=2, dropout_rate=0.3):
        """Step 5: Train or load LSTM model for Bitcoin price prediction"""
        print("\nStep 5: Training LSTM Model for Bitcoin Price Prediction...")
        
        if not force_retrain and self.lstm_model_save_path in self.existing_files:
            print(f"✅ LSTM model exists at {self.lstm_model_save_path}")
            print("🔄 Loading existing model...")
        
        print(f"📋 LSTM Parameters:")
        print(f"   - Look back: {look_back} days")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Max epochs: {epochs}")
        print(f"   - Patience: {patience}")
        print(f"   - Hidden size: {hidden_size}")
        print(f"   - Layers: {num_layers}")
        print(f"   - Dropout: {dropout_rate}")
        
        self._load_or_create_dir(self.model_dir)
        
        # Prepare LSTM data
        X_np, y_np_scaled_pct_change = self.data_processor.prepare_lstm_data(
            self.processed_data, 
            look_back=look_back
        )

        if X_np.size == 0 or y_np_scaled_pct_change.size == 0:
            print("❌ Insufficient data for LSTM sequences. Skipping LSTM.")
            self.lstm_predicted_unscaled_pct_changes_for_ppo = None
            return None, pd.DataFrame(), {'rmse_pct_change': float('nan'), 'mae_pct_change': float('nan')}

        input_size = X_np.shape[2]
        print(f"📊 LSTM input shape: {X_np.shape} (samples, time steps, features)")
        print(f"📊 LSTM target shape: {y_np_scaled_pct_change.shape}")
        
        # Initialize LSTM model
        self.pytorch_lstm_model = LSTMModel(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            output_size=1, 
            dropout_rate=dropout_rate
        )

        # Load existing model or train new one
        if not force_retrain and os.path.exists(self.lstm_model_save_path):
            print(f"🔄 Loading pre-trained LSTM model...")
            try:
                self.pytorch_lstm_model.load_state_dict(torch.load(self.lstm_model_save_path, map_location=self.device))
                self.pytorch_lstm_model.to(self.device)
                print("✅ LSTM model loaded successfully.")
            except Exception as e: 
                print(f"❌ Error loading LSTM model: {e}. Will retrain...")
                force_retrain = True
        
        # Train new model if needed
        if force_retrain or not os.path.exists(self.lstm_model_save_path):
            print(f"🔄 Training new LSTM model...")
            lstm_train_split_idx = int(len(X_np) * train_ratio)
            X_lstm_train_np = X_np[:lstm_train_split_idx]
            y_lstm_train_np_scaled_pct_change = y_np_scaled_pct_change[:lstm_train_split_idx]

            if X_lstm_train_np.size == 0 or y_lstm_train_np_scaled_pct_change.size == 0:
                print("❌ Insufficient training data. Skipping training.")
                self.lstm_predicted_unscaled_pct_changes_for_ppo = None
                return None, pd.DataFrame(), {'rmse_pct_change': float('nan'), 'mae_pct_change': float('nan')}

            self.pytorch_lstm_model = self.pytorch_lstm_model.train_pytorch_lstm(
                X_lstm_train_np, 
                y_lstm_train_np_scaled_pct_change, 
                self.device,
                epochs=epochs, 
                batch_size=batch_size, 
                learning_rate=learning_rate, 
                validation_split=validation_split, 
                model_save_path=self.lstm_model_save_path, 
                patience=patience
            )

            print(f"✅ LSTM model training complete and saved to {self.lstm_model_save_path}")

        # Evaluate model
        lstm_eval_split_idx = int(len(X_np) * train_ratio)
        X_lstm_eval_np = X_np[lstm_eval_split_idx:]
        y_lstm_eval_np_scaled_pct_change = y_np_scaled_pct_change[lstm_eval_split_idx:]
        
        lstm_eval_metrics, _ = self.pytorch_lstm_model.evaluate_pytorch_lstm(
            X_lstm_eval_np, 
            y_lstm_eval_np_scaled_pct_change, 
            self.device, 
            self.data_processor.target_scaler, 
            batch_size=batch_size
        )
        
        print(f"📊 LSTM Evaluation Metrics:")
        print(f"   - RMSE (% change): {lstm_eval_metrics.get('rmse_pct_change', float('nan')):.4f}")
        print(f"   - MAE (% change): {lstm_eval_metrics.get('mae_pct_change', float('nan')):.4f}")

        # Generate predictions
        all_pred_scaled_pct_change = self.pytorch_lstm_model.predict_pytorch_lstm(
            X_np, 
            self.device, 
            batch_size=batch_size
        )
        
        if all_pred_scaled_pct_change.size == 0:
            self.lstm_predicted_unscaled_pct_changes_for_ppo = None
            return self.pytorch_lstm_model, pd.DataFrame(), lstm_eval_metrics

        self.lstm_predicted_unscaled_pct_changes_for_ppo = self.data_processor.target_scaler.inverse_transform(
            all_pred_scaled_pct_change
        )
        
        # Create predictions DataFrame
        predictions_df_index = self.processed_data.index[look_back:look_back + len(self.lstm_predicted_unscaled_pct_changes_for_ppo)]
        actual_prices_series_for_plot = self.processed_data['close_unscaled'].loc[predictions_df_index]
        
        predictions_plot_df = pd.DataFrame({
            'Predicted_Pct_Change': self.lstm_predicted_unscaled_pct_changes_for_ppo.flatten(),
        }, index=predictions_df_index)
        
        self.lstm_predictions = predictions_plot_df.copy()
        
        # Save predictions
        csv_save_path = os.path.join(self.data_predictions_dir, 'lstm_predicted_pct_changes.csv')
        if not os.path.exists(csv_save_path):
            predictions_plot_df.to_csv(csv_save_path)
            print(f"💾 LSTM predictions saved to {csv_save_path}")
        else:
            print(f"✅ LSTM predictions CSV already exists")
        
        # Create plot
        self._load_or_create_dir(self.results_dir)
        lstm_plot_save_path = os.path.join(self.results_dir, 'lstm_predictions.png')
        
        if not os.path.exists(lstm_plot_save_path):
            plot_lstm_predictions(
                predictions_plot_df, 
                actual_prices_series_for_plot,
                lstm_metrics=lstm_eval_metrics,
                save_path=lstm_plot_save_path, 
                interactive=self.interactive
            )
            print(f"📈 LSTM plot saved to {lstm_plot_save_path}")
        else:
            print(f"✅ LSTM plot already exists")

        # Create detailed predictions DataFrame
        try:
            idx_before_first_pred = self.processed_data.index.get_loc(predictions_df_index[0]) - 1
            if idx_before_first_pred < 0: 
                predictions_df = pd.DataFrame()
            else:
                start_price_for_recon = self.processed_data['close_unscaled'].iloc[idx_before_first_pred]
                reconstructed_prices = [start_price_for_recon]
                
                for pct_change in self.lstm_predicted_unscaled_pct_changes_for_ppo.flatten():
                    reconstructed_prices.append(reconstructed_prices[-1] * (1 + pct_change))
                    
                predictions_df = pd.DataFrame({
                    'Actual': actual_prices_series_for_plot.values, 
                    'Predicted': reconstructed_prices[1:]
                }, index=predictions_df_index)
        except Exception as e:
            print(f"⚠️ Warning: Could not create detailed predictions DataFrame: {e}")
            predictions_df = pd.DataFrame()
        
        return self.pytorch_lstm_model, predictions_df, lstm_eval_metrics
    
    def optimize_portfolio_black_litterman(self, risk_aversion=2.5, tau=0.03, confidence_level=0.65, view_impact=0.35):
        """Step 5.5: Black-Litterman portfolio optimization for spot trading"""
        print(f"\nStep 5.5: Black-Litterman Portfolio Optimization (Spot Trading)...")
        print(f"📋 BL Parameters:")
        print(f"   - Risk aversion: {risk_aversion}")
        print(f"   - Tau: {tau}")
        print(f"   - LSTM confidence: {confidence_level*100:.1f}%")
        print(f"   - View impact: {view_impact*100:.1f}%")
        
        # Get historical returns
        historical_returns_series = self.processed_data['daily_return'].dropna()
        if historical_returns_series.empty: 
            print("❌ No return data available for Black-Litterman optimization.")
            self.bl_results = None
            return None
        
        # Prepare LSTM views
        lstm_views = None
        if self.lstm_predictions is not None and not self.lstm_predictions.empty:
            lstm_raw = self.lstm_predictions['Predicted_Pct_Change']
            
            # Clip to reasonable range for spot trading
            max_daily_return = 0.0035
            min_daily_return = -0.0028
            lstm_clipped = np.clip(lstm_raw, min_daily_return, max_daily_return)
            
            lstm_views = pd.Series(lstm_clipped, index=lstm_raw.index)
            lstm_views = lstm_views.reindex(historical_returns_series.index).fillna(0)
            
            print(f"📊 LSTM views prepared: {len(lstm_views)} data points")
            print(f"📈 LSTM view range: {lstm_views.min():.4f} to {lstm_views.max():.4f}")
        else:
            print("⚠️ No LSTM predictions available for Black-Litterman views.")
        
        try:
            # Initialize BlackLitterman model (spot trading version - no leverage)
            self.bl_model = BlackLitterman(
                returns_series=historical_returns_series,
                lstm_predictions=lstm_views,
                risk_aversion=risk_aversion,
                tau=tau,
                confidence_level=confidence_level,
                view_impact=view_impact
            )
            
            self.bl_results = self.bl_model.get_bl_results()
            
            print(f"✅ Black-Litterman optimization completed:")
            if self.bl_results['prior_returns'] is not None:
                prior_daily = self.bl_results['prior_returns'][0] * 100
                prior_annual = prior_daily * 365
                print(f"   📈 Prior return: {prior_daily:.4f}% daily ({prior_annual:.2f}% annual)")
                
            if self.bl_results['posterior_returns'] is not None:
                posterior_daily = self.bl_results['posterior_returns'][0] * 100
                posterior_annual = posterior_daily * 365
                print(f"   📈 Posterior return: {posterior_daily:.4f}% daily ({posterior_annual:.2f}% annual)")
                
            print(f"   📊 Return adjustment: {self.bl_results['return_adjustment']*100:+.4f}%")
            print(f"   📊 Risk adjustment: {self.bl_results['risk_adjustment']:+.2f}%")
            print(f"   🎯 Optimal Bitcoin allocation: {self.bl_results['optimal_weights'][0]*100:.1f}% (Spot trading)")
            
            if self.bl_results['views_summary']['has_views']:
                print(f"   🤖 LSTM view influence: {self.bl_results['model_quality']['view_relative_impact']*100:.2f}%")
            
            # Create plot
            self._load_or_create_dir(self.results_dir)
            bl_plot_save_path = os.path.join(self.results_dir, 'black_litterman_analysis.png')
            
            if not os.path.exists(bl_plot_save_path):
                plot_black_litterman_results(
                    self.bl_results, 
                    save_path=bl_plot_save_path, 
                    interactive=self.interactive
                )
                print(f"📈 Black-Litterman plot saved to {bl_plot_save_path}")
            else:
                print(f"✅ Black-Litterman plot already exists")
            
            # Save results
            bl_results_path = os.path.join(self.data_predictions_dir, 'black_litterman_results.csv')
            if not os.path.exists(bl_results_path):
                try:
                    bl_df = pd.DataFrame({
                        'metric': ['prior_return', 'posterior_return', 'return_adjustment', 'risk_adjustment', 
                                  'prior_volatility', 'posterior_volatility', 'optimal_weight', 'view_impact'],
                        'value': [
                            self.bl_results['prior_returns'][0] if self.bl_results['prior_returns'] is not None else 0,
                            self.bl_results['posterior_returns'][0] if self.bl_results['posterior_returns'] is not None else 0,
                            self.bl_results['return_adjustment'],
                            self.bl_results['risk_adjustment'],
                            self.bl_results['prior_volatility_annualized'],
                            self.bl_results['posterior_volatility_annualized'],
                            self.bl_results['optimal_weights'][0],
                            self.bl_results['views_summary'].get('view_impact', 0)
                        ]
                    })
                    bl_df.to_csv(bl_results_path, index=False)
                    print(f"💾 Black-Litterman results saved to {bl_results_path}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not save BL results to CSV: {e}")
            else:
                print(f"✅ Black-Litterman results CSV already exists")
            
            return self.bl_results
            
        except Exception as e:
            print(f"❌ Error in Black-Litterman optimization: {e}")
            import traceback
            traceback.print_exc()
            self.bl_results = None
            return None
    
    def calculate_risk_metrics(self, confidence_level=0.90):
        """Step 6: Calculate Enhanced CVaR risk metrics"""
        print(f"\nStep 6: Enhanced CVaR Risk Assessment (Confidence: {confidence_level*100}%)...")
        
        historical_returns_series = self.processed_data['daily_return'].dropna()
        if historical_returns_series.empty: 
            print("❌ No return data available for risk calculation.")
            return np.nan, {}
        
        # Prepare indicator data
        indicator_data = {}
        
        # RSI data
        if 'result_rsi' in self.processed_data.columns and 'result_rsi_status' in self.processed_data.columns:
            rsi_cols = [col for col in self.processed_data.columns if col.startswith('result_rsi')]
            indicator_data['rsi'] = self.processed_data[rsi_cols]
            print(f"📊 RSI data available: {len(rsi_cols)} metrics")
        else:
            print("⚠️ Warning: RSI columns not found. Using basic CVaR.")
        
        # Keltner Channel data
        kc_columns = ['kc_trend', 'kc_trend_strength', 'kc_trend_retest']
        if all(col in self.processed_data.columns for col in kc_columns):
            kc_cols = [col for col in self.processed_data.columns if col.startswith('kc_')]
            indicator_data['keltner_channels'] = self.processed_data[kc_cols]
            print(f"📊 Keltner Channel data available: {len(kc_cols)} metrics")
        else:
            print("⚠️ Warning: Keltner Channel columns not found. Using basic CVaR.")
        
        try:
            # Initialize Enhanced CVaR model
            self.cvar_model = CVaR(
                historical_returns_series, 
                confidence_level=confidence_level,
                indicator_data=indicator_data,
                rsi_weight=0.3,
                kc_weight=0.3,
                use_enhanced_features=True,
                bl_results=self.bl_results
            )
            
            cvar_value = self.cvar_model.calculate_cvar()
            risk_components = self.cvar_model.get_risk_components()
            
            traditional_cvar = risk_components.get('traditional_cvar', cvar_value)
            
            print(f"📊 Enhanced CVaR Results:")
            print(f"   📉 Traditional CVaR: {traditional_cvar:.6f}")
            print(f"   📉 Enhanced CVaR: {cvar_value:.6f}")
            
            # Risk adjustments
            rsi_adj = risk_components.get('rsi_adjustment', 1.0)
            kc_adj = risk_components.get('kc_adjustment', 1.0)
            bl_adj = risk_components.get('bl_adjustment', 1.0)
            
            print(f"   🔧 RSI adjustment: {(rsi_adj-1)*100:+.1f}%")
            print(f"   🔧 Keltner adjustment: {(kc_adj-1)*100:+.1f}%")
            
            if self.bl_results is not None:
                print(f"   🔧 Black-Litterman adjustment: {(bl_adj-1)*100:+.1f}%")
                weight_dist = risk_components.get('weight_distribution', {})
                print(f"   📊 Component weights: RSI({weight_dist.get('rsi', 0.3)*100:.0f}%), "
                     f"KC({weight_dist.get('kc', 0.3)*100:.0f}%), "
                     f"BL({weight_dist.get('black_litterman', 0.4)*100:.0f}%)")
            
            # Display current market conditions
            if 'rsi_details' in risk_components:
                rsi_details = risk_components['rsi_details']
                print(f"   📈 Current RSI Status: {rsi_details.get('result_rsi_status', 'N/A')}")
            
            if 'kc_details' in risk_components:
                kc_details = risk_components['kc_details']
                print(f"   📈 Current KC Trend: {kc_details.get('kc_trend', 'N/A')}")
            
            # Create plot
            self._load_or_create_dir(self.results_dir)
            cvar_plot_save_path = os.path.join(self.results_dir, 'cvar_analysis.png')
            
            if not os.path.exists(cvar_plot_save_path):
                plot_cvar_analysis(
                    risk_components,
                    confidence_level=confidence_level,
                    save_path=cvar_plot_save_path,
                    interactive=self.interactive
                )
                print(f"📈 CVaR analysis plot saved to {cvar_plot_save_path}")
            else:
                print(f"✅ CVaR analysis plot already exists")
            
            return cvar_value, risk_components
            
        except Exception as e:
            print(f"❌ Error calculating enhanced CVaR: {e}")
            print("🔄 Falling back to basic CVaR calculation...")
            
            var = historical_returns_series.quantile(1 - confidence_level)
            cvar = historical_returns_series[historical_returns_series <= var].mean()
            
            print(f"📊 Basic CVaR ({confidence_level*100}% confidence): {cvar:.6f}")
            print(f"📊 Value at Risk (VaR): {var:.6f}")
            
            return cvar, {"traditional_cvar": cvar, "var": var, "adjusted_cvar": cvar}
    
    def prepare_synchronized_data_for_ppo(self):
        """Prepare synchronized dataset for PPO with LSTM predictions"""
        print("\n🔄 Preparing synchronized dataset for PPO...")
        
        synced_data = self.processed_data.copy()
        
        # Add LSTM predictions
        if self.lstm_predictions is not None and not self.lstm_predictions.empty:
            lstm_preds = self.lstm_predictions.copy()
            
            synced_data['lstm_pred_pct_change'] = lstm_preds.reindex(
                synced_data.index
            )['Predicted_Pct_Change'].fillna(0.0)
            
            print(f"📊 Added LSTM predictions for {len(lstm_preds)} dates")
            print(f"📅 LSTM date range: {lstm_preds.index.min()} to {lstm_preds.index.max()}")
        else:
            print("⚠️ No LSTM predictions available. Using zeros.")
            synced_data['lstm_pred_pct_change'] = 0.0
        
        # Clean data
        numeric_cols = synced_data.select_dtypes(include=np.number).columns
        nan_counts = synced_data[numeric_cols].isna().sum()
        
        if nan_counts.sum() > 0:
            print(f"🔧 Handling NaN values in {nan_counts[nan_counts > 0].index.tolist()}")
            synced_data.fillna(method='ffill', inplace=True)
            synced_data.fillna(method='bfill', inplace=True)
            synced_data.fillna(0, inplace=True)
        
        # Handle infinity values
        inf_mask = np.isinf(synced_data[numeric_cols])
        if inf_mask.any().any():
            print(f"🔧 Replacing infinity values...")
            synced_data.replace([np.inf, -np.inf], [1e9, -1e9], inplace=True)
        
        print(f"✅ Synchronized dataset prepared: {synced_data.shape}")
        
        return synced_data
    
    def train_or_load_ppo_model(self, test_ratio=0.2, force_retrain=False, total_timesteps=300000):
        """Steps 7 & 8: Train PPO model and run backtest"""
        print("\nSteps 7 & 8: PPO Reinforcement Learning Training & Backtesting...")
        
        if not force_retrain and self.ppo_model_save_path in self.existing_files:
            print(f"✅ PPO model exists at {self.ppo_model_save_path}")
            print("🔄 Loading existing model...")
        
        print(f"📋 PPO Configuration:")
        print(f"   - Total timesteps: {total_timesteps:,}")
        print(f"   - Test ratio: {test_ratio*100:.1f}%")
        print(f"   - Min transaction: {self.min_transaction_pct*100:.1f}%")
        print(f"   - Max hold duration: {self.max_hold_duration} days")
        
        self._load_or_create_dir(self.model_dir)
        self._load_or_create_dir(self.logs_dir)

        # Prepare data for PPO
        data_for_ppo_env = self.prepare_synchronized_data_for_ppo()

        if data_for_ppo_env.empty or len(data_for_ppo_env) < 100:
            print("❌ Insufficient data for PPO environment.")
            return None, {} 

        # Validate required columns
        env_check_cols = [
            'open', 'high', 'low', 'close', 'volume', 'result_rsi', 
            'close_unscaled', 'daily_return'
        ]
        
        if 'lstm_pred_pct_change' in data_for_ppo_env.columns: 
            env_check_cols.append('lstm_pred_pct_change')

        existing_cols_for_env = [col for col in env_check_cols if col in data_for_ppo_env.columns]
        if not existing_cols_for_env: 
            print("❌ No required columns found for PPO environment.")
            return None, {}
        
        relevant_data_for_env = data_for_ppo_env[existing_cols_for_env]
        
        # Final data validation
        nan_counts = relevant_data_for_env.isnull().sum()
        numeric_cols_for_inf_check = relevant_data_for_env.select_dtypes(include=np.number).columns
        inf_counts = pd.DataFrame(np.isinf(relevant_data_for_env[numeric_cols_for_inf_check]), 
                                columns=numeric_cols_for_inf_check).sum()
        
        if nan_counts.sum() > 0 or inf_counts.sum() > 0: 
            print("❌ Data contains NaN or inf values. Cannot create PPO environment.")
            return None, {}

        # Split data
        look_back_const = 60
        train_df_ppo = data_for_ppo_env.iloc[:int(len(data_for_ppo_env) * (1 - test_ratio))]
        test_df_ppo = data_for_ppo_env.copy()

        if train_df_ppo.empty or len(train_df_ppo) < (look_back_const + 50):
            print("❌ Insufficient training data for PPO.")
            return None, {}

        # Create training environment
        try:
            train_env = DummyVecEnv([lambda: BitcoinTradingEnv(
                train_df_ppo, 
                maker_fee=0.0005,    # 0.05% maker fee
                taker_fee=0.0015,    # 0.15% taker fee
                slippage_factor=0.0005,  # 0.05% slippage
                profit_reward_weight=3.0,
                cvar_penalty_weight=1.0,
                min_transaction_pct=self.min_transaction_pct,
                max_hold_duration=self.max_hold_duration
            )])
            print("✅ PPO training environment created")
        except ValueError as e:
            print(f"❌ Error creating PPO training environment: {e}")
            return None, {}
        
        ppo_device_str = "cuda" if self.device.type == "cuda" else "cpu"
        
        # Load or train PPO model
        model_loaded_successfully = False
        
        if not force_retrain and os.path.exists(self.ppo_model_save_path):
            print(f"🔄 Loading PPO model from {self.ppo_model_save_path}...")
            try:
                # Check if model file is not empty
                model_size = os.path.getsize(self.ppo_model_save_path)
                if model_size < 1000:  # Less than 1KB indicates empty/corrupted model
                    print(f"⚠️ Model file too small ({model_size} bytes). Will retrain...")
                    force_retrain = True
                else:
                    self.ppo_model = PPO.load(self.ppo_model_save_path, env=train_env, device=ppo_device_str)
                    print(f"✅ PPO model loaded successfully (Size: {model_size/1024:.1f} KB).")
                    model_loaded_successfully = True
            except Exception as e:
                print(f"❌ Error loading PPO model: {e}. Will retrain...")
                force_retrain = True
        
        if force_retrain or not model_loaded_successfully:
            print(f"🔄 Training PPO model on {ppo_device_str}...")
            
            try:
                self.ppo_model = PPO(
                    "MlpPolicy", 
                    train_env, 
                    verbose=1, 
                    device=ppo_device_str,
                    tensorboard_log=self.logs_dir, 
                    learning_rate=0.0003,
                    n_steps=2048, 
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))
                )
                
                print(f"📊 Starting PPO training for {total_timesteps:,} timesteps...")
                self.ppo_model.learn(total_timesteps=total_timesteps)
                
                # Enhanced model saving with validation
                print(f"💾 Saving PPO model...")
                try:
                    # Remove old model if exists
                    if os.path.exists(self.ppo_model_save_path):
                        os.remove(self.ppo_model_save_path)
                        print(f"🗑️ Removed old model file")
                    
                    # Save new model
                    self.ppo_model.save(self.ppo_model_save_path)
                    
                    # Validate saved model
                    if os.path.exists(self.ppo_model_save_path):
                        saved_size = os.path.getsize(self.ppo_model_save_path)
                        if saved_size > 1000:  # At least 1KB
                            print(f"✅ PPO model saved successfully (Size: {saved_size/1024:.1f} KB)")
                            print(f"📁 Location: {self.ppo_model_save_path}")
                            
                            # Test loading the saved model to ensure it's valid
                            try:
                                test_model = PPO.load(self.ppo_model_save_path, env=train_env, device=ppo_device_str)
                                print(f"✅ Model validation successful - can be loaded properly")
                                del test_model  # Clean up
                            except Exception as load_test_e:
                                print(f"⚠️ Warning: Saved model failed validation test: {load_test_e}")
                        else:
                            print(f"❌ Error: Saved model file too small ({saved_size} bytes)")
                            raise RuntimeError("Model file appears to be empty or corrupted")
                    else:
                        print(f"❌ Error: Model file was not created at {self.ppo_model_save_path}")
                        raise RuntimeError("Model file was not saved")
                
                except Exception as save_error:
                    print(f"❌ Error saving PPO model: {save_error}")
                    print(f"🔄 Attempting alternative save method...")
                    
                    # Try alternative save path
                    alt_save_path = self.ppo_model_save_path.replace('.zip', '_backup.zip')
                    try:
                        self.ppo_model.save(alt_save_path)
                        if os.path.exists(alt_save_path) and os.path.getsize(alt_save_path) > 1000:
                            print(f"✅ Model saved to alternative path: {alt_save_path}")
                            # Copy to original path
                            import shutil
                            shutil.copy2(alt_save_path, self.ppo_model_save_path)
                            print(f"✅ Model copied to original path")
                        else:
                            print(f"❌ Alternative save also failed")
                    except Exception as alt_save_error:
                        print(f"❌ Alternative save failed: {alt_save_error}")
                        print(f"⚠️ Model training completed but saving failed. Proceeding with in-memory model.")
            
            except Exception as training_error:
                print(f"❌ Error in PPO training: {training_error}")
                import traceback
                traceback.print_exc()
                return None, {}

        # Final check to ensure we have a valid model
        if not hasattr(self, 'ppo_model') or self.ppo_model is None:
            print("❌ No valid PPO model available for backtesting.")
            return None, {}

        # Backtesting
        print("\n🔄 Starting PPO backtesting...")
        if test_df_ppo.empty or len(test_df_ppo) < (look_back_const + 50): 
            print("❌ Insufficient test data for backtesting.")
            return self.ppo_model, {}

        try:
            backtest_env = BitcoinTradingEnv(
                test_df_ppo, 
                maker_fee=0.0005,
                taker_fee=0.0015,
                slippage_factor=0.0005,
                profit_reward_weight=3.0,
                cvar_penalty_weight=1.0,
                min_transaction_pct=self.min_transaction_pct,
                max_hold_duration=self.max_hold_duration
            )
            print("✅ Backtest environment created")
        except ValueError as e:
            print(f"❌ Error creating backtest environment: {e}")
            return self.ppo_model, {}

        # Run backtest
        obs, done = backtest_env.reset(), False
        portfolio_values = [backtest_env.initial_balance]
        actions_taken = []
        btc_prices_at_step = []
        rewards_log = []

        print("\n🔄 Running backtest...")
        step_count = 0

        while not done:
            try:
                action, _ = self.ppo_model.predict(obs, deterministic=False)
                obs, reward, done, info = backtest_env.step(action)
                
                prev_value = portfolio_values[-1]
                current_value = info['portfolio_value']
                pct_change = (current_value - prev_value)/prev_value * 100 if prev_value != 0 else 0
                
                # Log trade decisions
                trade_info = ""
                if 'trade_decision' in info and info['trade_decision']['action'] != "HOLD":
                    decision = info['trade_decision']
                    trade_info = f" | {decision['action']} {decision['position_size']*100:.1f}%"
                
                if step_count % 100 == 0 or done:
                    print(f"Step {step_count}: Portfolio: ${current_value:.2f} ({pct_change:+.2f}%){trade_info}")
                
                step_count += 1
                portfolio_values.append(current_value)
                actions_taken.append(action.item())
                btc_prices_at_step.append(info['current_price'])
                rewards_log.append(reward)
            except Exception as backtest_step_error:
                print(f"❌ Error during backtest step {step_count}: {backtest_step_error}")
                break

        # Get transaction history
        env_transaction_history = backtest_env.get_transaction_history()

        # Calculate performance metrics
        action_counts = {
            'buy': sum(1 for t in env_transaction_history if t['type'] == 'BUY'),
            'sell': sum(1 for t in env_transaction_history if t['type'] == 'SELL'),
            'hold': step_count - len(env_transaction_history)
        }
        
        print(f"\n📊 Backtest Summary:")
        print(f"   - Total steps: {step_count}")
        print(f"   - Buy orders: {action_counts['buy']}")
        print(f"   - Sell orders: {action_counts['sell']}")
        print(f"   - Hold decisions: {action_counts['hold']}")
        print(f"   - Initial: ${portfolio_values[0]:.2f}")
        print(f"   - Final: ${portfolio_values[-1]:.2f}")

        # Calculate returns and metrics
        if len(portfolio_values) < 2:
            print("❌ Insufficient portfolio values for performance calculation.")
            return self.ppo_model, {}
        
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        
        if not portfolio_returns.empty and len(portfolio_returns) > 1:
            mean_return = np.mean(portfolio_returns)
            std_return = np.std(portfolio_returns)
            
            if std_return > 1e-8:
                sharpe_ratio = (mean_return / std_return) * np.sqrt(365)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
            portfolio_returns = pd.Series()

        # Calculate other metrics
        max_drawdown_pct = calculate_max_drawdown(portfolio_values) if len(portfolio_values) > 1 else 0.0
        total_return_pct = ((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]) * 100
        
        # Buy & hold comparison
        initial_btc_amount = backtest_env.initial_balance / btc_prices_at_step[0] if btc_prices_at_step else 0
        final_btc_price = btc_prices_at_step[-1] if btc_prices_at_step else 0
        buy_hold_value = initial_btc_amount * final_btc_price
        buy_hold_return_pct = ((buy_hold_value - portfolio_values[0]) / portfolio_values[0]) * 100
        
        outperformance_pct = total_return_pct - buy_hold_return_pct
        
        # Final portfolio composition
        final_usd_balance = backtest_env.balance
        final_btc_held = backtest_env.btc_held
        final_btc_value = final_btc_held * btc_prices_at_step[-1] if btc_prices_at_step else 0

        # Compile metrics
        ppo_metrics = {
            'total_return': total_return_pct, 
            'sharpe_ratio': sharpe_ratio, 
            'max_drawdown': abs(max_drawdown_pct), 
            'num_trades': len(env_transaction_history),
            'buy_count': action_counts['buy'],
            'sell_count': action_counts['sell'],
            'portfolio_values': portfolio_values,
            'btc_buy_hold_values': backtest_env.btc_buy_hold_values,
            'btc_prices': btc_prices_at_step,
            'returns': portfolio_returns.tolist(),
            'transactions': env_transaction_history,
            'buy_hold_return': buy_hold_return_pct,
            'outperformance': outperformance_pct,
            'final_usd_balance': final_usd_balance,
            'final_btc_held': final_btc_held,
            'final_btc_value': final_btc_value
        }
        
        print(f"\n📊 Performance Results:")
        print(f"   📈 Strategy Return: {ppo_metrics['total_return']:.2f}%")
        print(f"   📈 Buy & Hold Return: {buy_hold_return_pct:.2f}%")
        print(f"   🎯 Alpha (Outperformance): {outperformance_pct:+.2f}%")
        print(f"   📊 Sharpe Ratio: {ppo_metrics['sharpe_ratio']:.2f}")
        print(f"   📉 Max Drawdown: {ppo_metrics['max_drawdown']:.2f}%")
        print(f"   💰 Final USD: ${final_usd_balance:,.2f}")
        print(f"   ₿ Final BTC: {final_btc_held:.6f} BTC (${final_btc_value:,.2f})")

        # Create plots and save results
        self._load_or_create_dir(self.results_dir)
        
        # PPO results plot
        ppo_plot_save_path = os.path.join(self.results_dir, 'ppo_results.png')
        if not os.path.exists(ppo_plot_save_path):
            plot_ppo_results(
                test_data_index=test_df_ppo.index,
                portfolio_values=portfolio_values, 
                btc_prices_at_step=btc_prices_at_step, 
                actions_taken=actions_taken,
                performance_metrics=ppo_metrics,
                btc_buy_hold_values=backtest_env.btc_buy_hold_values,
                save_path=ppo_plot_save_path, 
                interactive=self.interactive
            )
            print(f"📈 PPO results plot saved to {ppo_plot_save_path}")
        else:
            print(f"✅ PPO results plot already exists")
        
        # Wallet composition plot
        wallet_plot_save_path = os.path.join(self.results_dir, 'wallet_composition.png')
        if hasattr(backtest_env, 'plot_wallet_composition'):
            if not os.path.exists(wallet_plot_save_path):
                backtest_env.plot_wallet_composition(
                    save_path=wallet_plot_save_path,
                    interactive=self.interactive
                )
                print(f"📈 Wallet composition plot saved to {wallet_plot_save_path}")
            else:
                print(f"✅ Wallet composition plot already exists")
        
        # Save transaction history
        transactions_path = os.path.join(self.results_dir, 'transaction_history.csv')
        if not os.path.exists(transactions_path):
            pd.DataFrame(env_transaction_history).to_csv(transactions_path, index=False)
            print(f"💾 Transaction history saved to {transactions_path}")
        else:
            print(f"✅ Transaction history CSV already exists")
        
        return self.ppo_model, ppo_metrics
    
    def run_integrated_strategy(self, force_retrain_lstm=False, force_retrain_ppo=False, 
                          bl_risk_aversion=2.5, bl_tau=0.03, bl_confidence=0.65, bl_view_impact=0.35):
        """Run complete integrated Bitcoin portfolio optimization strategy"""
        start_time = time.time()
        self._load_or_create_dir(self.results_dir)

        try:
            print("\n" + "="*80)
            print("🚀 BITCOIN PORTFOLIO OPTIMIZATION - INTEGRATED STRATEGY")
            print("="*80)
            print(f"📅 Analysis Period: {self.start_date} to {self.end_date}")
            print(f"🎯 Focus: Spot Trading (No Leverage)")
            print(f"🤖 Pipeline: LSTM → Black-Litterman → Enhanced CVaR → PPO")
            print("="*80)

            # Step 1-4: Data preparation and indicators
            raw_data = self.fetch_yahoo_data(start_date=self.start_date, end_date=self.end_date)
            self.preprocess_data(raw_data, lstm_predictions_for_ppo=None)
            self.feature_engineering()
            indicators = self.check_technical_indicators()

            # Step 5: LSTM prediction
            _, predictions_df, lstm_eval_metrics = self.train_or_load_lstm_model(
                force_retrain=force_retrain_lstm, 
                epochs=100,
                patience=15
            )

            # Step 5.5: Black-Litterman optimization (spot trading)
            bl_results = self.optimize_portfolio_black_litterman(
                risk_aversion=bl_risk_aversion,
                tau=bl_tau,
                confidence_level=bl_confidence,
                view_impact=bl_view_impact
            )

            # Step 6: Enhanced CVaR risk assessment
            cvar_value, risk_components = self.calculate_risk_metrics(confidence_level=0.90)

            # Reprocess data with LSTM predictions for PPO
            lstm_predictions_for_ppo = None
            if self.lstm_predictions is not None and not self.lstm_predictions.empty:
                lstm_predictions_for_ppo = self.lstm_predictions
                print("\n🔄 Re-preprocessing data to include LSTM predictions for PPO...")
                self.preprocess_data(raw_data, lstm_predictions_for_ppo=lstm_predictions_for_ppo)
                print("✅ Data re-processed with LSTM predictions")
            
            # Steps 7-8: PPO training and backtesting
            _, ppo_eval_metrics = self.train_or_load_ppo_model(
                force_retrain=force_retrain_ppo, 
                total_timesteps=200000
            )
            
            # Calculate total time
            end_time = time.time()
            computation_time = end_time - start_time
            
            # Compile results
            combined_metrics = {
                'lstm_prediction_accuracy': lstm_eval_metrics if lstm_eval_metrics else {},
                'bl_results': bl_results if bl_results else {},
                'cvar_historical_value': cvar_value,
                'cvar_adjusted_value': risk_components.get('adjusted_cvar', cvar_value) if risk_components else cvar_value,
                'cvar_components': risk_components if risk_components else {},
                'ppo_trading_metrics': ppo_eval_metrics if ppo_eval_metrics else {},
                'computation_time_seconds': computation_time
            }
            
            # Print comprehensive summary
            print("\n" + "="*80)
            print("📊 INTEGRATED STRATEGY PERFORMANCE SUMMARY")
            print("="*80)
            
            # LSTM Results
            if combined_metrics['lstm_prediction_accuracy']:
                rmse = combined_metrics['lstm_prediction_accuracy'].get('rmse_pct_change', float('nan'))
                mae = combined_metrics['lstm_prediction_accuracy'].get('mae_pct_change', float('nan'))
                print(f"🤖 LSTM Model Performance:")
                print(f"   - RMSE: {rmse:.4f}")
                print(f"   - MAE: {mae:.4f}")
                quality = "Excellent" if rmse < 0.015 else "Good" if rmse < 0.03 else "Needs Improvement"
                print(f"   - Quality: {quality}")
            
            # Black-Litterman Results
            if combined_metrics['bl_results']:
                bl_res = combined_metrics['bl_results']
                print(f"📈 Black-Litterman Optimization (Spot Trading):")
                print(f"   - Return Adjustment: {bl_res.get('return_adjustment', 0)*100:+.3f}%")
                print(f"   - Risk Adjustment: {bl_res.get('risk_adjustment', 0)*100:+.3f}%")
                print(f"   - Optimal BTC Allocation: {bl_res.get('optimal_weights', [0])[0]*100:.1f}%")
                
                if bl_res.get('views_summary', {}).get('has_views', False):
                    views = bl_res['views_summary']
                    print(f"   - LSTM View Confidence: {views.get('confidence_level', 0)*100:.1f}%")
                    print(f"   - View Impact: {views.get('view_impact', 0)*100:.1f}%")
                
            # CVaR Results
            print(f"📉 Enhanced CVaR Risk Assessment:")
            print(f"   - Traditional CVaR (90%): {combined_metrics['cvar_historical_value']:.6f}")
            print(f"   - Multi-Factor CVaR: {combined_metrics['cvar_adjusted_value']:.6f}")
            
            risk_change = 0
            if (combined_metrics['cvar_historical_value'] != 0 and 
                not np.isnan(combined_metrics['cvar_historical_value']) and
                not np.isnan(combined_metrics['cvar_adjusted_value'])):
                
                risk_change = ((combined_metrics['cvar_adjusted_value'] - combined_metrics['cvar_historical_value']) / 
                              abs(combined_metrics['cvar_historical_value'])) * 100
                print(f"   - Risk Change: {risk_change:+.1f}%")
            
            # PPO Trading Results
            if combined_metrics['ppo_trading_metrics']:
                ppo_m = combined_metrics['ppo_trading_metrics']
                print(f"🤖 PPO Trading Strategy Results:")
                print(f"   - Strategy Return: {ppo_m.get('total_return', 0):.2f}%")
                print(f"   - Buy & Hold Return: {ppo_m.get('buy_hold_return', 0):.2f}%")
                print(f"   - Alpha (Outperformance): {ppo_m.get('outperformance', 0):+.2f}%")
                print(f"   - Sharpe Ratio: {ppo_m.get('sharpe_ratio', 0):.2f}")
                print(f"   - Max Drawdown: {ppo_m.get('max_drawdown', 0):.2f}%")
                print(f"   - Total Transactions: {ppo_m.get('num_trades', 0)}")
                print(f"   - Buy/Sell Orders: {ppo_m.get('buy_count', 0)}/{ppo_m.get('sell_count', 0)}")
                
                # Performance rating
                outperf = ppo_m.get('outperformance', 0)
                rating = "🌟 Excellent" if outperf > 15 else "✓ Good" if outperf > 5 else "⚠️ Moderate" if outperf > 0 else "❌ Underperforming"
                print(f"   - Performance Rating: {rating}")
            
            print(f"\n⏱️ Total Computation Time: {computation_time:.1f} seconds")
            print("="*80)
            
            # Create integrated results plot
            integrated_plot_save_path = os.path.join(self.results_dir, 'integrated_strategy_results.png')
            if not os.path.exists(integrated_plot_save_path):
                plot_integrated_results(
                    combined_metrics, 
                    save_path=integrated_plot_save_path, 
                    interactive=self.interactive
                )
                print(f"📈 Integrated results plot saved to {integrated_plot_save_path}")
            else:
                print(f"✅ Integrated results plot already exists")
            
            return combined_metrics
            
        except Exception as e:
            print(f"❌ Error in integrated strategy: {e}")
            import traceback
            traceback.print_exc()
            
            end_time = time.time()
            computation_time = end_time - start_time
            
            # Return partial results if available
            partial_metrics = {
                'computation_time_seconds': computation_time,
                'error': str(e)
            }
            
            if hasattr(self, 'lstm_eval_metrics') and self.lstm_eval_metrics:
                partial_metrics['lstm_prediction_accuracy'] = self.lstm_eval_metrics
                
            if hasattr(self, 'bl_results') and self.bl_results:
                partial_metrics['bl_results'] = self.bl_results
                
            if hasattr(self, 'cvar_value') and self.cvar_value:
                partial_metrics['cvar_historical_value'] = self.cvar_value
                
            return partial_metrics


def main(non_interactive_mode=False, retrain_lstm=False, retrain_ppo=False, use_gpu_flag=False, 
         output_directory=None, min_transaction_pct=0.1, max_hold_duration=20,
         bl_risk_aversion=2.5, bl_tau=0.03, bl_confidence=0.65, bl_view_impact=0.35,
         create_csv=False):
    """Main function to run Bitcoin Portfolio Optimization System"""
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    if use_gpu_flag and torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(42)

    # Setup output directory
    default_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    base_output_path = os.path.abspath(output_directory if output_directory else default_output_path)
    
    # Handle CSV creation
    if create_csv:
        print("\n" + "="*80)
        print("📊 CREATING BITCOIN CSV FOR MODEL STORAGE")
        print("="*80)
        
        csv_output_dir = os.path.join(base_output_path, "model")
        csv_path = create_bitcoin_csv_for_model(csv_output_dir)
        
        if csv_path:
            print(f"✅ Bitcoin CSV created successfully at {csv_path}")
        else:
            print("❌ Failed to create Bitcoin CSV!")
        
        print("="*80)
        return
    
    # Print system configuration
    print("\n" + "="*80)
    print("🚀 BITCOIN PORTFOLIO OPTIMIZATION SYSTEM")
    print("📊 Multi-Model Integration: LSTM + Black-Litterman + CVaR + PPO")
    print("🎯 Focus: Spot Trading (No Leverage)")
    print("="*80)
    print(f"📁 Output Directory: {base_output_path}")
    print(f"📅 Data Period: 2019-01-01 to 2025-01-01")
    print(f"🖥️ System Configuration:")
    print(f"   - Interactive Mode: {not non_interactive_mode}")
    print(f"   - Force Retrain LSTM: {retrain_lstm}")
    print(f"   - Force Retrain PPO: {retrain_ppo}")
    print(f"   - GPU Acceleration: {use_gpu_flag}")
    print(f"📊 Trading Parameters:")
    print(f"   - Min Transaction Size: {min_transaction_pct*100:.1f}%")
    print(f"   - Max Hold Duration: {max_hold_duration} days")
    print(f"🎯 Black-Litterman Parameters (Spot Trading):")
    print(f"   - Risk Aversion: {bl_risk_aversion}")
    print(f"   - Tau (Uncertainty): {bl_tau}")
    print(f"   - LSTM Confidence: {bl_confidence*100:.1f}%")
    print(f"   - View Impact: {bl_view_impact*100:.1f}%")
    print("="*80)
    
    # Initialize optimizer
    optimizer = BitcoinPortfolioOptimizer(
        base_output_dir=base_output_path, 
        interactive=not non_interactive_mode, 
        use_gpu=use_gpu_flag,
        min_transaction_pct=min_transaction_pct,
        max_hold_duration=max_hold_duration
    )
    
    try:
        # Run integrated strategy
        metrics = optimizer.run_integrated_strategy(
            force_retrain_lstm=retrain_lstm, 
            force_retrain_ppo=retrain_ppo,
            bl_risk_aversion=bl_risk_aversion,
            bl_tau=bl_tau,
            bl_confidence=bl_confidence,
            bl_view_impact=bl_view_impact
        )
        
        # Save performance metrics
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.float32):
                    return float(obj)
                if isinstance(obj, pd.Timestamp):
                    return str(obj)
                return json.JSONEncoder.default(self, obj)
        
        metrics_path = os.path.join(base_output_path, 'results', 'performance_metrics.json')
        
        # Filter metrics for JSON serialization
        filtered_metrics = {}
        for k, v in metrics.items():
            if k in ['lstm_prediction_accuracy', 'cvar_historical_value', 'cvar_adjusted_value', 'computation_time_seconds']:
                filtered_metrics[k] = v
            elif k == 'bl_results' and v:
                filtered_bl = {
                    'return_adjustment': v.get('return_adjustment', 0),
                    'risk_adjustment': v.get('risk_adjustment', 0),
                    'prior_volatility_annualized': v.get('prior_volatility_annualized', 0),
                    'posterior_volatility_annualized': v.get('posterior_volatility_annualized', 0),
                    'optimal_weights': v.get('optimal_weights', [0]).tolist() if hasattr(v.get('optimal_weights', [0]), 'tolist') else v.get('optimal_weights', [0]),
                    'views_summary': v.get('views_summary', {}),
                    'model_quality': v.get('model_quality', {})
                }
                filtered_metrics[k] = filtered_bl
            elif k == 'cvar_components' and v:
                filtered_cvar = {
                    'traditional_cvar': v.get('traditional_cvar', 0),
                    'adjusted_cvar': v.get('adjusted_cvar', 0),
                    'weighted_adjustment': v.get('weighted_adjustment', 0),
                    'weight_distribution': v.get('weight_distribution', {})
                }
                filtered_metrics[k] = filtered_cvar
            elif k == 'ppo_trading_metrics' and v:
                filtered_ppo = {}
                for pk, pv in v.items():
                    if pk not in ['portfolio_values', 'btc_buy_hold_values', 'transactions', 'returns']:
                        filtered_ppo[pk] = pv
                filtered_metrics[k] = filtered_ppo
                
        if not os.path.exists(metrics_path):
            with open(metrics_path, 'w') as f:
                json.dump(filtered_metrics, f, cls=NumpyEncoder, indent=2)
                print(f"💾 Performance metrics saved to {metrics_path}")
        else:
            print(f"✅ Performance metrics JSON already exists")
        
    except Exception as e:
        print(f"❌ Error running integrated strategy: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Final summary
    print(f"\n" + "="*80)
    print("🎉 BITCOIN PORTFOLIO OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"📁 All outputs saved to: {base_output_path}")
    print(f"📊 Check these directories:")
    print(f"   - 📈 Results & Plots: {optimizer.results_dir}")
    print(f"   - 🤖 Trained Models: {optimizer.model_dir}")
    print(f"   - 📋 Training Logs: {optimizer.logs_dir}")
    print(f"   - 📊 Data & Predictions: {optimizer.data_predictions_dir}")
    print(f"\n🔄 System is ready for future use with new CSV data!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bitcoin Portfolio Optimization - Spot Trading System")
    
    # Basic options
    parser.add_argument("--non-interactive", action="store_true", 
                       help="Run in non-interactive mode (no plot displays)")
    parser.add_argument("--retrain-lstm", action="store_true", 
                       help="Force retraining of LSTM model")
    parser.add_argument("--retrain-ppo", action="store_true", 
                       help="Force retraining of PPO model")
    parser.add_argument("--gpu", action="store_true", 
                       help="Enable GPU acceleration if available")
    
    # Directory options
    parser.add_argument("--output-dir", type=str, default=None, 
                       help="Base directory for all outputs")
    
    # Trading parameters
    parser.add_argument("--min-transaction", type=float, default=0.1, 
                       help="Minimum transaction percentage (0.1 = 10%%)")
    parser.add_argument("--max-hold", type=int, default=20, 
                       help="Maximum hold duration in days")
    
    # Black-Litterman parameters (for spot trading)
    parser.add_argument("--bl-risk-aversion", type=float, default=2.5,
                       help="Risk aversion parameter (default: 2.5)")
    parser.add_argument("--bl-tau", type=float, default=0.03,
                       help="Uncertainty scaling parameter (default: 0.03)")
    parser.add_argument("--bl-confidence", type=float, default=0.65,
                       help="Confidence in LSTM predictions (0.0-1.0, default: 0.65)")
    parser.add_argument("--bl-view-impact", type=float, default=0.35,
                       help="Impact of LSTM views (0.0-1.0, default: 0.35)")
    
    # Utility options
    parser.add_argument("--create-csv", action="store_true",
                       help="Create Bitcoin OHLCV CSV file and exit")
    
    args = parser.parse_args()
    
    main(
        non_interactive_mode=args.non_interactive, 
        retrain_lstm=args.retrain_lstm, 
        retrain_ppo=args.retrain_ppo, 
        use_gpu_flag=args.gpu,
        output_directory=args.output_dir,
        min_transaction_pct=args.min_transaction,
        max_hold_duration=args.max_hold,
        bl_risk_aversion=args.bl_risk_aversion,
        bl_tau=args.bl_tau,
        bl_confidence=args.bl_confidence,
        bl_view_impact=args.bl_view_impact,
        create_csv=args.create_csv
    )