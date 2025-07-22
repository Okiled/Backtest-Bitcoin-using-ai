# 🚀 Bitcoin AI Portfolio Optimization System

An advanced AI-powered Bitcoin portfolio management framework integrating **LSTM neural networks**, **Black-Litterman model**, **Enhanced CVaR risk management**, and **PPO reinforcement learning** for optimal cryptocurrency trading strategies.

## 📋 Overview

This research presents a comprehensive Bitcoin portfolio optimization system that combines cutting-edge artificial intelligence techniques with modern portfolio theory. The framework addresses Bitcoin's inherent volatility and non-normal return distributions through an integrated approach of predictive modeling, Bayesian portfolio optimization, advanced risk management, and adaptive trading strategies.

### 🎯 Key Innovation

Unlike traditional backtesting systems, this framework integrates multiple AI methodologies to create a robust, adaptive trading system that can handle the extreme volatility and unique characteristics of cryptocurrency markets.

## ✨ Key Features

- **🧠 LSTM Price Prediction**: Deep learning model capturing temporal dependencies in Bitcoin price movements
- **📊 Black-Litterman Integration**: Bayesian portfolio optimization incorporating AI-driven market views
- **⚠️ Enhanced CVaR Risk Management**: Multi-component Conditional Value at Risk for tail risk protection
- **🤖 PPO Reinforcement Learning**: Adaptive trading agent optimizing risk-adjusted returns
- **📈 Technical Analysis Integration**: RSI and Keltner Channel indicators for market momentum analysis
- **📊 Comprehensive Performance Analytics**: Detailed backtesting results and risk metrics
- **🔄 Dynamic Portfolio Rebalancing**: Real-time position adjustment based on market conditions

## 🏆 Performance Highlights

### Backtesting Results (2019-2025)
- **Total Return**: 6,280.12%
- **Benchmark Outperformance**: 3,832.45% alpha over buy-and-hold
- **Sharpe Ratio**: 1.50
- **Maximum Drawdown**: 57.00%
- **LSTM Prediction Accuracy**: RMSE 0.0262, MAE 0.0191
- **Trading Efficiency**: 148 trades (70% win rate)

*Note: Past performance does not guarantee future results*

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for LSTM training)
- 8GB+ RAM recommended

### Quick Setup

1. Clone the repository:
```bash
git clone https://github.com/Okiled/Backtest-Bitcoin-using-ai.git
cd Backtest-Bitcoin-using-ai
```

2. Install dependencies

3. (Optional) GPU setup for faster training:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚦 Quick Start

### Basic Execution

Run the complete optimization pipeline:
```bash
python main.py
```

### Custom Parameters

```bash
python main.py --start-date 2020-01-01 --end-date 2024-01-01 --initial-capital 100000
```

### Example Implementation

```python
from project_component import BitcoinPortfolioOptimizer

# Initialize the optimization system
optimizer = BitcoinPortfolioOptimizer(
    initial_capital=100000,
    start_date='2020-01-01',
    end_date='2024-01-01',
    lstm_sequence_length=60,
    confidence_level=0.65
)

# Run the complete pipeline
results = optimizer.run_optimization_pipeline()

# Generate comprehensive reports
optimizer.generate_performance_report()
optimizer.visualize_results()
```

## 🏗️ System Architecture

### 1. **Data Processing Pipeline**
- Historical Bitcoin price data (OHLCV) from Yahoo Finance
- Technical indicator calculation (RSI, Keltner Channels)
- Min-Max normalization and missing value imputation
- Feature engineering for predictive modeling

### 2. **LSTM Prediction Engine**
```
Input Layer (18 features) → LSTM Layers (50 units, 2 layers) → Dropout (0.3) → Dense Output
```
- **Sequence Length**: 60 time steps
- **Features**: Price data + technical indicators
- **Optimization**: Adam optimizer (lr=0.00005)
- **Early Stopping**: Implemented at epoch 85

### 3. **Black-Litterman Portfolio Optimization**
- **Market Equilibrium**: δ × Σ × W_market
- **Investor Views**: LSTM predictions with 65% confidence
- **Posterior Returns**: Bayesian combination of equilibrium and views
- **Risk Aversion Parameter**: 2.5

### 4. **Enhanced CVaR Risk Management**
Multi-component CVaR calculation:
- Base Model (32%)
- RSI-adjusted (24%) 
- Keltner-adjusted (24%)
- Black-Litterman adjusted (20%)

### 5. **PPO Trading Agent**
- **State Space**: 18-dimensional market features
- **Action Space**: Continuous [0,1] → Buy/Sell/Hold
- **Reward Function**: 0.8×Returns - 0.1×CVaR_penalty + 0.1×Trade_bonus
- **Training**: 150,000 timesteps, batch size 128

## 📊 Model Performance Analysis

### LSTM Prediction Accuracy
- **RMSE**: 0.0262 (2.62% prediction error)
- **MAE**: 0.0191 (1.91% average error)
- **Trend Capture**: Successfully identified major trend reversals during 2020-2022 bull run

### Black-Litterman Optimization
- **Initial Market Return**: 9.12% annually
- **LSTM-Adjusted Return**: 7.00% annually
- **Risk Reduction**: Volatility decreased from 19.1% to 14.3%
- **Posterior Sharpe Ratio**: 0.35

### CVaR Risk Management
- **90% CVaR**: -6.74% (tail risk quantification)
- **Dynamic Risk Adjustment**: Adapts to market volatility conditions
- **Early Warning System**: Technical indicators provide pre-emptive risk signals



## 🔧 Configuration & Customization

### Model Hyperparameters

```python
LSTM_CONFIG = {
    'sequence_length': 60,
    'hidden_size': 50,
    'num_layers': 2,
    'dropout_rate': 0.3,
    'learning_rate': 0.00005,
    'batch_size': 32
}

BLACK_LITTERMAN_CONFIG = {
    'confidence_level': 0.65,
    'risk_aversion': 2.5,
    'tau': 0.025
}

PPO_CONFIG = {
    'total_timesteps': 150000,
    'batch_size': 128,
    'learning_rate': 5e-4,
    'clip_range': 0.2
}
```

### Risk Management Settings

```python
RISK_CONFIG = {
    'cvar_confidence': 0.90,
    'max_position_size': 0.95,
    'stop_loss_threshold': -0.15,
    'rebalancing_frequency': 'daily'
}
```

## 📈 Key Research Contributions

### 1. **Novel Integration Framework**
First comprehensive system combining LSTM, Black-Litterman, Enhanced CVaR, and PPO for cryptocurrency portfolio optimization.

### 2. **Enhanced CVaR Methodology**
Multi-component Conditional Value at Risk incorporating technical analysis and model-based adjustments for superior tail risk management.

### 3. **Adaptive Trading Strategy**
PPO-based reinforcement learning agent that dynamically adjusts trading decisions based on market conditions and risk metrics.

### 4. **Robust Performance Validation**
Comprehensive backtesting across multiple market cycles with detailed performance attribution analysis.

## 📊 Technical Indicators Implementation

### Relative Strength Index (RSI)
```
RSI = 100 - (100 / (1 + RS))
RS = Average_Gain_14 / Average_Loss_14
```

### Keltner Channel
```
Middle Line = EMA₂₆(close)
Upper Band = Middle Line + (0.5 × ATR₁₀)
Lower Band = Middle Line - (0.5 × ATR₁₀)
```

## 🎯 Use Cases

### Institutional Applications
- **Hedge Funds**: Systematic cryptocurrency trading strategies
- **Asset Managers**: Portfolio diversification with digital assets
- **Pension Funds**: Risk-controlled crypto allocation

### Individual Investors
- **Retail Trading**: Automated Bitcoin investment strategies
- **Portfolio Management**: Professional-grade optimization tools
- **Risk Management**: Advanced downside protection methods

### Academic Research
- **Quantitative Finance**: Novel AI integration methodologies
- **Cryptocurrency Studies**: Volatility modeling and prediction
- **Reinforcement Learning**: Financial applications research

## ⚠️ Important Disclaimers

### Risk Warnings
- **High Volatility**: Cryptocurrency markets are extremely volatile
- **Model Risk**: AI predictions are not guaranteed to be accurate
- **Market Risk**: Past performance does not predict future results
- **Regulatory Risk**: Cryptocurrency regulations may change
- **Technical Risk**: System failures can result in losses

### Limitations
- **Single Asset Focus**: Currently optimized for Bitcoin only
- **Historical Data Dependency**: Performance based on past market conditions
- **Computational Requirements**: Requires significant computing resources
- **Market Assumption**: Assumes liquid markets and accurate data feeds

## 🔮 Future Research Directions

### Short-term Enhancements
- **Multi-Asset Extension**: Include Ethereum, other major cryptocurrencies
- **Real-time Implementation**: Live trading system development
- **Sentiment Integration**: Social media and news sentiment analysis
- **Alternative Data**: On-chain metrics and macroeconomic indicators

### Long-term Development
- **Transformer Models**: Advanced attention-based architectures
- **Deep Reinforcement Learning**: More sophisticated RL approaches
- **Cross-Market Analysis**: Integration with traditional financial markets
- **Regulatory Compliance**: Framework for institutional requirements


### Contribution Areas
- Model improvements and new algorithms
- Performance optimization and efficiency
- Additional risk metrics and analysis
- Real-time trading implementation
- Documentation and examples



### 📊 Performance Dashboard

| Metric | Value | Benchmark | Outperformance |
|--------|-------|-----------|----------------|
| Total Return | 6,280.12% | 2,447.67% | +3,832.45% |
| Sharpe Ratio | 1.50 | 0.85 | +0.65 |
| Max Drawdown | 57.00% | 84.20% | +27.20% |
| Win Rate | 70% | 100%* | N/A |
| Volatility | 14.3% | 19.1% | -4.8% |

*Buy-and-hold has 100% "win rate" but with higher volatility and drawdown

