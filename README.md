# Bitcoin Portfolio Optimization using AI 🧠📈

An advanced Bitcoin portfolio optimization system that combines **LSTM**, **Black-Litterman**, **Enhanced CVaR**, and **Proximal Policy Optimization (PPO)** to generate superior risk-adjusted returns in volatile cryptocurrency markets.

---

## 📊 Key Highlights

- 🔮 **LSTM** — Predicts future BTC returns using deep learning on historical + technical indicators  
- 📈 **Black-Litterman** — Bayesian portfolio optimization with AI-based investor views  
- ⚠️ **CVaR** — Enhanced Conditional Value at Risk for tail risk management  
- 🤖 **PPO** — Reinforcement learning agent for dynamic trade execution  

---

## 🧪 Results Summary

| Metric                | Value        |
|-----------------------|--------------|
| Cumulative Return     | 6,280.12%    |
| Sharpe Ratio          | 1.50         |
| Max Drawdown          | 57.00%       |
| Trading Period        | 2019–2025    |
| Total Trades          | 148 (103 buy, 45 sell) |

![PPO Results](<img width="3570" height="1998" alt="image" src="https://github.com/user-attachments/assets/7b0cf363-c5db-4e03-9bb4-76f8c70a80b1" />)  
*Fig: PPO Strategy Portfolio Performance*

---
🧠 Model Design Overview
🔮 LSTM (Long Short-Term Memory)
Predicts percentage returns using historical BTC data, RSI, and Keltner Channels.
→ Evaluation: RMSE: 0.0262, MAE: 0.0191

📊 Black-Litterman Portfolio Optimization
Incorporates LSTM-predicted views with prior market equilibrium and user confidence.

⚠️ Enhanced CVaR Risk Control
Multi-component risk model (RSI-adjusted, Keltner-adjusted, BL-adjusted) to anticipate tail losses.

🤖 PPO Agent
Reinforcement learning agent trained to optimize long-term returns, guided by risk-aware reward shaping.

📄 Project Origin
This project was developed as part of a research initiative on AI-based cryptocurrency trading strategies.
Although not formally published, the methodology and results were modeled on real financial research standards and backtested on actual BTC data.

