# 🤖 RL Entry Optimizer

A reinforcement learning-powered trade entry optimization engine that learns to enter long positions based on historical price data and technical indicators.

---

## 🔧 Features

- ✅ PPO agent via Stable-Baselines3
- ✅ Custom Gymnasium-compatible `TradeEnv`
- ✅ RSI and MACD features included
- ✅ Batch training for 11 stocks and 7 cryptocurrencies
- ✅ Interactive equity viewer (next/prev buttons, arrow keys)
- ✅ Dockerized for easy deployment

---

## 📊 Assets Trained

**Stocks**: AMZN, EPD, ET, GOOGL, IBM, META, MSFT, PG, RGTI, RITM, TSM  
**Cryptos**: JASMY, RENDER, BTC, ADA, ALGO, XLM, XRP

---

## ▶️ Running the Project

### Locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
