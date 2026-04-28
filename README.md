# ⚡ AlphaTrade — Algorithmic ETF Trading System

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![USI](https://img.shields.io/badge/USI-Programming%20in%20Finance%20II-purple?style=flat-square)

> Final project for **Programming in Finance II** — USI Lugano, 2026
> Group: Andrea Matteri · Giovanni Zoppis · Manuela Benfante · Federico Pizzati

## Overview

AlphaTrade is an algorithmic trading system that dynamically allocates across five thematic ETFs using quantitative strategies, portfolio optimization, and machine learning. The system combines rule-based signals with an ensemble voting mechanism and AI-powered market analysis.

**Universe**: QQQ (Tech) · XLE (Energy) · GLD (Gold) · XLV (Healthcare) · ARKK (Innovation)

**Core thesis**: A rules-based system can exploit regime changes across uncorrelated thematic sectors — rotating between growth, value, safe-haven, and defensive assets — outperforming a static equal-weight benchmark on a risk-adjusted basis.

---

## 🏆 Key Results

| Asset | Strategy | Return | Buy & Hold | Sharpe |
|-------|----------|--------|------------|--------|
| XLE | Mean Reversion | **+49.4%** | +25.5% | 1.8 |
| GLD | Momentum | **+63.6%** | +98.9% | 1.29 |
| QQQ | MACD Crossover | **+31.5%** | +98.9% | 0.92 |

---

## ✨ Features

- **Live market data** — historical OHLCV via yfinance (2Y, 2501 rows)
- **Technical indicators** — SMA, RSI, MACD, Bollinger Bands (auto-calculated)
- **4 trading strategies** — Momentum, Mean Reversion, MACD Crossover, ML (Random Forest)
- **Ensemble signal** — voting system combining all 3 quant strategies (-3 to +3 score)
- **Portfolio optimization** — Markowitz mean-variance, max-Sharpe ratio
- **Backtesting engine** — equity curve, Sharpe ratio, max drawdown
- **Web dashboard** — Streamlit with candlestick charts, RSI subplot, interactive signals
- **AI market analysis** — LLM-powered insights via Groq LLaMA 3.3 70B
- **Agentic development** — AI agent contributions via AGENTS.md + Pull Requests

---

## 📁 Project Structure
alphatrade/
├── data/
│   ├── fetcher.py        # yfinance historical download
│   ├── stream.py         # Alpaca WebSocket live feed
│   ├── normalizer.py     # Cleaning, validation, technical indicators
│   └── database.py       # SQLite read/write helpers
├── strategies/
│   ├── base.py           # Abstract Strategy interface
│   ├── mean_reversion.py # Z-score mean reversion
│   ├── momentum.py       # Dual moving average crossover
│   ├── macd_crossover.py # MACD signal line crossover (AI agent PR)
│   └── ml_model.py       # Random Forest classifier
├── portfolio/
│   ├── optimizer.py      # Markowitz / max-Sharpe optimization
│   └── risk.py           # Backtesting engine, drawdown, Sharpe
├── execution/
│   └── order_manager.py  # Paper trading via Alpaca
├── dashboard/
│   └── app.py            # Streamlit dashboard
├── tests/
│   └── test_strategies.py
├── docs/
├── AGENTS.md             # AI agent contribution guide
└── requirements.txt

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/ZppGiovanni/alphatrade.git
cd alphatrade

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
```

Edit .env with your API keys:
ALPACA_KEY=your_alpaca_key
ALPACA_SECRET=your_alpaca_secret
GROQ_API_KEY=your_groq_key

```bash
# 5. Initialize the database
python data/database.py --init

# 6. Download historical data
python data/fetcher.py

# 7. Run the dashboard
streamlit run dashboard/app.py
```

---

## 📖 Usage

### Run a strategy backtest
```python
from data.database import load_ohlcv
from data.normalizer import add_indicators
from strategies.macd_crossover import MACDCrossoverStrategy
from portfolio.risk import backtest

df = load_ohlcv("QQQ")
df = add_indicators(df)
strategy = MACDCrossoverStrategy(params={})
signals = strategy.generate_signals(df)
results = backtest(df["close"], signals)
print(f"Return: {results['total_return']:+.1%}")
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
```

### Run tests
```bash
pytest tests/ -v
```

---

## 📊 Assets

| Ticker | Name | Theme | Risk |
|--------|------|-------|------|
| QQQ | Invesco Nasdaq-100 | Tech / Growth | High |
| XLE | Energy Select SPDR | Energy / Value | Medium |
| GLD | SPDR Gold Shares | Safe Haven | Low |
| XLV | Health Care Select SPDR | Defensive | Low |
| ARKK | ARK Innovation ETF | Disruptive Innovation | Very High |

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Data | yfinance, alpaca-py |
| Database | SQLite |
| Indicators | pandas, numpy |
| Optimization | PyPortfolioOpt |
| ML | scikit-learn (Random Forest) |
| Dashboard | Streamlit + Plotly |
| AI Analysis | Groq API (LLaMA 3.3 70B) |
| Testing | pytest |
| Code style | black |

---

## 👥 Team

| Name | Role | Contribution |
|------|------|-------------|
| **Andrea Matteri** | Strategies · AI Agent · Dashboard | MACD Crossover strategy, AI agent PR, dashboard redesign, ensemble signal, AI market analysis |
| **Giovanni Zoppis** | Architecture · Data · ML · Backend | Data pipeline, database, strategies, ML model, portfolio optimizer, backtesting engine, dashboard |
| **Manuela Benfante** | Documentation · Testing | LaTeX report, testing support |
| **Federico Pizzati** | Documentation · Presentation | LaTeX report, presentation slides |

---

## 🤖 AI Tools Used

- **Claude (Anthropic)** — architecture design, code generation, MACD Crossover strategy (AI agent PR)
- **Groq LLaMA 3.3 70B** — live AI market analysis in the dashboard
- **GitHub Copilot** — inline code suggestions

All AI-generated code was reviewed, tested, and integrated by the team.

---

## 📄 Academic Documentation

PDF report available on iCorsi. Includes project plan, development diary, mathematical background (Markowitz, MACD, RSI, Random Forest), sample results, and lessons learned.

---

## License

MIT — see LICENSE file.
