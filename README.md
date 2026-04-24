# AlphaTrade — Algorithmic ETF Trading System

> Final project for Programming in Finance II — USI Lugano, 2026  
> Group: [nomi del gruppo]

## Overview

AlphaTrade is an algorithmic trading system that allocates dynamically across five
thematic ETFs using quantitative strategies, portfolio optimization, and machine learning.

**Universe**: QQQ (Tech) · XLE (Energy) · GLD (Gold) · XLV (Healthcare) · ARKK (Innovation)

**Core thesis**: A rules-based system can exploit regime changes across uncorrelated
thematic sectors — rotating between growth, value, safe-haven, and defensive assets —
outperforming a static equal-weight benchmark on a risk-adjusted basis.

## Features

- **Live market data** via Alpaca WebSocket + historical data via yfinance
- **Quantitative strategies**: mean reversion, momentum, moving average crossover
- **Portfolio optimization**: Markowitz mean-variance, max-Sharpe ratio
- **ML signal**: scikit-learn model for directional price prediction
- **Backtesting engine**: walk-forward testing with performance metrics
- **Web dashboard**: Streamlit app with interactive charts and portfolio analytics
- **Agentic development**: AI-assisted contributions via AGENTS.md

## Project structure

```
alphatrade/
├── data/               # Market data fetching, streaming, storage
│   ├── fetcher.py      # yfinance historical download
│   ├── stream.py       # Alpaca WebSocket live feed
│   ├── normalizer.py   # Cleaning, validation, technical indicators
│   └── database.py     # SQLite read/write helpers
├── strategies/         # Trading signal generation
│   ├── base.py         # Abstract Strategy interface
│   ├── mean_reversion.py
│   ├── momentum.py
│   └── ml_model.py     # ML-based signal (scikit-learn)
├── portfolio/          # Portfolio construction
│   ├── optimizer.py    # Markowitz / max-Sharpe optimization
│   └── risk.py         # Position sizing, stop-loss, drawdown
├── execution/          # Order management
│   └── order_manager.py  # Paper trading via Alpaca
├── dashboard/          # Web interface
│   └── app.py          # Streamlit dashboard
├── tests/              # Unit tests (pytest)
├── docs/               # Additional documentation
├── AGENTS.md           # AI agent contribution guide
└── requirements.txt
```

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/alphatrade.git
cd alphatrade

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your Alpaca API keys (free at alpaca.markets)

# 5. Initialize the database
python data/database.py --init

# 6. Download historical data
python data/fetcher.py

# 7. Run the dashboard
streamlit run dashboard/app.py
```

## Usage

### Download historical data
```python
from data.fetcher import fetch_all

fetch_all()  # Downloads 2y of daily OHLCV for all 5 ETFs
```

### Run a strategy backtest
```python
from data.database import load_ohlcv
from strategies.momentum import MomentumStrategy
from portfolio.optimizer import MaxSharpeOptimizer

df = load_ohlcv("QQQ")
strategy = MomentumStrategy(params={"window": 20})
signals = strategy.generate_signals(df)
```

### Launch dashboard
```bash
streamlit run dashboard/app.py
```

## Assets

| Ticker | Name | Theme | Risk |
|--------|------|-------|------|
| QQQ | Invesco Nasdaq-100 | Tech / Growth | High |
| XLE | Energy Select SPDR | Energy / Value | Medium |
| GLD | SPDR Gold Shares | Safe Haven | Low |
| XLV | Health Care Select SPDR | Defensive | Low |
| ARKK | ARK Innovation ETF | Disruptive Innovation | Very High |

## Tech stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Data | yfinance, alpaca-py |
| Database | SQLite + SQLAlchemy |
| Indicators | pandas-ta |
| Optimization | PyPortfolioOpt |
| ML | scikit-learn |
| Dashboard | Streamlit + Plotly |
| Testing | pytest |

## Team

| Name | Role |
|------|------|
| [Nome 1] | Data layer — pipeline, database, live feed |
| [Nome 2] | Strategies — signals, backtesting, optimization |
| [Nome 3] | ML + execution — model, order manager, risk |
| [Nome 4] | Dashboard — Streamlit app, docs, LaTeX report |

## Academic documentation

PDF report available on iCorsi. Includes project plan, development diary,
mathematical background (Markowitz, technical indicators, ML model),
sample results, and lessons learned.

## License

MIT — see LICENSE file.
