# AlphaTrade — Algorithmic ETF Trading System

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![USI](https://img.shields.io/badge/USI-Programming%20in%20Finance%20II-purple?style=flat-square)

> Final project for **Programming in Finance II** — USI Lugano, 2026
> Group: Andrea Matteri · Giovanni Zoppis · Manuela Benfante · Federico Pizzati

## Overview

AlphaTrade is an algorithmic trading system that dynamically allocates across five thematic ETFs using quantitative strategies, portfolio optimization, and machine learning. The system combines rule-based signals with an ensemble voting mechanism and AI-powered market analysis via a Groq LLM.

The core idea is simple: a rules-based system that rotates between growth, value, safe-haven, and defensive assets depending on market regime — and does it better than a static equal-weight benchmark on a risk-adjusted basis.

**Universe**: QQQ (Tech) · XLE (Energy) · GLD (Gold) · XLV (Healthcare) · ARKK (Innovation)

---

## Key Results

| Asset | Strategy | Return | Buy & Hold | Sharpe |
|-------|----------|--------|------------|--------|
| XLE | Mean Reversion | **+49.4%** | +25.5% | 1.8 |
| GLD | Momentum | **+63.6%** | +98.9% | 1.29 |
| QQQ | MACD Crossover | **+31.5%** | +98.9% | 0.92 |

---

## Features

- **Live market data** — historical OHLCV via yfinance (2Y, 2501 rows)
- **Technical indicators** — SMA, RSI, MACD, Bollinger Bands (auto-calculated)
- **5 trading strategies** — Momentum, Mean Reversion, MACD Crossover, Bollinger Bands, ML (Random Forest)
- **Ensemble signal** — voting system combining all 4 quant strategies (score from -4 to +4)
- **Portfolio optimization** — Markowitz mean-variance, max-Sharpe ratio
- **Backtesting engine** — equity curve, Sharpe ratio, max drawdown
- **Web dashboard** — Streamlit with candlestick charts, RSI subplot, interactive signals
- **AI market analysis** — LLM-powered insights via Groq LLaMA 3.3 70B
- **Agentic development** — AI agent contributions tracked via AGENTS.md and Pull Requests

---

## Project Structure

```
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
│   ├── macd_crossover.py  # MACD signal line crossover
│   ├── bollinger_bands.py # Bollinger Bands mean reversion — AI agent PR #9
│   └── ml_model.py        # Random Forest classifier
├── portfolio/
│   ├── optimizer.py      # Markowitz / max-Sharpe optimization
│   └── risk.py           # Backtesting engine, drawdown, Sharpe
├── execution/
│   └── order_manager.py  # Paper trading via Alpaca
├── dashboard/
│   └── streamlit_app.py  # Streamlit dashboard (main)
├── tests/
│   └── test_strategies.py
├── docs/
│   └── architecture.md   # System architecture and design decisions
├── AGENTS.md             # AI agent contribution guide
└── requirements.txt
```

---

## Installation

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

Edit `.env` with your API keys:

```
ALPACA_KEY=your_alpaca_key
ALPACA_SECRET=your_alpaca_secret
GROQ_API_KEY=your_groq_key
```

```bash
# 5. Initialize the database
python data/database.py --init

# 6. Download historical data
python data/fetcher.py

# 7. Run the dashboard
streamlit run dashboard/streamlit_app.py
```

---

## User Guide

Once the dashboard is running at `http://localhost:8501`, here is how to use it:

### Sidebar controls

| Control | Description |
|---------|-------------|
| **Asset** | Select one of the 5 ETFs: QQQ, XLE, GLD, XLV, ARKK |
| **Strategy** | Choose the signal-generation strategy: Momentum, Mean Reversion, MACD Crossover, Bollinger Bands, or ML Model |
| **Period** | Filter the data window: 1M, 3M, 6M, 1Y, 2Y, 5Y |

The **Quick Stats** panel at the bottom of the sidebar always shows the last close price, 1-day return, RSI, and current ensemble signal for the selected asset.

### Tabs

**Price & Signals** — Candlestick chart with SMA 20/50, Bollinger Bands, buy/sell markers from the selected strategy, volume bar chart, and RSI subplot. Four KPI cards at the top show last close, RSI, 1D return, and 5D return.

**MACD** — MACD line vs signal line chart with histogram. Useful for identifying momentum shifts independently from the selected strategy.

**Backtest** — Equity curve of the selected strategy vs buy-and-hold, drawdown chart, rolling Sharpe ratio, and rolling volatility. Shows Strategy Return, Buy & Hold return, Sharpe Ratio, and Max Drawdown as KPIs.

**Comparison** — Side-by-side performance table and bar chart for all four quantitative strategies (Momentum, Mean Reversion, MACD Crossover, Bollinger Bands) on the selected asset and period.

**Portfolio** — Max-Sharpe portfolio weights via Markowitz optimization (donut + bar chart), asset correlation heatmap, and monthly returns heatmap.

**Consensus** — Ensemble signal combining all four quant strategies into a score from -4 (strong sell) to +4 (strong buy). Includes a gauge chart, consensus bar chart over time, and daily returns distribution.

**AI Analysis** — Click "Generate AI Analysis" to call the Groq LLaMA 3.3 70B API. The model receives the last 10 days of market data, current indicators, and all strategy results, then returns a professional analysis covering market conditions, signal interpretation, risk considerations, and outlook.

### Running a backtest from code

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

print(f"Return:    {results['total_return']:+.1%}")
print(f"Sharpe:    {results['sharpe_ratio']:.2f}")
print(f"Max DD:    {results['max_drawdown']:.1%}")
print(f"N Trades:  {results['n_trades']}")
```

### Running tests

```bash
pytest tests/ -v
```

---

## API Documentation

### `strategies/base.py` — Abstract Strategy

All strategies inherit from `Strategy` and must implement `generate_signals`.

```python
class Strategy(ABC):
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with columns [open, high, low, close, volume, sma_20, sma_50,
            rsi, macd, macd_signal, bb_upper, bb_lower, return_1d, return_5d].
            Index: date strings (YYYY-MM-DD). One ticker at a time.

        Returns
        -------
        pd.Series
            Signal series with values in {-1, 0, 1} (sell, hold, buy).
            Same index as input df.
        """
```

### `strategies/momentum.py` — MomentumStrategy

Dual SMA crossover: buy when SMA20 crosses above SMA50, sell when it crosses below.

```python
MomentumStrategy(params={"short_window": 20, "long_window": 50})
```

### `strategies/mean_reversion.py` — MeanReversionStrategy

Z-score mean reversion: buy when z-score < -threshold, sell when z-score > +threshold.

```python
MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5})
```

### `strategies/macd_crossover.py` — MACDCrossoverStrategy

MACD signal line crossover: buy when MACD crosses above signal, sell when it crosses below.

```python
MACDCrossoverStrategy(params={})  # uses standard 12/26/9 params
```

### `strategies/bollinger_bands.py` — BollingerBandsStrategy

Bollinger Bands mean reversion: buy when price crosses below the lower band, sell when it crosses above the upper band. Generated via AI agent (PR #9).

```python
BollingerBandsStrategy(params={"window": 20, "num_std": 2.0})
```

### `strategies/ml_model.py` — MLStrategy

Random Forest classifier trained on lagged returns and technical indicators.

```python
MLStrategy(params={"n_estimators": 100, "threshold": 0.6})
```

### `portfolio/risk.py` — backtest()

```python
def backtest(prices: pd.Series, signals: pd.Series, initial_capital: float = 10000) -> dict:
    """
    Returns
    -------
    dict with keys:
        total_return   : float  — strategy total return
        buy_and_hold   : float  — passive B&H return over same period
        sharpe_ratio   : float  — annualized Sharpe ratio
        max_drawdown   : float  — maximum peak-to-trough drawdown
        n_trades       : int    — number of completed round trips
        equity_curve   : pd.Series — portfolio value over time
    """
```

### `portfolio/optimizer.py` — compute_weights()

```python
def compute_weights(prices: pd.DataFrame) -> dict:
    """
    Markowitz mean-variance optimization (max Sharpe).

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of close prices, one column per ticker.

    Returns
    -------
    dict  — {ticker: weight} mapping, weights sum to 1.
    """
```

### `data/database.py` — load_ohlcv()

```python
def load_ohlcv(ticker: str) -> pd.DataFrame:
    """
    Load OHLCV data from SQLite for the given ticker.

    Returns
    -------
    pd.DataFrame with columns [open, high, low, close, volume], index = date string.
    """
```

### `data/normalizer.py` — add_indicators()

```python
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to an OHLCV DataFrame.
    Computes: sma_20, sma_50, rsi, macd, macd_signal, bb_upper, bb_lower,
              return_1d, return_5d.
    Returns the same DataFrame with new columns appended.
    """
```

---

## Assets

| Ticker | Name | Theme | Risk |
|--------|------|-------|------|
| QQQ | Invesco Nasdaq-100 | Tech / Growth | High |
| XLE | Energy Select SPDR | Energy / Value | Medium |
| GLD | SPDR Gold Shares | Safe Haven | Low |
| XLV | Health Care Select SPDR | Defensive | Low |
| ARKK | ARK Innovation ETF | Disruptive Innovation | Very High |

---

## Tech Stack

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

## Team

| Name | Role | Contribution |
|------|------|-------------|
| **Andrea Matteri** | Strategies · Dashboard | MACD Crossover strategy, Bollinger Bands dashboard integration, dashboard redesign, ensemble signal, AI market analysis |
| **Giovanni Zoppis** | Architecture · Data · ML · Backend | Data pipeline, database, strategies, ML model, portfolio optimizer, backtesting engine, dashboard |
| **Manuela Benfante** | Documentation · Testing | LaTeX report, testing support |
| **Federico Pizzati** | Documentation · Presentation | LaTeX report, presentation slides |

---

## AI Tools Used

- **Claude (Anthropic)** — architecture design, code generation, Bollinger Bands strategy (AI agent PR #9)
- **Groq LLaMA 3.3 70B** — live AI market analysis in the dashboard
- **GitHub Copilot** — inline code suggestions

All AI-generated code was reviewed, tested, and integrated by the team.

---

## Project Tracking

Issues and task tracking: [github.com/ZppGiovanni/alphatrade/issues](https://github.com/ZppGiovanni/alphatrade/issues)

---

## Academic Documentation

PDF report available on iCorsi. Includes project plan, development diary, mathematical background (Markowitz, MACD, RSI, Random Forest), sample results, and lessons learned.

---

## License

MIT — see LICENSE file.
