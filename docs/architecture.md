# AlphaTrade — Technical Architecture

## System Overview

AlphaTrade is organized as a modular pipeline:

yfinance API          Alpaca WebSocket
│                          │
▼                          ▼
data/fetcher.py        data/stream.py
│  (historical)        │  (live bars)
└──────────┬───────────┘
           ▼
       SQLite DB
    ┌──────────────┐
    │ ohlcv        │  historical daily bars
    │ ohlcv_live   │  real-time minute bars
    └──────┬───────┘
           ▼
    data/normalizer.py
           │
           ▼
    strategies/ (signals)
           │
  ┌────────┼────────┐
  ▼        ▼        ▼
Momentum MeanRev  MACD + Bollinger
  └────────┼────────┘
           ▼
  Ensemble Signal (-4 to +4)
           │
  ┌────────┴────────┐
  ▼                 ▼
portfolio/risk.py   portfolio/optimizer.py
(backtesting)       (Markowitz)
           │
           ▼
  dashboard/streamlit_app.py
  (Streamlit + Groq AI + Live Price)

## Data Layer

- **fetcher.py**: Downloads 2 years of daily OHLCV data for all 5 ETFs via yfinance
- **stream.py**: Alpaca WebSocket client — subscribes to minute bars for all 5 ETFs and writes them to the `ohlcv_live` SQLite table. Run as a background process alongside the dashboard (`python data/stream.py`)
- **database.py**: SQLite storage with 4 tables (`ohlcv`, `ohlcv_live`, `signals`, `portfolio_weights`). Key helpers: `load_ohlcv(ticker)`, `load_live_price(ticker)`
- **normalizer.py**: Calculates SMA(20/50), RSI(14), MACD(12/26/9), Bollinger Bands(20,2)

### Real-time data flow

The dashboard fetches near-realtime prices two ways (in priority order):
1. **yfinance 1-min bar** — `@st.cache_data(ttl=60)`, refreshes every 60 seconds
2. **Alpaca WebSocket** — `stream.py` writes minute bars to `ohlcv_live`; the dashboard reads `load_live_price(ticker)` as fallback

The sidebar shows a **LIVE** badge next to the current price when live data is available.

## Strategy Layer

All strategies inherit from `strategies/base.py`:

```python
class Strategy(ABC):
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Returns pd.Series of {-1, 0, 1}
```

| Strategy | Logic | Signals |
|----------|-------|---------|
| Momentum | SMA(20) vs SMA(50) crossover | Buy when short > long |
| Mean Reversion | Z-score of price vs rolling mean | Buy when Z < -1.5 |
| MACD Crossover | MACD vs Signal line crossover | Buy on bullish cross |
| Bollinger Bands | Price vs upper/lower bands (20, 2σ) | Buy when price < lower band |
| ML (Random Forest) | 6 features, 80/20 train/test split | Buy when P(up) > 0.6 |

## Ensemble Signal

The consensus score sums signals from all 4 quant strategies:
score = sig_momentum + sig_mean_reversion + sig_macd + sig_bollinger
range: [-4, +4]

| Score | Signal |
|-------|--------|
| +3, +4 | STRONG BUY |
| +1, +2 | WEAK BUY |
| 0 | HOLD |
| -1, -2 | WEAK SELL |
| -3, -4 | STRONG SELL |

## Portfolio Optimization

Markowitz mean-variance optimization via PyPortfolioOpt:
- Maximizes Sharpe ratio
- Uses 2Y daily returns covariance matrix
- Long-only constraints

## Backtesting Engine

Simulates strategy on historical data with $10,000 initial capital:
- Metrics: total return, buy & hold return, Sharpe ratio, max drawdown, N trades
- Equity curve for visualization

## AI Analysis

On-demand market analysis via Groq API (LLaMA 3.3 70B):
- Input: last 10 days OHLCV, indicators, backtest results, ensemble signal
- Output: professional market analysis (max 200 words)
