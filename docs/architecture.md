# AlphaTrade — Technical Architecture

## System Overview

AlphaTrade is organized as a modular pipeline:
yfinance API
│
▼
data/fetcher.py ──► SQLite DB ──► data/normalizer.py
│
▼
strategies/ (signals)
│
┌─────────┼─────────┐
▼         ▼         ▼
Momentum  MeanRev   MACD
└─────────┼─────────┘
│
▼
Ensemble Signal (-3 to +3)
│
┌─────────┴─────────┐
▼                   ▼
portfolio/risk.py    portfolio/optimizer.py
(backtesting)        (Markowitz)
│
▼
dashboard/app.py
(Streamlit + Groq AI)

## Data Layer

- **fetcher.py**: Downloads 2 years of daily OHLCV data for all 5 ETFs via yfinance
- **database.py**: SQLite storage with `load_ohlcv(ticker)` helper
- **normalizer.py**: Calculates SMA(20/50), RSI(14), MACD(12/26/9), Bollinger Bands(20,2)

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
| ML (Random Forest) | 6 features, 80/20 train/test split | Buy when P(up) > 0.6 |

## Ensemble Signal

The consensus score sums signals from all 3 quant strategies:
score = sig_momentum + sig_mean_reversion + sig_macd
range: [-3, +3]

| Score | Signal |
|-------|--------|
| +2, +3 | STRONG BUY |
| +1 | WEAK BUY |
| 0 | HOLD |
| -1 | WEAK SELL |
| -2, -3 | STRONG SELL |

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
