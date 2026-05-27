# AGENTS.md — AlphaTrade

This file tells AI agents how to contribute to this project.
Read it fully before making any changes.

---

## Project overview

AlphaTrade is an algorithmic trading system for five thematic ETFs (QQQ, XLE, GLD, XLV, ARKK).
It fetches historical and live market data, generates trading signals via quantitative strategies,
optimizes a multi-asset portfolio using Markowitz mean-variance, and visualizes everything on a
Streamlit dashboard with AI-powered market analysis.

**Stack**: Python 3.11+, SQLite, yfinance, alpaca-py, pandas, scikit-learn, Streamlit, Plotly, Groq API

**Repo structure**:

```
alphatrade/
├── data/
│   ├── fetcher.py         # yfinance historical download
│   ├── normalizer.py      # cleaning, validation, indicator calc
│   └── database.py        # SQLite read/write helpers
├── strategies/
│   ├── base.py            # abstract Strategy class
│   ├── mean_reversion.py  # z-score mean reversion
│   ├── momentum.py        # dual SMA crossover
│   ├── macd_crossover.py  # MACD signal line crossover
│   ├── bollinger_bands.py # Bollinger Bands mean reversion (AI agent PR #9)
│   └── ml_model.py        # Random Forest classifier
├── portfolio/
│   ├── optimizer.py       # Markowitz / max-Sharpe
│   └── risk.py            # backtesting engine, drawdown, Sharpe
├── dashboard/
│   └── streamlit_app.py   # Streamlit dashboard (main)
├── tests/
│   └── test_strategies.py
├── docs/
│   └── architecture.md
├── AGENTS.md              # this file
└── README.md
```

---

## AI agent contributions

### Completed

| PR | Branch | Description | Author |
|----|--------|-------------|--------|
| [#5](https://github.com/ZppGiovanni/alphatrade/pull/5) | `feat/macd-crossover-strategy` | MACD Crossover strategy + tests | Andrea Matteri |
| [#9](https://github.com/ZppGiovanni/alphatrade/pull/9) | `feat/bollinger-bands-strategy` | Bollinger Bands strategy + tests | AI agent (Claude) |

### How to contribute as an agent

The preferred task for an AI agent is adding a new trading strategy.
This is a well-scoped, self-contained task that touches exactly the right parts of the codebase
and produces a verifiable, testable output.

**Steps:**

1. Create `strategies/your_strategy_name.py`
2. Inherit from `strategies/base.py`:

```python
from strategies.base import Strategy
import pandas as pd

class YourStrategy(Strategy):
    def __init__(self, params: dict):
        super().__init__(params)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Input:  OHLCV DataFrame with columns [open, high, low, close, volume]
                plus indicators [sma_20, sma_50, rsi, macd, macd_signal,
                bb_upper, bb_lower, return_1d, return_5d].
                Indexed by date string, one ticker at a time.
        Output: pd.Series of signals: 1 (buy), -1 (sell), 0 (hold),
                same index as input df.
        """
        ...
```

3. Register the strategy in `strategies/__init__.py`
4. Add at least one test in `tests/test_strategies.py`
5. Open a pull request with title: `feat: add <strategy_name> strategy`

---

## Coding conventions

- **Language**: Python 3.11+
- **Style**: PEP 8 — run `black .` before committing
- **Type hints**: required on all function signatures
- **Docstrings**: Google style, required on all public functions and classes
- **No hardcoded credentials**: use environment variables via `python-dotenv`
- **No print statements**: use `logging` module
- **DataFrame columns**: always lowercase (`open`, `high`, `low`, `close`, `volume`)
- **Timestamps**: always UTC, stored as ISO 8601 strings in SQLite

---

## Environment setup

```bash
git clone https://github.com/ZppGiovanni/alphatrade.git
cd alphatrade
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys
pytest tests/
```

Required `.env` variables:
```
ALPACA_KEY=your_key_here
ALPACA_SECRET=your_secret_here
GROQ_API_KEY=your_groq_key_here
```

---

## Testing requirements

- Every new strategy **must** include a test in `tests/test_strategies.py`
- Tests use `pytest` and synthetic data — no live API calls in tests
- Minimum: verify output is a `pd.Series` of `{-1, 0, 1}` with the correct index

Example test pattern:

```python
import pandas as pd
import numpy as np
from data.normalizer import add_indicators

def make_fake_ohlcv(n: int = 100) -> pd.DataFrame:
    idx   = pd.date_range("2023-01-01", periods=n, freq="1D")
    close = pd.Series(np.random.randn(n).cumsum() + 100, index=idx)
    return pd.DataFrame({
        "open":   close * 0.99,
        "high":   close * 1.01,
        "low":    close * 0.98,
        "close":  close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })

def test_your_strategy_signals():
    df       = add_indicators(make_fake_ohlcv())
    strategy = YourStrategy(params={})
    signals  = strategy.generate_signals(df)
    assert isinstance(signals, pd.Series)
    assert set(signals.dropna().unique()).issubset({-1, 0, 1})
    assert len(signals) == len(df)
```

---

## Pull request checklist

Before opening a PR:

- [ ] `black .` passes with no formatting errors
- [ ] `pytest tests/ -v` passes with no failures
- [ ] All public functions have type hints and Google-style docstrings
- [ ] No hardcoded secrets or credentials
- [ ] `requirements.txt` updated if new packages were added
- [ ] PR title follows convention: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`

---

## What agents must NOT do

- Do not modify `data/database.py` schema without updating all dependent queries
- Do not commit API keys, passwords, or secrets of any kind
- Do not add new dependencies without updating `requirements.txt`
- Do not rename or remove existing public function signatures (breaks other modules)
- Do not execute live trades — the system is paper-trading only
- Do not modify `portfolio/optimizer.py` core Markowitz logic without opening an issue first
