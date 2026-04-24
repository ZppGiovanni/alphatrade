# AGENTS.md — Algorithmic Trading System

This file instructs AI agents (Claude, Codex, Cursor, GitHub Copilot, etc.) on how
to contribute to this project. Read it fully before making any changes.

---

## Project overview

An algorithmic trading system that fetches live and historical market data, generates
trading signals via quantitative strategies, optimizes a multi-asset portfolio, and
visualizes results on a web dashboard.

**Stack**: Python 3.11+, SQLite, yfinance, alpaca-py, pandas, scikit-learn, Streamlit  
**Markets**: Crypto (BTC-USD, ETH-USD, SOL-USD) — extendable to equity/FX  
**Repo structure**:

```
algo-trading/
├── data/
│   ├── fetcher.py        # yfinance historical download
│   ├── stream.py         # Alpaca WebSocket live feed
│   ├── normalizer.py     # cleaning, validation, indicator calc
│   └── database.py       # SQLite read/write helpers
├── strategies/
│   ├── base.py           # abstract Strategy class
│   ├── mean_reversion.py
│   ├── momentum.py
│   └── ml_model.py       # ML-based signal (sklearn)
├── portfolio/
│   ├── optimizer.py      # Markowitz / max-Sharpe
│   └── risk.py           # position sizing, stop-loss
├── execution/
│   └── order_manager.py  # paper trading via Alpaca
├── dashboard/
│   └── app.py            # Streamlit dashboard
├── tests/
│   └── test_strategies.py
├── docs/
│   └── architecture.md
├── AGENTS.md             # this file
└── README.md
```

---

## How agents should contribute

### Preferred tasks for AI agents

Agents are especially useful for:

- Adding a new trading strategy in `strategies/` that follows the `base.py` interface
- Adding unit tests in `tests/` for existing modules
- Improving docstrings and type hints across all modules
- Adding a new indicator to `data/normalizer.py` (RSI, MACD, ATR, etc.)
- Refactoring duplicated logic (e.g. DB connection boilerplate)
- Adding error handling and logging where missing
- Writing or improving documentation in `docs/`

### How to add a new strategy (main agentic task)

1. Create `strategies/your_strategy_name.py`
2. Inherit from `strategies/base.py`:

```python
from strategies.base import Strategy
import pandas as pd

class YourStrategy(Strategy):
    def __init__(self, params: dict):
        self.params = params

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Input:  OHLCV DataFrame with columns [open, high, low, close, volume]
                indexed by timestamp, one ticker at a time.
        Output: pd.Series of signals: 1 (buy), -1 (sell), 0 (hold)
                same index as input df.
        """
        # your logic here
        ...
```

3. Register the strategy in `strategies/__init__.py`
4. Add at least one test in `tests/test_strategies.py`
5. Open a pull request with title: `feat: add <strategy_name> strategy`

---

## Coding conventions

- **Language**: Python 3.11+
- **Style**: PEP 8. Use `black` for formatting (`black .` before committing)
- **Type hints**: required on all function signatures
- **Docstrings**: Google style, required on all public functions and classes
- **No hardcoded credentials**: use environment variables via `python-dotenv`
- **No print statements**: use `logging` module instead
- **DataFrame columns**: always lowercase (`open`, `high`, `low`, `close`, `volume`)
- **Timestamps**: always UTC, stored as ISO 8601 strings in SQLite

---

## Environment setup

```bash
# 1. Clone and enter
git clone https://github.com/YOUR_USERNAME/algo-trading.git
cd algo-trading

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env with your Alpaca API keys

# 5. Initialize database
python data/database.py --init

# 6. Run tests
pytest tests/
```

Required `.env` variables:
```
ALPACA_KEY=your_key_here
ALPACA_SECRET=your_secret_here
```

---

## Testing requirements

- Every new strategy **must** include a test in `tests/test_strategies.py`
- Tests use `pytest` and synthetic data (no live API calls in tests)
- Minimum test: verify output is a pd.Series of {-1, 0, 1} with correct index
- Run all tests with: `pytest tests/ -v`

Example test pattern:

```python
import pandas as pd
import numpy as np
from strategies.mean_reversion import MeanReversionStrategy

def make_fake_ohlcv(n=100) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="1D")
    close = pd.Series(np.random.randn(n).cumsum() + 100, index=idx)
    return pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.01,
        "low":  close * 0.98,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })

def test_mean_reversion_signals():
    df = make_fake_ohlcv()
    strategy = MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5})
    signals = strategy.generate_signals(df)
    assert isinstance(signals, pd.Series)
    assert set(signals.unique()).issubset({-1, 0, 1})
    assert len(signals) == len(df)
```

---

## What agents must NOT do

- Do not modify `data/database.py` schema without updating all dependent queries
- Do not commit API keys, passwords, or secrets of any kind
- Do not add new dependencies without updating `requirements.txt`
- Do not delete or rename existing public function signatures (breaks other modules)
- Do not make live trades — the system is paper-trading only
- Do not modify `portfolio/optimizer.py` core Markowitz logic without opening an issue first

---

## Pull request checklist

Before opening a PR, verify:

- [ ] `black .` run successfully (no formatting errors)
- [ ] `pytest tests/ -v` passes with no failures
- [ ] New public functions have type hints and docstrings
- [ ] No hardcoded secrets or credentials
- [ ] `requirements.txt` updated if new packages added
- [ ] PR title follows convention: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`

---

## Suggested first PR for an AI agent

> **Task**: Add a `BollingerBandsStrategy` in `strategies/bollinger.py`
>
> Signals: buy when `close < lower_band`, sell when `close > upper_band`, hold otherwise.  
> Parameters: `window` (default 20), `num_std` (default 2.0).  
> Include one test in `tests/test_strategies.py`.  
> Register in `strategies/__init__.py`.

This is a well-scoped, self-contained task that touches exactly the right parts of the
codebase and produces a verifiable, testable output — ideal for an AI agent.
