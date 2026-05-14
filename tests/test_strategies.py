"""test_strategies.py — Unit tests for trading strategies."""

import pytest
import pandas as pd
import numpy as np
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy


def make_fake_ohlcv(n: int = 100) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    idx = pd.date_range("2023-01-01", periods=n, freq="1D")
    close = pd.Series(np.random.randn(n).cumsum() + 100, index=idx)
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        }
    )


def assert_valid_signals(signals: pd.Series, df: pd.DataFrame) -> None:
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(df)
    assert set(signals.dropna().unique()).issubset({-1, 0, 1})


def test_mean_reversion_signals():
    df = make_fake_ohlcv()
    strategy = MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5})
    signals = strategy.generate_signals(df)
    assert_valid_signals(signals, df)


def test_momentum_signals():
    df = make_fake_ohlcv()
    strategy = MomentumStrategy(params={"short_window": 10, "long_window": 30})
    signals = strategy.generate_signals(df)
    assert_valid_signals(signals, df)


def test_mean_reversion_repr():
    s = MeanReversionStrategy(params={"window": 20})
    assert "MeanReversionStrategy" in repr(s)


from strategies.macd_crossover import MACDCrossoverStrategy
from data.normalizer import add_indicators


def test_macd_crossover_signals():
    df = make_fake_ohlcv(n=100)
    df = add_indicators(df)
    strategy = MACDCrossoverStrategy(params={})
    signals = strategy.generate_signals(df)
    assert_valid_signals(signals, df)


def test_macd_crossover_repr():
    strategy = MACDCrossoverStrategy(params={})
    assert "MACDCrossoverStrategy" in repr(strategy)
def test_bollinger_bands_signals():
    from strategies.bollinger_bands import BollingerBandsStrategy
    df = make_fake_ohlcv()
    strategy = BollingerBandsStrategy(params={"window": 20, "num_std": 2.0})
    signals = strategy.generate_signals(df)
    assert_valid_signals(signals, df)
