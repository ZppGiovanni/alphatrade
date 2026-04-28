"""
macd_crossover.py — MACD Crossover trading strategy for AlphaTrade.

Buy when MACD line crosses above the signal line (bullish crossover).
Sell when MACD line crosses below the signal line (bearish crossover).
Hold otherwise.
"""

import pandas as pd
from strategies.base import Strategy


class MACDCrossoverStrategy(Strategy):
    """
    MACD Crossover trend-following strategy.

    Signals:
        Buy  ( 1): MACD crosses above signal line (bullish momentum)
        Sell (-1): MACD crosses below signal line (bearish momentum)
        Hold ( 0): no crossover detected

    Params:
        (none required — uses macd and macd_signal columns already in df)
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        macd = df["macd"]
        signal = df["macd_signal"]

        # True when macd is above signal, False when below
        macd_above = macd > signal

        # Crossover: today different from yesterday
        signals = pd.Series(0, index=df.index)
        signals[macd_above & ~macd_above.shift(1).fillna(False).astype(bool)] = 1
        signals[~macd_above & macd_above.shift(1).fillna(True).astype(bool)] = -1

        return signals


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from data.database import load_ohlcv
    from data.normalizer import add_indicators

    df = load_ohlcv("QQQ")
    df = add_indicators(df)
    strategy = MACDCrossoverStrategy(params={})
    signals = strategy.generate_signals(df)
    print(signals.value_counts())
    print("MACD Crossover signals OK!")
