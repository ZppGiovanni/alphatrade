import pandas as pd
from strategies.base import Strategy


class MomentumStrategy(Strategy):
    """
    Buy when short MA crosses above long MA (golden cross).
    Sell when short MA crosses below long MA (death cross).

    Params:
        short_window (int): Short MA window. Default 20.
        long_window (int): Long MA window. Default 50.
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        short = self.params.get("short_window", 20)
        long_ = self.params.get("long_window", 50)
        close = df["close"]
        ma_short = close.rolling(short).mean()
        ma_long = close.rolling(long_).mean()
        signals = pd.Series(0, index=df.index)
        signals[ma_short > ma_long] = 1
        signals[ma_short < ma_long] = -1
        return signals


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.database import load_ohlcv
    df = load_ohlcv("QQQ")
    strategy = MomentumStrategy(params={"short_window": 20, "long_window": 50})
    signals = strategy.generate_signals(df)
    print(signals.value_counts())
    print("Momentum signals OK!")