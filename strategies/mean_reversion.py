import pandas as pd
from strategies.base import Strategy


class MeanReversionStrategy(Strategy):
    """
    Buy when price is Z standard deviations below rolling mean.
    Sell when price is Z standard deviations above rolling mean.

    Params:
        window (int): Rolling window. Default 20.
        z_threshold (float): Z-score threshold. Default 1.5.
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        window = self.params.get("window", 20)
        z = self.params.get("z_threshold", 1.5)
        close = df["close"]
        rolling_mean = close.rolling(window).mean()
        rolling_std = close.rolling(window).std()
        z_score = (close - rolling_mean) / rolling_std
        signals = pd.Series(0, index=df.index)
        signals[z_score < -z] = 1
        signals[z_score > z] = -1
        return signals


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from data.database import load_ohlcv

    df = load_ohlcv("QQQ")
    strategy = MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5})
    signals = strategy.generate_signals(df)
    print(signals.value_counts())
    print("Mean reversion signals OK!")
