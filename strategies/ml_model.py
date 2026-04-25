import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from strategies.base import Strategy


class MLStrategy(Strategy):
    """
    Random Forest classifier that predicts next-day price direction.

    Params:
        n_estimators (int): Number of trees. Default 100.
        threshold (float): Confidence threshold for signal. Default 0.6.
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        features = pd.DataFrame(index=df.index)
        features["return_1d"] = close.pct_change(1)
        features["return_5d"] = close.pct_change(5)
        features["sma_ratio"] = close.rolling(20).mean() / close.rolling(50).mean()
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        features["rsi"] = 100 - (100 / (1 + gain / loss))
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        features["macd"] = ema12 - ema26
        std20 = close.rolling(20).std()
        features["bb_width"] = (2 * std20) / close.rolling(20).mean()
        return features

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        features = self._build_features(df)
        close = df["close"]
        target = (close.shift(-1) > close).astype(int) * 2 - 1
        valid = features.dropna().index.intersection(target.dropna().index)
        X = features.loc[valid]
        y = target.loc[valid]
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train = y.iloc[:split]
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        self.model.fit(X_train_s, y_train)
        self.is_fitted = True
        proba = self.model.predict_proba(X_test_s)
        threshold = self.params.get("threshold", 0.6)
        signals = pd.Series(0, index=df.index)
        for i, idx in enumerate(X_test.index):
            if proba[i][1] > threshold:
                signals[idx] = 1
            elif proba[i][0] > threshold:
                signals[idx] = -1
        return signals


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.database import load_ohlcv
    df = load_ohlcv("QQQ")
    strategy = MLStrategy(params={"n_estimators": 100, "threshold": 0.6})
    signals = strategy.generate_signals(df)
    print(signals.value_counts())
    print(f"Accuracy proxy: {(signals != 0).sum()} confident predictions out of {len(signals)}")
    print("ML strategy OK!")