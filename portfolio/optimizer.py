import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns


def compute_weights(prices: pd.DataFrame) -> dict:
    """
    Compute max-Sharpe portfolio weights via Markowitz optimization.

    Args:
        prices: DataFrame with dates as index, tickers as columns, close prices as values.

    Returns:
        Dictionary of ticker -> weight (weights sum to 1).
    """
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    weights = ef.clean_weights()
    return dict(weights)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.database import load_ohlcv

    ASSETS = ["QQQ", "XLE", "GLD", "XLV", "ARKK"]
    prices = pd.DataFrame()
    for ticker in ASSETS:
        df = load_ohlcv(ticker)
        prices[ticker] = df["close"]

    weights = compute_weights(prices)
    print("\nOptimal portfolio weights (Max Sharpe):")
    for ticker, w in weights.items():
        print(f"  {ticker}: {w:.1%}")
    total = sum(weights.values())
    print(f"\nTotal: {total:.1%}")
    print("Portfolio optimizer OK!")