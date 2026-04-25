"""
fetcher.py — Download historical OHLCV data via yfinance.
"""
import logging
import sys
sys.path.insert(0, '.')

import yfinance as yf
import pandas as pd
from data.database import save_ohlcv, init_db

logger = logging.getLogger(__name__)

ASSETS = ["QQQ", "XLE", "GLD", "XLV", "ARKK"]

def fetch_historical(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Download historical OHLCV for a single ticker."""
    logger.info("Fetching %s (%s, %s)", ticker, period, interval)
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    # Fix: flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = df.index.strftime("%Y-%m-%d")
    df.index.name = "timestamp"
    df["ticker"] = ticker
    return df

def fetch_all(period: str = "2y") -> None:
    """Download and store historical data for all assets."""
    init_db()
    for ticker in ASSETS:
        df = fetch_historical(ticker, period=period)
        save_ohlcv(df)
        logger.info("Saved %d rows for %s", len(df), ticker)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetch_all()
    print("Historical data download complete.")
