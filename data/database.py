"""
database.py — SQLite helpers for AlphaTrade.
"""

import sqlite3
import logging
import argparse
from contextlib import closing
from functools import lru_cache
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).parent.parent / "alphatrade.db"
logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    timestamp   TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    open        REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (timestamp, ticker)
);
CREATE TABLE IF NOT EXISTS ohlcv_live (
    timestamp   TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    open        REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (timestamp, ticker)
);
CREATE TABLE IF NOT EXISTS signals (
    timestamp   TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    signal      INTEGER,
    PRIMARY KEY (timestamp, ticker, strategy)
);
CREATE TABLE IF NOT EXISTS portfolio_weights (
    timestamp   TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    weight      REAL,
    PRIMARY KEY (timestamp, ticker)
);
"""


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.executescript(SCHEMA)
        conn.commit()
    logger.info("Database initialised at %s", DB_PATH)


def _insert_or_ignore(table, conn, keys, data_iter):
    placeholders = ", ".join(["?"] * len(keys))
    cols = ", ".join(keys)
    sql = f"INSERT OR IGNORE INTO {table.name} ({cols}) VALUES ({placeholders})"
    conn.executemany(sql, list(data_iter))


def save_ohlcv(df: pd.DataFrame) -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        df.to_sql("ohlcv", conn, if_exists="append", index=True, method=_insert_or_ignore)


@lru_cache(maxsize=32)
def _load_ohlcv_cached(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    query = "SELECT * FROM ohlcv WHERE ticker = ?"
    params: list = [ticker]
    if start:
        query += " AND timestamp >= ?"
        params.append(start)
    if end:
        query += " AND timestamp <= ?"
        params.append(end)
    query += " ORDER BY timestamp ASC"
    with closing(sqlite3.connect(DB_PATH)) as conn:
        return pd.read_sql_query(query, conn, params=params, index_col="timestamp")


def load_ohlcv(
    ticker: str, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    return _load_ohlcv_cached(ticker, start, end).copy()


def load_live_price(ticker: str) -> float | None:
    """Return the latest live close price for a ticker, or None if unavailable.

    Args:
        ticker: ETF ticker symbol.

    Returns:
        Latest close price from ohlcv_live, or None if no live data exists.
    """
    sql = """
        SELECT close FROM ohlcv_live
        WHERE ticker = ?
        ORDER BY timestamp DESC LIMIT 1
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute(sql, [ticker]).fetchone()
    return float(row[0]) if row else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true")
    args = parser.parse_args()
    if args.init:
        init_db()
        print(f"Database ready at {DB_PATH}")
