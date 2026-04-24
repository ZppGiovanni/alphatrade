"""
database.py — SQLite helpers for AlphaTrade.
"""
import sqlite3
import logging
import argparse
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
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    logger.info("Database initialised at %s", DB_PATH)

def save_ohlcv(df: pd.DataFrame) -> None:
    conn = get_connection()
    df.to_sql("ohlcv", conn, if_exists="append", index=True)
    conn.close()

def load_ohlcv(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    conn = get_connection()
    query = "SELECT * FROM ohlcv WHERE ticker = ?"
    params: list = [ticker]
    if start:
        query += " AND timestamp >= ?"; params.append(start)
    if end:
        query += " AND timestamp <= ?"; params.append(end)
    query += " ORDER BY timestamp ASC"
    df = pd.read_sql_query(query, conn, params=params, index_col="timestamp")
    conn.close()
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true")
    args = parser.parse_args()
    if args.init:
        init_db()
        print(f"Database ready at {DB_PATH}")
