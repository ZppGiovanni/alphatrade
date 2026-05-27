"""
stream.py — Live market data feed via Alpaca WebSocket.

Run as a background process alongside the dashboard:
    python data/stream.py

Subscribes to minute bars for all 5 ETFs and writes them to the
ohlcv_live table in SQLite. The dashboard reads from this table to
display real-time prices.
"""

import logging
import os
import sys

sys.path.insert(0, ".")

from contextlib import closing
from dotenv import load_dotenv
from alpaca.data.live import StockDataStream

from data.database import get_connection, init_db

load_dotenv()
logger = logging.getLogger(__name__)

ASSETS = ["QQQ", "XLE", "GLD", "XLV", "ARKK"]

_INSERT_SQL = """
    INSERT OR REPLACE INTO ohlcv_live
        (timestamp, ticker, open, high, low, close, volume)
    VALUES (?, ?, ?, ?, ?, ?, ?)
"""


def save_live_bar(bar) -> None:
    """Persist a single Alpaca live bar to ohlcv_live."""
    ts = bar.timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    with closing(get_connection()) as conn:
        conn.execute(
            _INSERT_SQL,
            (ts, bar.symbol, bar.open, bar.high, bar.low, bar.close, bar.volume),
        )
        conn.commit()
    logger.info("Live bar: %s %s close=%.2f", bar.symbol, ts, bar.close)


async def _bar_handler(bar) -> None:
    save_live_bar(bar)


class LiveFeed:
    """Alpaca WebSocket live feed for minute bars.

    Args:
        assets: List of ticker symbols to subscribe to.
    """

    def __init__(self, assets: list[str] = ASSETS) -> None:
        api_key = os.getenv("ALPACA_KEY")
        api_secret = os.getenv("ALPACA_SECRET")
        if not api_key or not api_secret:
            raise EnvironmentError("ALPACA_KEY and ALPACA_SECRET must be set in .env")
        self._stream = StockDataStream(api_key, api_secret)
        self._assets = assets

    def run(self) -> None:
        """Start streaming (blocking). Press Ctrl+C to stop."""
        self._stream.subscribe_bars(_bar_handler, *self._assets)
        logger.info("LiveFeed started — subscribed to: %s", ", ".join(self._assets))
        self._stream.run()

    def stop(self) -> None:
        """Gracefully stop the stream."""
        self._stream.stop()
        logger.info("LiveFeed stopped.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    init_db()
    feed = LiveFeed()
    try:
        feed.run()
    except KeyboardInterrupt:
        feed.stop()
