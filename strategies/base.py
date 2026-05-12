"""
base.py — Abstract base class for all AlphaTrade strategies.
All strategies must inherit from this class and implement generate_signals().
"""

from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    To add a new strategy:
    1. Create a new file in strategies/
    2. Inherit from Strategy
    3. Implement generate_signals()
    4. Register in strategies/__init__.py
    """

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume],
                indexed by timestamp. One ticker at a time.

        Returns:
            pd.Series of int signals with same index as df:
                1  = buy
               -1  = sell
                0  = hold
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.params})"
