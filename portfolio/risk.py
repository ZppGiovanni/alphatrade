import pandas as pd
import numpy as np


def backtest(prices: pd.Series, signals: pd.Series,
             initial_capital: float = 10000.0) -> dict:
    """
    Simulate a trading strategy on historical data.

    Args:
        prices:  Close price series indexed by date.
        signals: Signal series (1=buy, -1=sell, 0=hold), same index.
        initial_capital: Starting portfolio value in USD.

    Returns:
        Dictionary with performance metrics and equity curve.
    """
    position = 0
    cash = initial_capital
    equity = []

    for date, price in prices.items():
        signal = signals.get(date, 0)
        if signal == 1 and position == 0:
            position = cash / price
            cash = 0.0
        elif signal == -1 and position > 0:
            cash = position * price
            position = 0.0
        total = cash + position * price
        equity.append({"date": date, "equity": total})

    equity_df = pd.DataFrame(equity).set_index("date")
    equity_curve = equity_df["equity"]

    total_return = (equity_curve.iloc[-1] / initial_capital) - 1
    buy_and_hold = (prices.iloc[-1] / prices.iloc[0]) - 1
    daily_returns = equity_curve.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    n_trades = (signals != 0).sum()

    return {
        "total_return": total_return,
        "buy_and_hold": buy_and_hold,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "n_trades": n_trades,
        "equity_curve": equity_curve,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.database import load_ohlcv
    from strategies.momentum import MomentumStrategy
    from strategies.mean_reversion import MeanReversionStrategy

    ASSETS = ["QQQ", "XLE", "GLD", "XLV", "ARKK"]

    print("=" * 55)
    print(f"{'BACKTESTING REPORT — AlphaTrade':^55}")
    print("=" * 55)

    for ticker in ASSETS:
        df = load_ohlcv(ticker)
        prices = df["close"]

        for strat_name, strategy in [
            ("Momentum", MomentumStrategy({"short_window": 20, "long_window": 50})),
            ("MeanRev",  MeanReversionStrategy({"window": 20, "z_threshold": 1.5})),
        ]:
            signals = strategy.generate_signals(df)
            r = backtest(prices, signals)
            print(f"\n{ticker} | {strat_name}")
            print(f"  Return:      {r['total_return']:+.1%}")
            print(f"  Buy & Hold:  {r['buy_and_hold']:+.1%}")
            print(f"  Sharpe:      {r['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown:{r['max_drawdown']:.1%}")
            print(f"  N trades:    {r['n_trades']}")

    print("\n" + "=" * 55)
    print("Backtest complete.")