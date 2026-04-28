import sys

sys.path.insert(0, ".")
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from data.database import load_ohlcv
from data.normalizer import add_indicators
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.macd_crossover import MACDCrossoverStrategy
from strategies.ml_model import MLStrategy
from portfolio.optimizer import compute_weights
from portfolio.risk import backtest

ASSETS = ["QQQ", "XLE", "GLD", "XLV", "ARKK"]

st.set_page_config(page_title="AlphaTrade", page_icon="📈", layout="wide")
st.title("📈 AlphaTrade — Algorithmic ETF Trading System")
st.caption("USI Programming in Finance II, 2026")

# Sidebar
st.sidebar.header("Settings")
selected = st.sidebar.selectbox("Asset", ASSETS)
strategy_name = st.sidebar.selectbox(
    "Strategy", ["Momentum", "Mean Reversion", "MACD Crossover", "ML (Random Forest)"]
)

# Load data
df = load_ohlcv(selected)
df = add_indicators(df)

# --- SECTION 1: Price & Signals ---
st.subheader(f"{selected} — Price & Signals")

if strategy_name == "Mean Reversion":
    strategy = MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5})
elif strategy_name == "MACD Crossover":
    strategy = MACDCrossoverStrategy(params={})
elif strategy_name == "ML (Random Forest)":
    strategy = MLStrategy(params={"n_estimators": 100, "threshold": 0.6})
else:
    strategy = MomentumStrategy(params={"short_window": 20, "long_window": 50})

signals = strategy.generate_signals(df)

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(color="#4FC3F7"))
)
buys = df[signals == 1]
sells = df[signals == -1]
fig.add_trace(
    go.Scatter(
        x=buys.index,
        y=buys["close"],
        mode="markers",
        name="Buy",
        marker=dict(color="green", size=8, symbol="triangle-up"),
    )
)
fig.add_trace(
    go.Scatter(
        x=sells.index,
        y=sells["close"],
        mode="markers",
        name="Sell",
        marker=dict(color="red", size=8, symbol="triangle-down"),
    )
)
fig.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig, use_container_width=True)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Last Close", f"${df['close'].iloc[-1]:.2f}")
col2.metric("RSI", f"{df['rsi'].iloc[-1]:.1f}")
col3.metric("1D Return", f"{df['return_1d'].iloc[-1]:.2%}")
col4.metric("5D Return", f"{df['return_5d'].iloc[-1]:.2%}")

# --- SECTION 2: Backtest ---
st.subheader(f"{selected} — Backtest Results")

results = backtest(df["close"], signals)
equity_curve = results["equity_curve"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Strategy Return", f"{results['total_return']:+.1%}")
col2.metric("Buy & Hold", f"{results['buy_and_hold']:+.1%}")
col3.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
col4.metric("Max Drawdown", f"{results['max_drawdown']:.1%}")

fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        name="Strategy",
        line=dict(color="#4FC3F7"),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.1)",
    )
)
bh_curve = 10000 * (df["close"] / df["close"].iloc[0])
fig3.add_trace(
    go.Scatter(
        x=bh_curve.index,
        y=bh_curve.values,
        name="Buy & Hold",
        line=dict(color="#FFB74D", dash="dash"),
    )
)
fig3.update_layout(
    template="plotly_dark",
    height=350,
    yaxis_title="Portfolio Value ($)",
    xaxis_title="Date",
)
st.plotly_chart(fig3, use_container_width=True)

# --- SECTION 3: Backtest all strategies comparison ---
st.subheader(f"{selected} — Strategy Comparison")

strategies_map = {
    "Momentum": MomentumStrategy(params={"short_window": 20, "long_window": 50}),
    "Mean Reversion": MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5}),
    "MACD Crossover": MACDCrossoverStrategy(params={}),
}

comparison_data = []
for name, strat in strategies_map.items():
    sig = strat.generate_signals(df)
    res = backtest(df["close"], sig)
    comparison_data.append(
        {
            "Strategy": name,
            "Return": f"{res['total_return']:+.1%}",
            "Buy & Hold": f"{res['buy_and_hold']:+.1%}",
            "Sharpe": f"{res['sharpe_ratio']:.2f}",
            "Max Drawdown": f"{res['max_drawdown']:.1%}",
            "N Trades": int(res["n_trades"]),
        }
    )

st.dataframe(pd.DataFrame(comparison_data).set_index("Strategy"), use_container_width=True)

# --- SECTION 4: Portfolio weights ---
st.subheader("Optimal Portfolio Weights (Max Sharpe)")

prices = pd.DataFrame()
for ticker in ASSETS:
    d = load_ohlcv(ticker)
    prices[ticker] = d["close"]

weights = compute_weights(prices)
fig2 = px.pie(
    values=list(weights.values()),
    names=list(weights.keys()),
    color_discrete_sequence=px.colors.sequential.Blues_r,
)
fig2.update_layout(template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)

st.caption("Data: yfinance | Optimization: PyPortfolioOpt | © AlphaTrade 2026")
