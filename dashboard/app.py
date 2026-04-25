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
from portfolio.optimizer import compute_weights

ASSETS = ["QQQ", "XLE", "GLD", "XLV", "ARKK"]

st.set_page_config(page_title="AlphaTrade", page_icon="📈", layout="wide")
st.title("📈 AlphaTrade — Algorithmic ETF Trading System")
st.caption("USI Programming in Finance II, 2026")

# Sidebar
st.sidebar.header("Settings")
selected = st.sidebar.selectbox("Asset", ASSETS)
strategy_name = st.sidebar.selectbox("Strategy", ["Momentum", "Mean Reversion"])

# Load data
df = load_ohlcv(selected)
df = add_indicators(df)

# Price chart
st.subheader(f"{selected} — Price & Signals")
if strategy_name == "Mean Reversion":
    strategy = MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5})
else:
    strategy = MomentumStrategy(params={"short_window": 20, "long_window": 50})

signals = strategy.generate_signals(df)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(color="#4FC3F7")))
buys = df[signals == 1]
sells = df[signals == -1]
fig.add_trace(go.Scatter(x=buys.index, y=buys["close"], mode="markers",
    name="Buy", marker=dict(color="green", size=8, symbol="triangle-up")))
fig.add_trace(go.Scatter(x=sells.index, y=sells["close"], mode="markers",
    name="Sell", marker=dict(color="red", size=8, symbol="triangle-down")))
fig.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig, use_container_width=True)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Last Close", f"${df['close'].iloc[-1]:.2f}")
col2.metric("RSI", f"{df['rsi'].iloc[-1]:.1f}")
col3.metric("1D Return", f"{df['return_1d'].iloc[-1]:.2%}")
col4.metric("5D Return", f"{df['return_5d'].iloc[-1]:.2%}")

# Portfolio weights
st.subheader("Optimal Portfolio Weights (Max Sharpe)")
prices = pd.DataFrame()
for ticker in ASSETS:
    d = load_ohlcv(ticker)
    prices[ticker] = d["close"]

weights = compute_weights(prices)
fig2 = px.pie(values=list(weights.values()), names=list(weights.keys()),
    color_discrete_sequence=px.colors.sequential.Blues_r)
fig2.update_layout(template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)

st.caption("Data: yfinance | Optimization: PyPortfolioOpt | © AlphaTrade 2026")