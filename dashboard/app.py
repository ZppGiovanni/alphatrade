import sys
sys.path.insert(0, ".")

import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from data.database import load_ohlcv
from data.normalizer import add_indicators
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.macd_crossover import MACDCrossoverStrategy
from strategies.ml_model import MLStrategy
from portfolio.optimizer import compute_weights
from portfolio.risk import backtest

load_dotenv()
ASSETS = ["QQQ", "XLE", "GLD", "XLV", "ARKK"]

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="AlphaTrade", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-container { background: #1a1a2e; border-radius: 8px; padding: 12px; }
    div[data-testid="stMetric"] {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    .stDataFrame { border: 1px solid #0f3460; border-radius: 8px; }
    h1 { color: #4FC3F7; }
    h2, h3 { color: #e0e0e0; border-bottom: 1px solid #0f3460; padding-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

st.title("📈 AlphaTrade — Algorithmic ETF Trading System")
st.caption("USI Programming in Finance II, 2026")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
selected = st.sidebar.selectbox("Asset", ASSETS)
strategy_name = st.sidebar.selectbox(
    "Strategy", ["Momentum", "Mean Reversion", "MACD Crossover", "ML (Random Forest)"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.caption("AlphaTrade uses quantitative strategies, portfolio optimization and ML to trade 5 thematic ETFs.")

# ── Load data ─────────────────────────────────────────────────
df = load_ohlcv(selected)
df = add_indicators(df)

if strategy_name == "Mean Reversion":
    strategy = MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5})
elif strategy_name == "MACD Crossover":
    strategy = MACDCrossoverStrategy(params={})
elif strategy_name == "ML (Random Forest)":
    strategy = MLStrategy(params={"n_estimators": 100, "threshold": 0.6})
else:
    strategy = MomentumStrategy(params={"short_window": 20, "long_window": 50})

signals = strategy.generate_signals(df)

# ── SECTION 1: Price & Signals ────────────────────────────────
st.subheader(f"📊 {selected} — Price & Signals")

ret_1d = df['return_1d'].iloc[-1]
ret_5d = df['return_5d'].iloc[-1]
rsi = df['rsi'].iloc[-1]
last_close = df['close'].iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Last Close", f"${last_close:.2f}")
col2.metric("RSI", f"{rsi:.1f}", delta="Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral"))
col3.metric("1D Return", f"{ret_1d:.2%}", delta=f"{ret_1d:.2%}")
col4.metric("5D Return", f"{ret_5d:.2%}", delta=f"{ret_5d:.2%}")

# Candlestick + signals + RSI subplot
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.7, 0.3], vertical_spacing=0.05,
    subplot_titles=(f"{selected} Candlestick", "RSI"))

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["open"], high=df["high"],
    low=df["low"], close=df["close"],
    name="Price",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350",
), row=1, col=1)

# Bollinger Bands
fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
    line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
    line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
    fill="tonexty", fillcolor="rgba(255,255,255,0.03)", showlegend=False), row=1, col=1)

# Buy/Sell signals
buys = df[signals == 1]
sells = df[signals == -1]
fig.add_trace(go.Scatter(x=buys.index, y=buys["low"] * 0.99, mode="markers", name="Buy",
    marker=dict(color="#26a69a", size=10, symbol="triangle-up")), row=1, col=1)
fig.add_trace(go.Scatter(x=sells.index, y=sells["high"] * 1.01, mode="markers", name="Sell",
    marker=dict(color="#ef5350", size=10, symbol="triangle-down")), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
    line=dict(color="#CE93D8", width=1.5)), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.5)", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.5)", row=2, col=1)

fig.update_layout(
    template="plotly_dark",
    height=550,
    xaxis_rangeslider_visible=False,
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=0, r=0, t=30, b=0),
)
fig.update_yaxes(gridcolor="#1f2937")
fig.update_xaxes(gridcolor="#1f2937")
st.plotly_chart(fig, use_container_width=True)

# ── SECTION 2: Backtest ───────────────────────────────────────
st.subheader(f"📈 {selected} — Backtest Results")

results = backtest(df["close"], signals)
equity_curve = results["equity_curve"]
total_ret = results['total_return']
bh_ret = results['buy_and_hold']
sharpe = results['sharpe_ratio']
drawdown = results['max_drawdown']

col1, col2, col3, col4 = st.columns(4)
col1.metric("Strategy Return", f"{total_ret:+.1%}", delta=f"{total_ret - bh_ret:+.1%} vs B&H")
col2.metric("Buy & Hold", f"{bh_ret:+.1%}")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}", delta="Good" if sharpe > 1 else "Low")
col4.metric("Max Drawdown", f"{drawdown:.1%}")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, name="Strategy",
    line=dict(color="#4FC3F7", width=2), fill="tozeroy", fillcolor="rgba(79,195,247,0.08)"))
bh_curve = 10000 * (df["close"] / df["close"].iloc[0])
fig3.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve.values, name="Buy & Hold",
    line=dict(color="#FFB74D", width=1.5, dash="dash")))
fig3.update_layout(
    template="plotly_dark", height=320,
    yaxis_title="Portfolio Value ($)", xaxis_title="",
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
fig3.update_yaxes(gridcolor="#1f2937")
fig3.update_xaxes(gridcolor="#1f2937")
st.plotly_chart(fig3, use_container_width=True)

# ── SECTION 3: Strategy Comparison ───────────────────────────
st.subheader(f"⚖️ {selected} — Strategy Comparison")

strategies_map = {
    "Momentum": MomentumStrategy(params={"short_window": 20, "long_window": 50}),
    "Mean Reversion": MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5}),
    "MACD Crossover": MACDCrossoverStrategy(params={}),
}

comparison_data = []
for name, strat in strategies_map.items():
    sig = strat.generate_signals(df)
    res = backtest(df["close"], sig)
    comparison_data.append({
        "Strategy": name,
        "Return": f"{res['total_return']:+.1%}",
        "Buy & Hold": f"{res['buy_and_hold']:+.1%}",
        "Sharpe": f"{res['sharpe_ratio']:.2f}",
        "Max Drawdown": f"{res['max_drawdown']:.1%}",
        "N Trades": int(res["n_trades"]),
    })

st.dataframe(pd.DataFrame(comparison_data).set_index("Strategy"), use_container_width=True)

# ── SECTION 4: Portfolio Weights ──────────────────────────────
st.subheader("🥧 Optimal Portfolio Weights (Max Sharpe)")

prices = pd.DataFrame()
for ticker in ASSETS:
    d = load_ohlcv(ticker)
    prices[ticker] = d["close"]

weights = compute_weights(prices)

col1, col2 = st.columns([1, 1])
with col1:
    fig2 = px.pie(values=list(weights.values()), names=list(weights.keys()),
        color_discrete_sequence=["#4FC3F7", "#26a69a", "#FFB74D", "#CE93D8", "#ef5350"])
    fig2.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
        margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    weight_data = [{"Asset": k, "Weight": f"{v:.1%}"} for k, v in weights.items()]
    st.dataframe(pd.DataFrame(weight_data).set_index("Asset"), use_container_width=True)
    st.caption("Markowitz Mean-Variance Optimization — Max Sharpe Ratio")

# ── SECTION 5: Consensus Signal ───────────────────────────────
st.subheader(f"🎯 {selected} — Consensus Signal (Ensemble)")

sig_momentum = MomentumStrategy(params={"short_window": 20, "long_window": 50}).generate_signals(df)
sig_mean_rev = MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5}).generate_signals(df)
sig_macd = MACDCrossoverStrategy(params={}).generate_signals(df)

consensus = sig_momentum + sig_mean_rev + sig_macd
latest_consensus = int(consensus.iloc[-1])

if latest_consensus >= 2:
    signal_label = "🟢 STRONG BUY"
    signal_color = "#26a69a"
elif latest_consensus == 1:
    signal_label = "🟡 WEAK BUY"
    signal_color = "#FFB74D"
elif latest_consensus == 0:
    signal_label = "⚪ HOLD"
    signal_color = "#9e9e9e"
elif latest_consensus == -1:
    signal_label = "🟠 WEAK SELL"
    signal_color = "#FF8C00"
else:
    signal_label = "🔴 STRONG SELL"
    signal_color = "#ef5350"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Momentum", int(sig_momentum.iloc[-1]))
col2.metric("Mean Reversion", int(sig_mean_rev.iloc[-1]))
col3.metric("MACD Crossover", int(sig_macd.iloc[-1]))
col4.metric("Consensus Score", latest_consensus)

st.markdown(f"""
<div style="background:{signal_color}22; border-left: 4px solid {signal_color};
border-radius: 8px; padding: 16px 24px; margin: 8px 0;">
<span style="font-size:1.4rem; font-weight:700; color:{signal_color};">
Current Signal: {signal_label}
</span>
</div>
""", unsafe_allow_html=True)

fig_cons = go.Figure()
fig_cons.add_trace(go.Scatter(x=consensus.index, y=consensus.values, name="Consensus",
    line=dict(color="#CE93D8", width=2), fill="tozeroy", fillcolor="rgba(206,147,216,0.1)"))
fig_cons.add_hline(y=2, line_dash="dash", line_color="#26a69a", annotation_text="STRONG BUY")
fig_cons.add_hline(y=-2, line_dash="dash", line_color="#ef5350", annotation_text="STRONG SELL")
fig_cons.add_hline(y=0, line_dash="dot", line_color="#9e9e9e")
fig_cons.update_layout(
    template="plotly_dark", height=280,
    yaxis_title="Consensus Score", xaxis_title="",
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    yaxis=dict(tickvals=[-3, -2, -1, 0, 1, 2, 3]),
    margin=dict(l=0, r=0, t=10, b=0),
)
fig_cons.update_yaxes(gridcolor="#1f2937")
fig_cons.update_xaxes(gridcolor="#1f2937")
st.plotly_chart(fig_cons, use_container_width=True)

# ── SECTION 6: AI Market Analysis ────────────────────────────
st.subheader(f"🤖 {selected} — AI Market Analysis")

if st.button("⚡ Generate AI Analysis", type="primary"):
    with st.spinner("Analyzing market data with AI..."):
        recent = df.tail(10)[["close", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower"]].round(2)

        prompt = f"""You are a quantitative financial analyst. Analyze the following market data for {selected} ETF and provide a concise professional analysis.

Current market data (last 10 days):
{recent.to_string()}

Latest metrics:
- Last Close: ${df['close'].iloc[-1]:.2f}
- RSI: {df['rsi'].iloc[-1]:.1f}
- 1D Return: {df['return_1d'].iloc[-1]:.2%}
- 5D Return: {df['return_5d'].iloc[-1]:.2%}
- Consensus Signal Score: {latest_consensus} (range -3 to +3)
- Signal: {signal_label}

Strategy backtest results:
- Momentum Return: {backtest(df['close'], sig_momentum)['total_return']:+.1%}
- Mean Reversion Return: {backtest(df['close'], sig_mean_rev)['total_return']:+.1%}
- MACD Crossover Return: {backtest(df['close'], sig_macd)['total_return']:+.1%}
- Buy & Hold Return: {backtest(df['close'], sig_momentum)['buy_and_hold']:+.1%}

Provide:
1. Current market conditions assessment
2. Signal interpretation
3. Risk considerations
4. Brief outlook

Keep it concise and professional. Max 200 words."""

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        message = client.chat.completions.create(
            model="llama-3.3-70b-versatile", max_tokens=1024,
            messages=[{"role": "user", "content": prompt}])

        st.markdown("#### 📋 Analysis")
        st.info(message.choices[0].message.content)

st.markdown("---")
st.caption("Data: yfinance · Optimization: PyPortfolioOpt · AI: Groq LLaMA 3.3 · © AlphaTrade 2026")
