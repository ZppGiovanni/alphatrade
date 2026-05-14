import sys
sys.path.insert(0, ".")

import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
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
# Auto-download data if database is empty
from data.database import init_db
from data.fetcher import fetch_all
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "alphatrade.db"

def _ensure_data():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    conn.close()
    if count == 0:
        with st.spinner("📥 First run — downloading market data (30s)…"):
            fetch_all(period="5y")

_ensure_data()

# ── Palette ───────────────────────────────────────────────────
C = {
    "bg":      "#0e1117",
    "surface": "#161b27",
    "grid":    "#1f2937",
    "border":  "#0f3460",
    "green":   "#26a69a",
    "red":     "#ef5350",
    "blue":    "#4FC3F7",
    "amber":   "#FFB74D",
    "purple":  "#CE93D8",
    "orange":  "#FF8C00",
    "grey":    "#9e9e9e",
}

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="AlphaTrade", page_icon="📈", layout="wide")

st.markdown(f"""
<style>
    .block-container {{ padding-top: 1.5rem; }}

    /* Base metric card */
    div[data-testid="stMetric"] {{
        background: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 8px;
        padding: 12px 16px;
        border-top: 3px solid {C['grey']};
    }}
    div[data-testid="stMetricValue"] {{ font-size: 1.6rem; font-weight: 700; }}

    /* Metric top border by delta direction */
    div[data-testid="stMetric"]:has([data-testid="stMetricDeltaIcon-Up"]) {{
        border-top: 3px solid {C['green']} !important;
    }}
    div[data-testid="stMetric"]:has([data-testid="stMetricDeltaIcon-Down"]) {{
        border-top: 3px solid {C['red']} !important;
    }}

    .stDataFrame {{ border: 1px solid {C['border']}; border-radius: 8px; }}
    h1 {{ color: {C['blue']}; }}
    h2, h3 {{ color: #e0e0e0; border-bottom: 1px solid {C['border']}; padding-bottom: 6px; }}

    /* Section header */
    .section-header {{
        background: linear-gradient(90deg, {C['border']}55, transparent);
        border-left: 4px solid {C['blue']};
        border-radius: 0 6px 6px 0;
        padding: 8px 16px;
        margin: 20px 0 10px 0;
    }}
    .section-header span {{
        color: #e0e0e0;
        font-size: 1.1rem;
        font-weight: 600;
    }}

    /* Badge pills */
    .badge {{
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 4px;
    }}
    .badge-green {{ background: {C['green']}25; color: {C['green']}; border: 1px solid {C['green']}55; }}
    .badge-red   {{ background: {C['red']}25;   color: {C['red']};   border: 1px solid {C['red']}55; }}
    .badge-blue  {{ background: {C['blue']}25;  color: {C['blue']};  border: 1px solid {C['blue']}55; }}
    .badge-amber {{ background: {C['amber']}25; color: {C['amber']}; border: 1px solid {C['amber']}55; }}

    /* AI analysis card */
    .ai-card {{
        background: {C['surface']};
        border: 1px solid {C['border']};
        border-left: 4px solid {C['blue']};
        border-radius: 10px;
        padding: 20px 24px;
        margin-top: 14px;
        line-height: 1.7;
        color: #e0e0e0;
    }}
    .ai-card-header {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {C['blue']};
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid {C['border']};
    }}

    /* AI button gradient */
    [data-testid="stBaseButton-primary"] {{
        background: linear-gradient(135deg, #1a237e 0%, {C['blue']} 100%) !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.04em !important;
        box-shadow: 0 4px 18px rgba(79,195,247,0.25) !important;
        transition: box-shadow 0.2s ease !important;
    }}
    [data-testid="stBaseButton-primary"]:hover {{
        box-shadow: 0 6px 24px rgba(79,195,247,0.45) !important;
    }}

    /* Sidebar brand */
    .sidebar-brand {{
        text-align: center;
        padding: 12px 4px 14px 4px;
        border-bottom: 1px solid {C['border']};
        margin-bottom: 12px;
    }}
    .sidebar-brand .brand-title {{
        color: {C['blue']};
        font-size: 1.55rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }}
    .sidebar-brand .brand-sub {{
        color: {C['grey']};
        font-size: 0.78rem;
        margin: 3px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    /* Sidebar timestamp */
    .sidebar-ts {{
        color: {C['grey']};
        font-size: 0.75rem;
        padding: 4px 0;
        line-height: 1.8;
    }}

    /* Footer */
    .footer-bar {{
        margin-top: 24px;
        padding-top: 14px;
        border-top: 1px solid {C['border']};
        color: {C['grey']};
        font-size: 0.78rem;
        text-align: center;
        letter-spacing: 0.03em;
    }}
</style>
""", unsafe_allow_html=True)

st.title("📈 AlphaTrade — Algorithmic ETF Trading System")
st.caption("USI Programming in Finance II, 2026")


def _dark(**extra) -> dict:
    base = dict(
        template="plotly_dark",
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        font=dict(color="#e0e0e0"),
    )
    base.update(extra)
    return base


def _style_axes(fig: go.Figure):
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False)
    fig.update_xaxes(gridcolor=C["grid"], showgrid=False)


def _section_header(text: str, color: str = None):
    border_color = color or C["blue"]
    st.markdown(
        f'<div class="section-header" style="border-left-color:{border_color};">'
        f'<span>{text}</span></div>',
        unsafe_allow_html=True,
    )



@st.cache_data(ttl=300)
def _load(ticker: str) -> pd.DataFrame:
    return add_indicators(load_ohlcv(ticker))


@st.cache_data(ttl=300)
def _load_prices() -> pd.DataFrame:
    return pd.DataFrame({t: load_ohlcv(t)["close"] for t in ASSETS})


# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="brand-title">📈 AlphaTrade</div>
    <div class="brand-sub">ETF Trading System</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ Settings")
selected = st.sidebar.selectbox("📌 Asset", ASSETS)
strategy_name = st.sidebar.selectbox(
    "🧠 Strategy", ["Momentum", "Mean Reversion", "MACD Crossover", "ML (Random Forest)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Date Range")
_df_full = _load(selected)
_idx = pd.to_datetime(_df_full.index)
_min_date, _max_date = _idx.min().date(), _idx.max().date()
date_range = st.sidebar.date_input(
    "From / To",
    value=(_min_date, _max_date),
    min_value=_min_date,
    max_value=_max_date,
)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = str(date_range[0]), str(date_range[1])
else:
    start_date, end_date = str(_min_date), str(_max_date)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.caption("AlphaTrade uses quantitative strategies, portfolio optimization and ML to trade 5 thematic ETFs.")

# Sidebar bottom: current date + last data update
from datetime import date as _date_cls
_today_str  = _date_cls.today().strftime("%B %d, %Y")
_update_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<div class="sidebar-ts">📆 <b>Today:</b> {_today_str}<br>'
    f'🔄 <b>Last update:</b> {_update_str}</div>',
    unsafe_allow_html=True,
)

# ── Load data (with date filter) ─────────────────────────────
with st.spinner(f"Loading {selected} market data…"):
    df = _load(selected)
    df = df[(df.index >= start_date) & (df.index <= end_date)]

if strategy_name == "Mean Reversion":
    strategy = MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5})
elif strategy_name == "MACD Crossover":
    strategy = MACDCrossoverStrategy(params={})
elif strategy_name == "ML (Random Forest)":
    strategy = MLStrategy(params={"n_estimators": 100, "threshold": 0.6})
else:
    strategy = MomentumStrategy(params={"short_window": 20, "long_window": 50})

with st.spinner(f"Computing {strategy_name} signals…"):
    signals = strategy.generate_signals(df)

# ── SECTION 1: Price & Signals ────────────────────────────────
_section_header(f"📊 {selected} — Price & Signals")

ret_1d = df['return_1d'].iloc[-1]
ret_5d = df['return_5d'].iloc[-1]
rsi = df['rsi'].iloc[-1]
last_close = df['close'].iloc[-1]


col1, col2, col3, col4 = st.columns(4)

# Last Close — custom HTML card + 7-day sparkline
_close_color = C["green"] if ret_1d >= 0 else C["red"]
with col1:
    st.markdown(
        f'<div style="background:{C["surface"]};border:1px solid {C["border"]};'
        f'border-top:3px solid {_close_color};border-radius:8px;padding:12px 16px 6px 16px;">'
        f'<div style="color:{C["grey"]};font-size:0.82rem;margin-bottom:2px;">Last Close</div>'
        f'<div style="font-size:1.6rem;font-weight:700;color:#e0e0e0;">${last_close:.2f}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# RSI — custom HTML card
_rsi_color = C["red"] if rsi > 70 else (C["green"] if rsi < 30 else C["grey"])
_rsi_label = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
with col2:
    st.markdown(
        f'<div style="background:{C["surface"]};border:1px solid {C["border"]};'
        f'border-top:3px solid {_rsi_color};border-radius:8px;padding:12px 16px;">'
        f'<div style="color:{C["grey"]};font-size:0.82rem;margin-bottom:2px;">RSI</div>'
        f'<div style="font-size:1.6rem;font-weight:700;color:#e0e0e0;">{rsi:.1f}</div>'
        f'<div style="font-size:0.78rem;color:{_rsi_color};margin-top:2px;">{_rsi_label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# 1D Return — custom HTML card + 7-day sparkline
_ret1d_color = C["green"] if ret_1d >= 0 else C["red"]
with col3:
    st.markdown(
        f'<div style="background:{C["surface"]};border:1px solid {C["border"]};'
        f'border-top:3px solid {_ret1d_color};border-radius:8px;padding:12px 16px 6px 16px;">'
        f'<div style="color:{C["grey"]};font-size:0.82rem;margin-bottom:2px;">1D Return</div>'
        f'<div style="font-size:1.6rem;font-weight:700;color:{_ret1d_color};">{ret_1d:.2%}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# 5D Return — custom HTML card
_ret5d_color = C["green"] if ret_5d >= 0 else C["red"]
with col4:
    st.markdown(
        f'<div style="background:{C["surface"]};border:1px solid {C["border"]};'
        f'border-top:3px solid {_ret5d_color};border-radius:8px;padding:12px 16px;">'
        f'<div style="color:{C["grey"]};font-size:0.82rem;margin-bottom:2px;">5D Return</div>'
        f'<div style="font-size:1.6rem;font-weight:700;color:{_ret5d_color};">{ret_5d:.2%}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Candlestick | Volume | RSI  (3 rows)
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.58, 0.18, 0.24],
    vertical_spacing=0.03,
    subplot_titles=(f"{selected} Candlestick + SMA", "Volume", "RSI (14)"),
)

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["open"], high=df["high"],
    low=df["low"], close=df["close"],
    name="Price",
    increasing_line_color=C["green"],
    decreasing_line_color=C["red"],
    increasing_fillcolor=C["green"],
    decreasing_fillcolor=C["red"],
), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
    line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
    line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
    fill="tonexty", fillcolor="rgba(255,255,255,0.03)", showlegend=False), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], name="SMA 20",
    line=dict(color=C["amber"], width=1.4)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50",
    line=dict(color=C["purple"], width=1.4)), row=1, col=1)

buys = df[signals == 1]
sells = df[signals == -1]
fig.add_trace(go.Scatter(x=buys.index, y=buys["low"] * 0.985, mode="markers", name="Buy",
    marker=dict(color=C["green"], size=9, symbol="triangle-up", line=dict(color="#fff", width=0.5))), row=1, col=1)
fig.add_trace(go.Scatter(x=sells.index, y=sells["high"] * 1.015, mode="markers", name="Sell",
    marker=dict(color=C["red"], size=9, symbol="triangle-down", line=dict(color="#fff", width=0.5))), row=1, col=1)

vol_colors = [C["green"] if c >= o else C["red"]
              for c, o in zip(df["close"], df["open"])]
fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume",
    marker_color=vol_colors, opacity=0.7, showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(
    x=list(df.index) + list(df.index[::-1]),
    y=[70] * len(df) + [30] * len(df),
    fill="toself",
    fillcolor="rgba(255,255,255,0.04)",
    line=dict(color="rgba(0,0,0,0)"),
    showlegend=False, name="RSI Zone",
), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
    line=dict(color=C["purple"], width=1.5)), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.5)", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.5)", row=3, col=1)
fig.add_hline(y=50, line_dash="dot",  line_color="rgba(255,255,255,0.15)", row=3, col=1)

fig.update_layout(**_dark(height=680, xaxis_rangeslider_visible=False))
fig.update_layout(xaxis3=dict(rangeslider_visible=False))
_style_axes(fig)
st.plotly_chart(fig, width="stretch")

# ── SECTION 2: MACD ──────────────────────────────────────────
_section_header(f"📉 {selected} — MACD")

macd_hist = df["macd"] - df["macd_signal"]
hist_colors = [C["green"] if v >= 0 else C["red"] for v in macd_hist]

fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.55, 0.45], vertical_spacing=0.05,
    subplot_titles=("MACD Line & Signal", "Histogram"))

fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
    line=dict(color=C["blue"], width=1.8)), row=1, col=1)
fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
    line=dict(color=C["orange"], width=1.4, dash="dash")), row=1, col=1)
fig_macd.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=1, col=1)

fig_macd.add_trace(go.Bar(x=df.index, y=macd_hist, name="Histogram",
    marker_color=hist_colors, opacity=0.85, showlegend=False), row=2, col=1)
fig_macd.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)

fig_macd.update_layout(**_dark(height=380))
_style_axes(fig_macd)
st.plotly_chart(fig_macd, width="stretch")

# ── SECTION 3: Backtest ───────────────────────────────────────
_section_header(f"📈 {selected} — Backtest Results", color=C["green"])

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

bh_curve = 10000 * (df["close"] / df["close"].iloc[0])
rolling_max = equity_curve.cummax()
dd_series = (equity_curve - rolling_max) / rolling_max * 100

fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.65, 0.35], vertical_spacing=0.04,
    subplot_titles=("Portfolio Value", "Drawdown (%)"))

fig3.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, name="Strategy",
    line=dict(color=C["blue"], width=2),
    fill="tozeroy", fillcolor="rgba(79,195,247,0.07)"), row=1, col=1)
fig3.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve.values, name="Buy & Hold",
    line=dict(color=C["amber"], width=1.5, dash="dash")), row=1, col=1)

fig3.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values, name="Drawdown",
    line=dict(color=C["red"], width=1.5),
    fill="tozeroy", fillcolor="rgba(239,83,80,0.15)",
    showlegend=False), row=2, col=1)

fig3.update_layout(**_dark(height=480, yaxis_title="Portfolio Value ($)", yaxis2_title="DD (%)"))
_style_axes(fig3)
st.plotly_chart(fig3, width="stretch")

# ── SECTION 4: Strategy Comparison ───────────────────────────
_section_header(f"⚖️ {selected} — Strategy Comparison")

strategies_map = {
    "Momentum":      MomentumStrategy(params={"short_window": 20, "long_window": 50}),
    "Mean Reversion": MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5}),
    "MACD Crossover": MACDCrossoverStrategy(params={}),
}

with st.spinner("Running strategy comparison…"):
    comparison_data = []
    for name, strat in strategies_map.items():
        sig = strat.generate_signals(df)
        res = backtest(df["close"], sig)
        comparison_data.append({
            "Strategy":    name,
            "Return":      res['total_return'],
            "Buy & Hold":  res['buy_and_hold'],
            "Sharpe":      res['sharpe_ratio'],
            "Max Drawdown": res['max_drawdown'],
            "N Trades":    int(res["n_trades"]),
        })

cdf = pd.DataFrame(comparison_data).set_index("Strategy")

col_table, col_chart = st.columns([1, 1])
with col_table:
    display = cdf.copy()
    display["Return"] = display["Return"].map("{:+.1%}".format)
    display["Buy & Hold"] = display["Buy & Hold"].map("{:+.1%}".format)
    display["Sharpe"] = display["Sharpe"].map("{:.2f}".format)
    display["Max Drawdown"] = display["Max Drawdown"].map("{:.1%}".format)
    st.dataframe(display, width="stretch")

with col_chart:
    bar_colors = [C["green"] if v >= 0 else C["red"] for v in cdf["Return"]]
    fig_bar = go.Figure(go.Bar(
        x=cdf.index,
        y=cdf["Return"] * 100,
        marker_color=bar_colors,
        text=[f"{v:+.1%}" for v in cdf["Return"]],
        textposition="outside",
    ))
    fig_bar.add_hline(y=cdf["Buy & Hold"].iloc[0] * 100,
        line_dash="dash", line_color=C["amber"],
        annotation_text="B&H", annotation_position="top right")
    fig_bar.update_layout(**_dark(height=280,
        yaxis_title="Return (%)", showlegend=False, margin=dict(l=0, r=0, t=30, b=40)))
    _style_axes(fig_bar)
    st.plotly_chart(fig_bar, width="stretch")

# ── SECTION 5: Portfolio Weights ──────────────────────────────
_section_header("🥧 Optimal Portfolio Weights (Max Sharpe)")

with st.spinner("Optimizing portfolio weights…"):
    prices = _load_prices()
    weights = compute_weights(prices)

w_values = list(weights.values())
w_names  = list(weights.keys())
COLORS   = [C["blue"], C["green"], C["amber"], C["purple"], C["red"]]

col1, col2 = st.columns([1, 1])
with col1:
    fig2 = go.Figure(go.Pie(
        values=w_values, labels=w_names,
        hole=0.45,
        marker=dict(colors=COLORS, line=dict(color=C["bg"], width=2)),
        textinfo="label+percent",
        insidetextorientation="radial",
    ))
    fig2.update_layout(**_dark(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        annotations=[dict(text="Weights", x=0.5, y=0.5, font_size=13,
                          showarrow=False, font_color="#e0e0e0")],
    ))
    st.plotly_chart(fig2, width="stretch")

with col2:
    fig_wb = go.Figure(go.Bar(
        x=w_values, y=w_names,
        orientation="h",
        marker_color=COLORS,
        text=[f"{v:.1%}" for v in w_values],
        textposition="outside",
    ))
    fig_wb.update_layout(**_dark(
        height=320, showlegend=False,
        xaxis_tickformat=".0%",
        margin=dict(l=0, r=60, t=10, b=0),
    ))
    _style_axes(fig_wb)
    st.plotly_chart(fig_wb, width="stretch")

st.caption("Markowitz Mean-Variance Optimization — Max Sharpe Ratio")

# ── SECTION 6: Consensus Signal ───────────────────────────────
_section_header(f"🎯 {selected} — Consensus Signal (Ensemble)")

sig_momentum = MomentumStrategy(params={"short_window": 20, "long_window": 50}).generate_signals(df)
sig_mean_rev = MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5}).generate_signals(df)
sig_macd     = MACDCrossoverStrategy(params={}).generate_signals(df)

consensus = sig_momentum + sig_mean_rev + sig_macd
latest_consensus = int(consensus.iloc[-1])

if latest_consensus >= 2:
    signal_label = "🟢 STRONG BUY";   signal_color = C["green"]
elif latest_consensus == 1:
    signal_label = "🟡 WEAK BUY";     signal_color = C["amber"]
elif latest_consensus == 0:
    signal_label = "⚪ HOLD";          signal_color = C["grey"]
elif latest_consensus == -1:
    signal_label = "🟠 WEAK SELL";    signal_color = C["orange"]
else:
    signal_label = "🔴 STRONG SELL";  signal_color = C["red"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Momentum",       int(sig_momentum.iloc[-1]))
col2.metric("Mean Reversion", int(sig_mean_rev.iloc[-1]))
col3.metric("MACD Crossover", int(sig_macd.iloc[-1]))
col4.metric("Consensus Score", latest_consensus)

# Prominent signal card with horizontal gauge
_gauge_pct = (latest_consensus + 3) / 6 * 100
st.markdown(f"""
<div style="background:{signal_color}12; border:2px solid {signal_color}44;
     border-radius:14px; padding:24px 28px; margin:14px 0;">
    <div style="text-align:center; color:{C['grey']}; font-size:0.8rem;
         text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
        Current Ensemble Signal
    </div>
    <div style="text-align:center; font-size:2rem; font-weight:800;
         color:{signal_color}; margin-bottom:20px;">
        {signal_label}
    </div>
    <div style="max-width:540px; margin:0 auto;">
        <div style="display:flex; justify-content:space-between;
             margin-bottom:6px; font-size:0.72rem; font-weight:600;">
            <span style="color:{C['red']};">−3</span>
            <span style="color:{C['grey']};">STRONG SELL</span>
            <span style="color:{C['grey']};">HOLD</span>
            <span style="color:{C['green']};">STRONG BUY</span>
            <span style="color:{C['green']};">+3</span>
        </div>
        <div style="background:linear-gradient(90deg,{C['red']},{C['grey']} 50%,{C['green']});
             border-radius:10px; height:16px; position:relative;">
            <div style="position:absolute; left:{_gauge_pct:.1f}%; top:50%;
                 transform:translate(-50%,-50%); width:22px; height:22px;
                 border-radius:50%; background:{signal_color};
                 border:3px solid #fff;
                 box-shadow:0 0 12px {signal_color};"></div>
        </div>
        <div style="text-align:center; margin-top:8px; color:{C['grey']}; font-size:0.8rem;">
            Consensus score: <strong style="color:{signal_color};">{latest_consensus:+d}</strong>
            &nbsp;/&nbsp; range −3 to +3
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

bar_col = [C["green"] if v > 0 else (C["red"] if v < 0 else C["grey"])
           for v in consensus.values]

fig_cons = go.Figure()
fig_cons.add_trace(go.Bar(
    x=consensus.index, y=consensus.values,
    marker_color=bar_col, opacity=0.85,
    name="Consensus", showlegend=False,
))
for level, color, label in [
    (2, C["green"], "STRONG BUY"),
    (-2, C["red"],  "STRONG SELL"),
]:
    fig_cons.add_hline(y=level, line_dash="dash", line_color=color,
        annotation_text=label, annotation_position="top right",
        annotation_font_color=color)
fig_cons.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")

fig_cons.update_layout(**_dark(
    height=290,
    yaxis=dict(tickvals=[-3, -2, -1, 0, 1, 2, 3], title="Consensus Score"),
))
_style_axes(fig_cons)
st.plotly_chart(fig_cons, width="stretch")

# ── SECTION 7: Correlation Heatmap ───────────────────────────
_section_header("🔥 Asset Correlation Matrix")

returns_all = prices.pct_change().dropna()
corr = returns_all.corr()
COLORS_LIST = [C["blue"], C["green"], C["amber"], C["purple"], C["red"]]

fig_corr = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale=[
        [0.0, C["red"]],
        [0.5, C["bg"]],
        [1.0, C["green"]],
    ],
    zmin=-1, zmax=1,
    text=[[f"{v:.2f}" for v in row] for row in corr.values],
    texttemplate="%{text}",
    textfont=dict(size=13, color="#e0e0e0"),
    showscale=True,
    colorbar=dict(
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=["-1", "-0.5", "0", "0.5", "1"],
        thickness=14,
    ),
))
fig_corr.update_layout(**_dark(
    height=380,
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(side="bottom"),
))
st.plotly_chart(fig_corr, width="stretch")

# ── SECTION 8: Returns Distribution ──────────────────────────
_section_header(f"📊 {selected} — Daily Returns Distribution")

daily_ret = df["return_1d"].dropna() * 100
mean_r = daily_ret.mean()
std_r  = daily_ret.std()

fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=daily_ret,
    nbinsx=60,
    name="Daily Returns",
    marker_color=C["blue"],
    opacity=0.75,
))
fig_dist.add_vline(x=mean_r, line_dash="dash", line_color=C["amber"],
    annotation_text=f"Mean: {mean_r:.2f}%",
    annotation_position="top right", annotation_font_color=C["amber"])
fig_dist.add_vline(x=mean_r - 2 * std_r, line_dash="dot", line_color=C["red"],
    annotation_text="−2σ", annotation_position="top left",
    annotation_font_color=C["red"])
fig_dist.add_vline(x=mean_r + 2 * std_r, line_dash="dot", line_color=C["green"],
    annotation_text="+2σ", annotation_position="top right",
    annotation_font_color=C["green"])
fig_dist.update_layout(**_dark(
    height=320,
    xaxis_title="Daily Return (%)",
    yaxis_title="Frequency",
    bargap=0.05,
))
_style_axes(fig_dist)
st.plotly_chart(fig_dist, width="stretch")

# ── SECTION 9: Rolling Sharpe Ratio ──────────────────────────
_section_header(f"📐 {selected} — Rolling Sharpe Ratio (63-day)")

equity_curve = backtest(df["close"], signals)["equity_curve"]
roll_ret  = equity_curve.pct_change().dropna()
roll_mean = roll_ret.rolling(63).mean()
roll_std  = roll_ret.rolling(63).std().replace(0, float("nan"))
roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)

fig_sharpe = go.Figure()
fig_sharpe.add_trace(go.Scatter(
    x=roll_sharpe.index, y=roll_sharpe.values,
    name="Rolling Sharpe",
    line=dict(color=C["blue"], width=1.8),
    fill="tozeroy",
    fillcolor="rgba(79,195,247,0.07)",
))
fig_sharpe.add_hline(y=1,  line_dash="dash", line_color=C["green"],
    annotation_text="Good (1.0)", annotation_position="top right",
    annotation_font_color=C["green"])
fig_sharpe.add_hline(y=0,  line_dash="dot",  line_color="rgba(255,255,255,0.2)")
fig_sharpe.add_hline(y=-1, line_dash="dash", line_color=C["red"],
    annotation_text="Poor (−1.0)", annotation_position="bottom right",
    annotation_font_color=C["red"])
fig_sharpe.update_layout(**_dark(
    height=300,
    yaxis_title="Sharpe Ratio",
))
_style_axes(fig_sharpe)
st.plotly_chart(fig_sharpe, width="stretch")

# ── SECTION 10: Rolling Volatility ───────────────────────────
_section_header(f"🌊 {selected} — Rolling Volatility (21-day)")

roll_vol = df["return_1d"].rolling(21).std() * np.sqrt(252) * 100
current_vol = roll_vol.iloc[-1]
vol_color = C["red"] if current_vol > 25 else (C["amber"] if current_vol > 15 else C["green"])

col1, col2, col3 = st.columns(3)
col1.metric("Current Ann. Vol", f"{current_vol:.1f}%")
col2.metric("Max Vol (period)",  f"{roll_vol.max():.1f}%")
col3.metric("Min Vol (period)",  f"{roll_vol.min():.1f}%")

fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(
    x=roll_vol.index, y=roll_vol.values,
    name="Ann. Volatility (%)",
    line=dict(color=C["amber"], width=1.8),
    fill="tozeroy", fillcolor="rgba(255,183,77,0.08)",
))
fig_vol.add_hline(y=25, line_dash="dash", line_color=C["red"],
    annotation_text="High (25%)", annotation_position="top right",
    annotation_font_color=C["red"])
fig_vol.add_hline(y=15, line_dash="dot", line_color=C["amber"],
    annotation_text="Mid (15%)", annotation_position="top right",
    annotation_font_color=C["amber"])
fig_vol.update_layout(**_dark(height=300, yaxis_title="Volatility (%)"))
_style_axes(fig_vol)
st.plotly_chart(fig_vol, width="stretch")

# ── SECTION 11: Monthly Returns Heatmap ──────────────────────
_section_header(f"📅 {selected} — Monthly Returns Heatmap")

_dt_index = pd.to_datetime(df.index)
_ret_series = pd.Series(df["return_1d"].values, index=_dt_index)
monthly = _ret_series.resample("ME").apply(lambda x: (1 + x).prod() - 1)
monthly_df = pd.DataFrame({
    "year":  monthly.index.year,
    "month": monthly.index.month,
    "ret":   monthly.values,
})
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
pivot = monthly_df.pivot(index="year", columns="month", values="ret")
pivot.columns = [MONTH_LABELS[m - 1] for m in pivot.columns]

text_vals = [[f"{v:.1%}" if not pd.isna(v) else "" for v in row] for row in pivot.values]
fig_cal = go.Figure(go.Heatmap(
    z=pivot.values * 100,
    x=pivot.columns.tolist(),
    y=[str(y) for y in pivot.index.tolist()],
    colorscale=[
        [0.0,  C["red"]],
        [0.45, "#2d2d2d"],
        [0.5,  C["bg"]],
        [0.55, "#1a2e1a"],
        [1.0,  C["green"]],
    ],
    zmid=0,
    text=text_vals,
    texttemplate="%{text}",
    textfont=dict(size=11),
    showscale=True,
    colorbar=dict(title="Return %", thickness=14),
))
fig_cal.update_layout(**_dark(
    height=max(220, 60 + len(pivot) * 38),
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(side="top"),
))
st.plotly_chart(fig_cal, width="stretch")

# ── SECTION 12: Trade Log ─────────────────────────────────────
_section_header(f"📋 {selected} — Trade Log ({strategy_name})")

_trades = []
_entry_date, _entry_price = None, None
for date, price in df["close"].items():
    sig = signals.get(date, 0)
    if sig == 1 and _entry_date is None:
        _entry_date, _entry_price = date, price
    elif sig == -1 and _entry_date is not None:
        _pnl = (price - _entry_price) / _entry_price
        _days = (pd.Timestamp(date) - pd.Timestamp(_entry_date)).days
        _trades.append({
            "Entry Date":   _entry_date,
            "Exit Date":    date,
            "Entry Price":  f"${_entry_price:.2f}",
            "Exit Price":   f"${price:.2f}",
            "P&L":          _pnl,
            "Days Held":    _days,
        })
        _entry_date, _entry_price = None, None

if _trades:
    trade_df = pd.DataFrame(_trades)
    wins = (trade_df["P&L"] > 0).sum()
    losses = (trade_df["P&L"] <= 0).sum()
    avg_pnl = trade_df["P&L"].mean()
    win_rate = wins / len(trade_df)
    avg_days = trade_df['Days Held'].mean()

    # Summary badge row (replaces the 4 st.metric cards)
    _wr_cls = "badge-green" if win_rate > 0.5 else "badge-red"
    _pl_cls = "badge-green" if avg_pnl > 0 else "badge-red"
    st.markdown(
        f'<div style="margin-bottom:12px;">'
        f'<span class="badge {_wr_cls}">🏆 Win Rate: {win_rate:.0%}</span>'
        f'<span class="badge {_pl_cls}">📊 Avg P&L: {avg_pnl:+.2%}</span>'
        f'<span class="badge badge-blue">📋 Total Trades: {len(trade_df)}</span>'
        f'<span class="badge badge-amber">⏱ Avg Days: {avg_days:.0f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Colour P&L column — green for positive, red for negative
    pnl_display = trade_df.copy()
    pnl_display["P&L"] = trade_df["P&L"].map("{:+.2%}".format)

    def _pnl_style(val):
        if isinstance(val, str) and val.startswith('+'):
            return f"color: {C['green']}; font-weight: 600;"
        if isinstance(val, str) and val.startswith('-'):
            return f"color: {C['red']}; font-weight: 600;"
        return ""

    styled_tdf = (
        pnl_display.set_index("Entry Date")
        .style.map(_pnl_style, subset=["P&L"])
    )
    st.dataframe(styled_tdf, width="stretch")

    # Cumulative P&L per trade
    fig_tlog = go.Figure()
    cum_pnl = (1 + trade_df["P&L"]).cumprod() - 1
    bar_colors = [C["green"] if v >= 0 else C["red"] for v in trade_df["P&L"]]
    fig_tlog.add_trace(go.Bar(
        x=list(range(1, len(trade_df) + 1)),
        y=trade_df["P&L"] * 100,
        marker_color=bar_colors,
        name="Trade P&L (%)",
    ))
    fig_tlog.add_trace(go.Scatter(
        x=list(range(1, len(trade_df) + 1)),
        y=cum_pnl * 100,
        name="Cumulative P&L (%)",
        line=dict(color=C["blue"], width=2),
        yaxis="y2",
    ))
    fig_tlog.update_layout(**_dark(
        height=320,
        xaxis_title="Trade #",
        yaxis_title="Trade P&L (%)",
        yaxis2=dict(title="Cumulative P&L (%)", overlaying="y", side="right",
                    gridcolor=C["grid"], zeroline=False),
        bargap=0.3,
    ))
    _style_axes(fig_tlog)
    st.plotly_chart(fig_tlog, width="stretch")
else:
    st.info("No completed trades in the selected period.")

# ── SECTION 13: AI Market Analysis ───────────────────────────
_section_header(f"🤖 {selected} — AI Market Analysis", color=C["purple"])

if st.button("⚡ Generate AI Analysis", type="primary", use_container_width=True):
    with st.spinner("Analyzing market data with Groq LLaMA 3.3…"):
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

        ai_text = message.choices[0].message.content.replace("\n", "<br>")
        st.markdown(
            f'<div class="ai-card">'
            f'<div class="ai-card-header">🤖 AI Analysis — {selected}</div>'
            f'{ai_text}'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────
st.markdown(
    f'<div class="footer-bar">'
    f'Data: yfinance &nbsp;·&nbsp; Optimization: PyPortfolioOpt &nbsp;·&nbsp; '
    f'AI: Groq LLaMA 3.3 &nbsp;·&nbsp; © AlphaTrade 2026'
    f'</div>',
    unsafe_allow_html=True,
)
