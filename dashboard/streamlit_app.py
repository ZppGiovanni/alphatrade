import sys
sys.path.insert(0, ".")

import os
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from data.database import load_ohlcv, DB_PATH
from data.fetcher import fetch_all, fetch_historical
from data.normalizer import add_indicators
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.macd_crossover import MACDCrossoverStrategy
from strategies.bollinger_bands import BollingerBandsStrategy
from strategies.ml_model import MLStrategy
from portfolio.optimizer import compute_weights
from portfolio.risk import backtest

load_dotenv()

if not DB_PATH.exists():
    with st.spinner("First run: downloading market data (this takes ~30 seconds)…"):
        fetch_all()

ASSETS     = ["QQQ", "XLE", "GLD", "XLV", "ARKK"]
STRATEGIES = ["Momentum", "Mean Reversion", "MACD Crossover", "Bollinger Bands", "ML Model"]
COLORS     = ["#4FC3F7", "#26a69a", "#FFB74D", "#CE93D8", "#ef5350"]

C = dict(
    bg="#0e1117", surface="#161b27", grid="#1f2937", border="#0f3460",
    green="#26a69a", red="#ef5350", blue="#4FC3F7", amber="#FFB74D",
    purple="#CE93D8", orange="#FF8C00", grey="#9e9e9e",
)

st.set_page_config(page_title="AlphaTrade", layout="wide",
                   initial_sidebar_state="expanded")

# Force sidebar open regardless of browser localStorage state
import streamlit.components.v1 as _components
_components.html("""
<script>
(function() {
    function forceOpen() {
        var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
            var btn = window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"] button');
            if (btn) btn.click();
        }
    }
    setTimeout(forceOpen, 300);
    setTimeout(forceOpen, 800);
})();
</script>
""", height=0)

# ── Global CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], [data-testid], .stMarkdown, .stText,
.stButton, .stSelectbox, .stDataFrame, .stCaption, p, h1, h2, h3, h4, h5 {
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header {visibility: hidden;}

@keyframes gradient-shift {
    0%   { background-position: 0% 50%;   }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%;   }
}
.animated-title {
    background: linear-gradient(270deg, #4FC3F7, #1E88E5, #1565C0, #29B6F6, #4FC3F7);
    background-size: 400% 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradient-shift 8s ease infinite;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    line-height: 1.2;
}

[data-testid="stSidebar"] {
    background-color: #161b27 !important;
    border-right: 1px solid #0f3460;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #161b27;
    border-bottom: 1px solid #0f3460;
    padding: 0 4px;
    gap: 2px;
    border-radius: 8px 8px 0 0;
}
.stTabs [data-baseweb="tab"] {
    color: #9e9e9e !important;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 10px 16px;
    border-radius: 6px 6px 0 0;
    background: transparent !important;
    transition: all 0.15s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #e0e0e0 !important;
    background: rgba(79,195,247,0.08) !important;
}
.stTabs [aria-selected="true"] {
    color: #4FC3F7 !important;
    background: rgba(79,195,247,0.12) !important;
    border-bottom: 2px solid #4FC3F7 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.2rem;
    background: transparent;
}

[data-testid="stPlotlyChart"] {
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 2px;
}
[data-testid="stDataFrame"] {
    border: 1px solid #0f3460;
    border-radius: 8px;
    overflow: hidden;
}

.stButton > button {
    background: linear-gradient(135deg, #0f3460, #1a5a8a);
    color: #4FC3F7;
    border: 1px solid #0f3460;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0.5rem 1.6rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1a5a8a, #4FC3F7);
    color: #0e1117;
    border-color: #4FC3F7;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(79,195,247,0.25);
}
.stSpinner > div { border-top-color: #4FC3F7 !important; }
hr { border-color: #0f3460 !important; }

/* Keep sidebar always expanded */
[data-testid="stSidebarCollapseButton"] { visibility: hidden !important; }

/* Fixed sidebar toggle button */
#sb-toggle-wrap {
    position: fixed;
    top: 50vh;
    transform: translateY(-50%);
    z-index: 9999;
    transition: left 0.3s ease;
}
#sb-toggle-wrap button {
    background: #161b27 !important;
    border: 1px solid #0f3460 !important;
    border-left: none !important;
    border-radius: 0 8px 8px 0 !important;
    color: #4FC3F7 !important;
    width: 20px !important;
    min-width: 20px !important;
    height: 52px !important;
    padding: 0 !important;
    font-size: 0.75rem !important;
    cursor: pointer;
    box-shadow: 3px 0 8px rgba(0,0,0,0.5) !important;
    transition: background 0.2s !important;
}
#sb-toggle-wrap button:hover {
    background: #0f3460 !important;
    color: #e0e0e0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────
def _dark(**kw) -> dict:
    base = dict(
        template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font_size=11),
        font=dict(color="#e0e0e0", family="Inter, sans-serif"),
    )
    base.update(kw)
    return base


def _axes(fig):
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False)
    fig.update_xaxes(gridcolor=C["grid"], showgrid=False)


def _chart(fig):
    st.plotly_chart(fig, width="stretch",
                    config={"displayModeBar": False, "displaylogo": False})


def _kpi(label, value, delta="", delta_color=None, border_color=None, icon=""):
    bc = border_color or C["border"]
    dc = delta_color or C["grey"]
    icon_html  = (f'<span style="font-size:1.2rem;margin-bottom:6px;display:block">'
                  f'{icon}</span>') if icon else ""
    delta_html = (f'<p style="color:{dc};font-size:0.8rem;margin:4px 0 0">{delta}</p>'
                  ) if delta else ""
    return f"""
    <div style="background:linear-gradient(135deg,{C['surface']},{C['bg']});
                border:1px solid {C['border']};border-left:3px solid {bc};
                border-radius:10px;padding:1.1rem 1.3rem 1rem;">
        {icon_html}
        <p style="color:{C['grey']};font-size:0.72rem;text-transform:uppercase;
                  letter-spacing:0.08em;margin:0 0 5px">{label}</p>
        <p style="color:#e0e0e0;font-size:1.5rem;font-weight:700;
                  margin:0;line-height:1.2">{value}</p>
        {delta_html}
    </div>"""


def _section(title):
    st.html(
        f'<div style="color:#e0e0e0;font-size:1.0rem;font-weight:600;'
        f'border-bottom:1px solid {C["border"]};padding-bottom:8px;'
        f'margin-bottom:1rem">{title}</div>'
    )


def _consensus_banner(label, color):
    st.html(f"""
    <div style="background:{color}22;border-left:4px solid {color};border-radius:8px;
                padding:14px 22px;margin:0.5rem 0 1rem">
        <span style="font-size:1.3rem;font-weight:700;color:{color}">
            Current Signal: {label}
        </span>
    </div>""")


def _get_strategy(name):
    if name == "Mean Reversion":
        return MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5})
    if name == "MACD Crossover":
        return MACDCrossoverStrategy(params={})
    if name == "Bollinger Bands":
        return BollingerBandsStrategy(params={"window": 20, "num_std": 2.0})
    if name == "ML Model":
        return MLStrategy(params={"n_estimators": 100, "threshold": 0.6})
    return MomentumStrategy(params={"short_window": 20, "long_window": 50})


# ── Cached heavy computations ─────────────────────────────────────
@st.cache_data(ttl=3600)
def _load(ticker):
    from data.database import save_ohlcv, _load_ohlcv_cached
    fresh = fetch_historical(ticker)
    save_ohlcv(fresh)
    _load_ohlcv_cached.cache_clear()
    df = load_ohlcv(ticker)
    return add_indicators(df)


@st.cache_data(ttl=3600)
def _comparison_data(ticker, n_bars):
    df = _load(ticker)
    rows, returns, bh_ref = [], [], None
    for name, strat in [
        ("Momentum",       MomentumStrategy(params={"short_window": 20, "long_window": 50})),
        ("Mean Reversion", MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5})),
        ("MACD Crossover", MACDCrossoverStrategy(params={})),
        ("Bollinger Bands", BollingerBandsStrategy(params={"window": 20, "num_std": 2.0})),
    ]:
        sig = strat.generate_signals(df).iloc[-n_bars:]
        res = backtest(df["close"].iloc[-n_bars:], sig)
        if bh_ref is None:
            bh_ref = res["buy_and_hold"]
        rows.append({
            "Strategy": name,
            "Return":   f"{res['total_return']:+.1%}",
            "B&H":      f"{res['buy_and_hold']:+.1%}",
            "Sharpe":   f"{res['sharpe_ratio']:.2f}",
            "Max DD":   f"{res['max_drawdown']:.1%}",
            "Trades":   int(res["n_trades"]),
        })
        returns.append(res["total_return"])
    return rows, returns, bh_ref


@st.cache_data(ttl=3600)
def _portfolio_weights(n_bars):
    prices = pd.DataFrame({t: _load(t)["close"] for t in ASSETS})
    return compute_weights(prices.iloc[-n_bars:])


# ── Sidebar: controls ─────────────────────────────────────────────
PERIODS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260}
with st.sidebar:
    st.html(f"""
    <div style="padding:0.5rem 0 1rem;font-family:'Inter',sans-serif">
        <h2 class="animated-title" style="margin:0;font-size:1.4rem;font-weight:800">
            AlphaTrade
        </h2>
        <p style="color:{C['grey']};font-size:0.78rem;margin:4px 0 0">
            Algorithmic Trading System
        </p>
    </div>""")
    st.divider()
    st.html(f'<p style="color:{C["grey"]};font-size:0.8rem;text-transform:uppercase;'
            f'letter-spacing:0.05em;margin-bottom:4px;font-family:Inter,sans-serif">Asset</p>')
    selected = st.selectbox("Asset", ASSETS, label_visibility="collapsed")
    st.html(f'<p style="color:{C["grey"]};font-size:0.8rem;text-transform:uppercase;'
            f'letter-spacing:0.05em;margin-top:0.8rem;margin-bottom:4px;font-family:Inter,sans-serif">Strategy</p>')
    strategy_name = st.selectbox("Strategy", STRATEGIES, label_visibility="collapsed")
    st.html(f'<p style="color:{C["grey"]};font-size:0.8rem;text-transform:uppercase;'
            f'letter-spacing:0.05em;margin-top:0.8rem;margin-bottom:4px;font-family:Inter,sans-serif">Period</p>')
    period_label = st.selectbox("Period", list(PERIODS.keys()),
                                index=3, label_visibility="collapsed")
    st.html("<div style='margin-top:1rem'></div>")
    if st.button("↻  Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()


# ── Load data once ────────────────────────────────────────────────
df_full  = _load(selected)
n_bars   = PERIODS[period_label]
df       = df_full.iloc[-n_bars:].copy()
# Compute signals on full history, then slice — avoids warm-up NaNs in
# rolling windows and spurious crossover signals at the period boundary
signals  = _get_strategy(strategy_name).generate_signals(df_full).iloc[-n_bars:]

# Pre-compute all metrics used across tabs and sidebar
close   = df["close"].iloc[-1]
ret_1d  = df["return_1d"].iloc[-1]
ret_5d  = df["return_5d"].iloc[-1]
rsi     = df["rsi"].iloc[-1]
rsi_lbl = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
rsi_col = C["red"] if rsi > 70 else (C["green"] if rsi < 30 else C["grey"])

sig_mom  = MomentumStrategy(params={"short_window": 20, "long_window": 50}).generate_signals(df_full).iloc[-n_bars:]
sig_mr   = MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5}).generate_signals(df_full).iloc[-n_bars:]
sig_macd = MACDCrossoverStrategy(params={}).generate_signals(df_full).iloc[-n_bars:]
sig_bb   = BollingerBandsStrategy(params={"window": 20, "num_std": 2.0}).generate_signals(df_full).iloc[-n_bars:]
consensus = sig_mom + sig_mr + sig_macd + sig_bb
latest    = int(consensus.iloc[-1])

if   latest >= 3:  label, con_color = "🟢 STRONG BUY",  C["green"]
elif latest >= 1:  label, con_color = "🟡 WEAK BUY",    C["amber"]
elif latest == 0:  label, con_color = "⚪ HOLD",         C["grey"]
elif latest >= -2: label, con_color = "🟠 WEAK SELL",   C["orange"]
else:              label, con_color = "🔴 STRONG SELL", C["red"]

# Sidebar: quick stats (added after data loads)
with st.sidebar:
    st.divider()
    st.html(f"""
    <div style="background:{C['surface']};border:1px solid {C['border']};
                border-radius:10px;padding:1rem 1.1rem;font-family:'Inter',sans-serif">
        <p style="color:{C['grey']};font-size:0.68rem;text-transform:uppercase;
                  letter-spacing:0.08em;margin:0 0 8px">{selected} — Quick Stats</p>
        <div style="display:flex;justify-content:space-between;margin-bottom:5px">
            <span style="color:{C['grey']};font-size:0.72rem">Last Close</span>
            <span style="color:#e0e0e0;font-weight:600;font-size:0.72rem">${close:.2f}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:5px">
            <span style="color:{C['grey']};font-size:0.72rem">1D Return</span>
            <span style="color:{'#26a69a' if ret_1d >= 0 else '#ef5350'};font-weight:600;font-size:0.72rem">
                {ret_1d:+.2%}
            </span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:5px">
            <span style="color:{C['grey']};font-size:0.72rem">RSI (14)</span>
            <span style="color:{rsi_col};font-weight:600;font-size:0.72rem">{rsi:.1f} · {rsi_lbl}</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="color:{C['grey']};font-size:0.72rem">Consensus</span>
            <span style="color:{con_color};font-weight:600;font-size:0.72rem">{label}</span>
        </div>
    </div>
    <br>
    <p style="color:{C['grey']};font-size:0.76rem;line-height:1.9;margin:0">
        Data: yfinance<br>
        Optimization: PyPortfolioOpt<br>
        AI: Groq LLaMA 3.3 70B<br>
        © AlphaTrade 2026
    </p>""")


# ── Header ────────────────────────────────────────────────────────
st.html(f"""
<div style="padding-bottom:0.8rem">
    <h1 class="animated-title">AlphaTrade</h1>
    <p style="color:{C['grey']};font-size:0.9rem;margin:6px 0 0">
        Algorithmic ETF Trading System &nbsp;·&nbsp;
        USI Programming in Finance II, 2026
    </p>
</div>""")
st.divider()


# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Price & Signals",
    "MACD",
    "Backtest",
    "Comparison",
    "Portfolio",
    "Consensus",
    "AI Analysis",
])


# ── Tab 1: Price & Signals ────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.html(_kpi("Last Close", f"${close:.2f}",
                     border_color=C["blue"]))
    c2.html(_kpi("RSI (14)", f"{rsi:.1f}", rsi_lbl, rsi_col, rsi_col))
    c3.html(_kpi("1D Return", f"{ret_1d:.2%}", f"{ret_1d:.2%}",
        C["green"] if ret_1d >= 0 else C["red"],
        C["green"] if ret_1d >= 0 else C["red"]))
    c4.html(_kpi("5D Return", f"{ret_5d:.2%}", f"{ret_5d:.2%}",
        C["green"] if ret_5d >= 0 else C["red"],
        C["green"] if ret_5d >= 0 else C["red"]))

    st.html("<div style='margin-top:2rem'></div>")
    _section(f"{selected} — Candlestick + SMA")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.58, 0.18, 0.24], vertical_spacing=0.06,
        subplot_titles=("", "Volume", "RSI (14)"))
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price",
        increasing_line_color=C["green"], decreasing_line_color=C["red"],
        increasing_fillcolor=C["green"], decreasing_fillcolor=C["red"],
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"],
        line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"],
        line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(255,255,255,0.03)", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], name="SMA 20",
        line=dict(color=C["amber"], width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50",
        line=dict(color=C["purple"], width=1.4)), row=1, col=1)
    buys  = df[signals == 1]
    sells = df[signals == -1]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["low"] * 0.985,
        mode="markers", name="Buy",
        marker=dict(color=C["green"], size=9, symbol="triangle-up")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells["high"] * 1.015,
        mode="markers", name="Sell",
        marker=dict(color=C["red"], size=9, symbol="triangle-down")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["volume"],
        line=dict(color=C["blue"], width=1),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.15)",
        showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=list(df.index) + list(df.index[::-1]),
        y=[70] * len(df) + [30] * len(df),
        fill="toself", fillcolor="rgba(255,255,255,0.04)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
        line=dict(color=C["purple"], width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.5)",    row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.5)",   row=3, col=1)
    fig.add_hline(y=50, line_dash="dot",  line_color="rgba(255,255,255,0.12)", row=3, col=1)
    fig.update_layout(**_dark(height=680, xaxis_rangeslider_visible=False,
                              margin=dict(l=0, r=0, t=30, b=0)))
    _axes(fig)
    _chart(fig)


# ── Tab 2: MACD ───────────────────────────────────────────────────
with tab2:
    hist = df["macd"] - df["macd_signal"]
    hcol = [C["green"] if v >= 0 else C["red"] for v in hist]

    fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.45], vertical_spacing=0.05,
        subplot_titles=("MACD Line & Signal", "Histogram"))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
        line=dict(color=C["blue"], width=1.8)), row=1, col=1)
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
        line=dict(color=C["orange"], width=1.4, dash="dash")), row=1, col=1)
    fig_macd.add_hline(y=0, line_dash="dot",
                       line_color="rgba(255,255,255,0.2)", row=1, col=1)
    fig_macd.add_trace(go.Scatter(x=df.index, y=hist,
        line=dict(color=C["blue"], width=1),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.15)",
        showlegend=False), row=2, col=1)
    fig_macd.add_hline(y=0, line_dash="dot",
                       line_color="rgba(255,255,255,0.2)", row=2, col=1)
    fig_macd.update_layout(**_dark(height=420))
    _axes(fig_macd)
    _chart(fig_macd)


# ── Tab 3: Backtest ───────────────────────────────────────────────
with tab3:
    res = backtest(df["close"], signals)
    eq  = res["equity_curve"]
    bh  = 10000 * (df["close"] / df["close"].iloc[0])
    dd  = (eq - eq.cummax()) / eq.cummax() * 100
    tr  = res["total_return"]
    bhr = res["buy_and_hold"]
    sr  = res["sharpe_ratio"]
    mdd = res["max_drawdown"]

    c1, c2, c3, c4 = st.columns(4)
    c1.html(_kpi("Strategy Return", f"{tr:+.1%}", f"{tr - bhr:+.1%} vs B&H",
        C["green"] if tr >= bhr else C["red"],
        C["green"] if tr >= bhr else C["red"]))
    c2.html(_kpi("Buy & Hold", f"{bhr:+.1%}",
                     border_color=C["amber"]))
    c3.html(_kpi("Sharpe Ratio", f"{sr:.2f}", "Good" if sr > 1 else "Low",
        C["green"] if sr > 1 else C["red"],
        C["green"] if sr > 1 else C["red"]))
    c4.html(_kpi("Max Drawdown", f"{mdd:.1%}",
                     border_color=C["red"]))

    st.html("<div style='margin-top:1rem'></div>")

    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.04,
        subplot_titles=("Portfolio Value ($)", "Drawdown (%)"))
    fig_bt.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy",
        line=dict(color=C["blue"], width=2),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.07)"), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=bh.index, y=bh.values, name="Buy & Hold",
        line=dict(color=C["amber"], width=1.5, dash="dash")), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
        line=dict(color=C["red"], width=1.5),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.15)",
        showlegend=False), row=2, col=1)
    fig_bt.update_layout(**_dark(height=500))
    _axes(fig_bt)
    _chart(fig_bt)


# ── Tab 4: Strategy Comparison ────────────────────────────────────
with tab4:
    rows_cmp, returns_cmp, bh_ref = _comparison_data(selected, n_bars)
    col_l, col_r = st.columns(2)
    with col_l:
        _section("Performance Table")
        def _perf_table(rows):
            cols = list(rows[0].keys())
            header = "".join(
                f'<th style="color:{C["grey"]};font-size:0.72rem;text-transform:uppercase;'
                f'letter-spacing:0.07em;padding:8px 12px;text-align:{"left" if i==0 else "right"};'
                f'border-bottom:1px solid {C["border"]};font-weight:500">{c}</th>'
                for i, c in enumerate(cols))
            rows_html = ""
            for r in rows:
                cells = ""
                for i, (k, v) in enumerate(r.items()):
                    color = "#e0e0e0"
                    if k in ("Return", "B&H") and isinstance(v, str):
                        color = C["green"] if v.startswith("+") else C["red"]
                    align = "left" if i == 0 else "right"
                    cells += (f'<td style="color:{color};font-size:0.86rem;'
                              f'padding:9px 12px;text-align:{align};'
                              f'border-bottom:1px solid {C["grid"]}">{v}</td>')
                rows_html += f"<tr>{cells}</tr>"
            return (f'<table style="width:100%;border-collapse:collapse;'
                    f'font-family:Inter,sans-serif;background:{C["surface"]};'
                    f'border:1px solid {C["border"]};border-radius:8px;overflow:hidden">'
                    f"<thead><tr>{header}</tr></thead>"
                    f"<tbody>{rows_html}</tbody></table>")
        col_l.html(_perf_table(rows_cmp))
    with col_r:
        _section("Return vs B&H")
        bar_colors_cmp = [C["green"] if v >= 0 else C["red"] for v in returns_cmp]
        fig_cmp = go.Figure(go.Bar(
            x=[r["Strategy"] for r in rows_cmp],
            y=[v * 100 for v in returns_cmp],
            marker_color=bar_colors_cmp,
            text=[f"{v:+.1%}" for v in returns_cmp], textposition="outside",
        ))
        fig_cmp.add_hline(y=bh_ref * 100, line_dash="dash", line_color=C["amber"],
                          annotation_text="B&H", annotation_position="top right")
        fig_cmp.update_layout(**_dark(height=300, showlegend=False,
            yaxis_title="Return (%)", margin=dict(l=0, r=0, t=30, b=40)))
        _axes(fig_cmp)
        _chart(fig_cmp)


# ── Tab 5: Portfolio Weights ──────────────────────────────────────
with tab5:
    try:
        weights = _portfolio_weights(n_bars)
        names_w = list(weights.keys())
        vals_w  = list(weights.values())
        col_l, col_r = st.columns(2)
        with col_l:
            _section("Allocation (Donut)")
            fig_donut = go.Figure(go.Pie(
                values=vals_w, labels=names_w, hole=0.45,
                marker=dict(colors=COLORS, line=dict(color=C["bg"], width=2)),
                textinfo="label+percent", insidetextorientation="radial",
            ))
            fig_donut.update_layout(
                **_dark(height=350, margin=dict(l=0, r=0, t=10, b=0),
                annotations=[dict(text="Weights", x=0.5, y=0.5,
                                  font_size=13, showarrow=False,
                                  font_color="#e0e0e0")]))
            _chart(fig_donut)
        with col_r:
            _section("Allocation (Bar)")
            fig_wbar = go.Figure(go.Bar(
                x=vals_w, y=names_w, orientation="h",
                marker_color=COLORS,
                text=[f"{v:.1%}" for v in vals_w], textposition="outside",
            ))
            fig_wbar.update_layout(**_dark(height=350, showlegend=False,
                xaxis_tickformat=".0%", margin=dict(l=0, r=60, t=10, b=0)))
            _axes(fig_wbar)
            _chart(fig_wbar)
        st.caption(f"Markowitz Mean-Variance Optimization — Max Sharpe Ratio · Period: {period_label}")
    except Exception as e:
        st.error(f"Portfolio optimization failed: {e}")


# ── Tab 6: Consensus ──────────────────────────────────────────────
with tab6:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.html(_kpi("Momentum",        str(int(sig_mom.iloc[-1]))))
    c2.html(_kpi("Mean Reversion",  str(int(sig_mr.iloc[-1]))))
    c3.html(_kpi("MACD Crossover",  str(int(sig_macd.iloc[-1]))))
    c4.html(_kpi("Bollinger Bands", str(int(sig_bb.iloc[-1]))))
    c5.html(_kpi("Consensus Score", str(latest),
                     border_color=con_color))

    st.html("<div style='margin-top:1rem'></div>")
    _consensus_banner(label, con_color)

    col_l, col_r = st.columns([1, 2])
    with col_l:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest,
            number={"font": {"color": con_color, "size": 44}},
            gauge={
                "axis": {
                    "range": [-4, 4],
                    "tickvals": [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                    "tickfont": {"color": C["grey"], "size": 10},
                    "tickcolor": C["border"],
                },
                "bar": {"color": con_color, "thickness": 0.25},
                "bgcolor": C["surface"],
                "borderwidth": 1,
                "bordercolor": C["border"],
                "steps": [
                    {"range": [-4, -2], "color": "rgba(239,83,80,0.15)"},
                    {"range": [-2, 2],  "color": "rgba(158,158,158,0.06)"},
                    {"range": [2, 4],   "color": "rgba(38,166,154,0.15)"},
                ],
                "threshold": {
                    "line": {"color": con_color, "width": 2},
                    "thickness": 0.8,
                    "value": latest,
                },
            },
            title={"text": "Consensus Score",
                   "font": {"color": C["grey"], "size": 13}},
        ))
        fig_gauge.update_layout(
            **_dark(height=280, margin=dict(l=20, r=20, t=40, b=20)))
        _chart(fig_gauge)
    with col_r:
        bar_col_con = [C["green"] if v > 0 else (C["red"] if v < 0 else C["grey"])
                       for v in consensus.values]
        fig_con = go.Figure(go.Bar(x=consensus.index, y=consensus.values,
            marker_color=bar_col_con, opacity=0.85, showlegend=False))
        for lvl, col, lbl in [(3, C["green"], "STRONG BUY"),
                              (-3, C["red"],  "STRONG SELL")]:
            fig_con.add_hline(y=lvl, line_dash="dash", line_color=col,
                annotation_text=lbl, annotation_position="top right",
                annotation_font_color=col)
        fig_con.add_hline(y=0, line_dash="dot",
                          line_color="rgba(255,255,255,0.2)")
        fig_con.update_layout(**_dark(height=280,
            yaxis=dict(tickvals=[-4, -3, -2, -1, 0, 1, 2, 3, 4], title="Score")))
        _axes(fig_con)
        _chart(fig_con)


# ── Tab 7: AI Analysis ────────────────────────────────────────────
with tab7:
    st.html(f"""
    <p style="color:{C['grey']};font-size:0.9rem;margin-bottom:1.2rem">
        Generate a qualitative market analysis for
        <strong style="color:#e0e0e0">{selected}</strong> using
        Groq LLaMA 3.3 70B based on current indicators and strategy signals.
    </p>""")

    if st.button("Generate AI Analysis"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY not set in environment.")
        else:
            recent = df.tail(10)[["close", "rsi", "macd", "macd_signal",
                                   "bb_upper", "bb_lower"]].round(2)
            prompt = f"""You are a quantitative financial analyst. Analyze {selected} ETF.

Market data (last 10 days):
{recent.to_string()}

Metrics: Close=${df['close'].iloc[-1]:.2f}, RSI={df['rsi'].iloc[-1]:.1f}, 1D={df['return_1d'].iloc[-1]:.2%}, 5D={df['return_5d'].iloc[-1]:.2%}
Consensus: {latest}/4, Signal: {label}
Momentum return: {backtest(df['close'], sig_mom)['total_return']:+.1%}
Mean Reversion return: {backtest(df['close'], sig_mr)['total_return']:+.1%}
MACD return: {backtest(df['close'], sig_macd)['total_return']:+.1%}
Bollinger Bands return: {backtest(df['close'], sig_bb)['total_return']:+.1%}

Provide: 1) Market conditions, 2) Signal interpretation, 3) Risk considerations, 4) Outlook. Max 200 words."""

            try:
                with st.spinner("Generating analysis..."):
                    client = Groq(api_key=api_key)
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile", max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}])
                content = resp.choices[0].message.content
                html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
                html_content = re.sub(r'^#{1,3} (.+)$', r'<strong style="font-size:1rem">\1</strong>', html_content, flags=re.MULTILINE)
                html_content = html_content.replace("\n", "<br>")
                st.html(f"""
                <div style="background:{C['surface']};border:1px solid {C['border']};
                            border-left:4px solid {C['blue']};border-radius:10px;
                            padding:1.4rem 1.8rem;margin-top:0.5rem;line-height:1.75;
                            color:#e0e0e0;font-size:0.95rem">
                    {html_content}
                </div>""")
            except Exception as e:
                st.error(f"AI request failed: {e}")


# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.html(f"""
<p style="color:{C['grey']};font-size:0.78rem;text-align:center;margin:0">
    Data: yfinance &nbsp;·&nbsp; Optimization: PyPortfolioOpt &nbsp;·&nbsp;
    AI: Groq LLaMA 3.3 70B &nbsp;·&nbsp; © AlphaTrade 2026
</p>""")
