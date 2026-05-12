import sys
sys.path.insert(0, ".")

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dotenv import load_dotenv
from groq import Groq

from data.database import load_ohlcv
from data.normalizer import add_indicators
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.macd_crossover import MACDCrossoverStrategy
from strategies.ml_model import MLStrategy
from portfolio.optimizer import compute_weights
from portfolio.risk import backtest

load_dotenv()

ASSETS     = ["QQQ", "XLE", "GLD", "XLV", "ARKK"]
STRATEGIES = ["Momentum", "Mean Reversion", "MACD Crossover", "ML Model"]
COLORS     = ["#4FC3F7", "#26a69a", "#FFB74D", "#CE93D8", "#ef5350"]

C = dict(
    bg="#0e1117", surface="#161b27", grid="#1f2937", border="#0f3460",
    green="#26a69a", red="#ef5350", blue="#4FC3F7", amber="#FFB74D",
    purple="#CE93D8", orange="#FF8C00", grey="#9e9e9e",
)


# ── Helpers ───────────────────────────────────────────────────
def _dark(**kw) -> dict:
    base = dict(
        template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font_size=11),
        font=dict(color="#e0e0e0"),
    )
    base.update(kw)
    return base


def _axes(fig):
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False)
    fig.update_xaxes(gridcolor=C["grid"], showgrid=False)


def _get_strategy(name):
    if name == "Mean Reversion":
        return MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5})
    if name == "MACD Crossover":
        return MACDCrossoverStrategy(params={})
    if name == "ML Model":
        return MLStrategy(params={"n_estimators": 100, "threshold": 0.6})
    return MomentumStrategy(params={"short_window": 20, "long_window": 50})


def _card(label, value, delta="", delta_color=None):
    delta_color = delta_color or C["grey"]
    return dbc.Card(dbc.CardBody([
        html.P(label, className="mb-1",
               style={"color": C["grey"], "fontSize": "0.78rem", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
        html.H5(value, className="mb-0 fw-bold"),
        html.Small(delta, style={"color": delta_color}) if delta else html.Span(),
    ]), style={"background": C["surface"], "border": f"1px solid {C['border']}", "borderRadius": 8})


def _section(title):
    return html.H5(title, style={
        "color": "#e0e0e0", "borderBottom": f"1px solid {C['border']}",
        "paddingBottom": 6, "marginTop": "1.8rem", "marginBottom": "1rem",
    })


# ── App ───────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], title="AlphaTrade")
app.layout = dbc.Container(fluid=True, style={"backgroundColor": C["bg"], "minHeight": "100vh", "padding": "1.5rem 2rem"}, children=[

    # Header
    html.H2("📈 AlphaTrade — Algorithmic ETF Trading System",
            style={"color": C["blue"], "marginBottom": 2}),
    html.P("USI Programming in Finance II, 2026",
           style={"color": C["grey"], "marginBottom": "1.5rem", "fontSize": "0.9rem"}),

    # Controls
    dbc.Row(className="mb-2 g-3", children=[
        dbc.Col(width=2, children=[
            html.Label("Asset", style={"color": C["grey"], "fontSize": "0.85rem"}),
            dcc.Dropdown(id="dd-asset", options=[{"label": a, "value": a} for a in ASSETS],
                         value="QQQ", clearable=False),
        ]),
        dbc.Col(width=3, children=[
            html.Label("Strategy", style={"color": C["grey"], "fontSize": "0.85rem"}),
            dcc.Dropdown(id="dd-strategy", options=[{"label": s, "value": s} for s in STRATEGIES],
                         value="Momentum", clearable=False),
        ]),
    ]),

    # ── Section 1: Price metrics + candlestick ────────────────
    _section("📊 Price & Signals"),
    dbc.Row(id="price-metrics", className="g-3 mb-3"),
    dcc.Graph(id="chart-candle", config={"displayModeBar": False}),

    # ── Section 2: MACD ───────────────────────────────────────
    _section("📉 MACD"),
    dcc.Graph(id="chart-macd", config={"displayModeBar": False}),

    # ── Section 3: Backtest ───────────────────────────────────
    html.H5(id="bt-title", style={"color": "#e0e0e0", "borderBottom": f"1px solid {C['border']}", "paddingBottom": 6, "marginTop": "1.8rem", "marginBottom": "1rem"}),
    dbc.Row(id="bt-metrics", className="g-3 mb-3"),
    dcc.Graph(id="chart-backtest", config={"displayModeBar": False}),

    # ── Section 4: Strategy comparison ───────────────────────
    html.H5(id="cmp-title", style={"color": "#e0e0e0", "borderBottom": f"1px solid {C['border']}", "paddingBottom": 6, "marginTop": "1.8rem", "marginBottom": "1rem"}),
    dbc.Row(className="g-3", children=[
        dbc.Col(width=6, children=[html.Div(id="cmp-table")]),
        dbc.Col(width=6, children=[dcc.Graph(id="chart-comparison", config={"displayModeBar": False})]),
    ]),

    # ── Section 5: Portfolio weights ──────────────────────────
    _section("🥧 Optimal Portfolio Weights (Max Sharpe)"),
    dbc.Row(className="g-3", children=[
        dbc.Col(width=6, children=[dcc.Graph(id="chart-donut", config={"displayModeBar": False})]),
        dbc.Col(width=6, children=[dcc.Graph(id="chart-wbar",  config={"displayModeBar": False})]),
    ]),
    html.P("Markowitz Mean-Variance Optimization — Max Sharpe Ratio",
           style={"color": C["grey"], "fontSize": "0.82rem", "marginTop": 4}),

    # ── Section 6: Consensus ──────────────────────────────────
    html.H5(id="con-title", style={"color": "#e0e0e0", "borderBottom": f"1px solid {C['border']}", "paddingBottom": 6, "marginTop": "1.8rem", "marginBottom": "1rem"}),
    dbc.Row(id="con-metrics", className="g-3 mb-3"),
    html.Div(id="con-banner", className="mb-3"),
    dcc.Graph(id="chart-consensus", config={"displayModeBar": False}),

    # ── Section 7: AI Analysis ────────────────────────────────
    html.H5(id="ai-title", style={"color": "#e0e0e0", "borderBottom": f"1px solid {C['border']}", "paddingBottom": 6, "marginTop": "1.8rem", "marginBottom": "1rem"}),
    dbc.Button("⚡ Generate AI Analysis", id="ai-btn", color="primary", className="mb-3"),
    dbc.Spinner(html.Div(id="ai-out"), color="primary"),

    # Footer
    html.Hr(style={"borderColor": C["border"], "marginTop": "2rem"}),
    html.P("Data: yfinance · Optimization: PyPortfolioOpt · AI: Groq LLaMA 3.3 · © AlphaTrade 2026",
           style={"color": C["grey"], "fontSize": "0.78rem"}),
])


# ── Callback: Price section ───────────────────────────────────
@app.callback(
    Output("price-metrics",  "children"),
    Output("chart-candle",   "figure"),
    Input("dd-asset",        "value"),
    Input("dd-strategy",     "value"),
)
def cb_price(selected, strategy_name):
    df = load_ohlcv(selected)
    df = add_indicators(df)
    signals = _get_strategy(strategy_name).generate_signals(df)

    ret_1d = df["return_1d"].iloc[-1]
    ret_5d = df["return_5d"].iloc[-1]
    rsi    = df["rsi"].iloc[-1]
    close  = df["close"].iloc[-1]

    rsi_lbl   = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
    rsi_color = C["red"] if rsi > 70 else (C["green"] if rsi < 30 else C["grey"])

    metrics = dbc.Row([
        dbc.Col(_card("Last Close", f"${close:.2f}"), width=3),
        dbc.Col(_card("RSI", f"{rsi:.1f}", rsi_lbl, rsi_color), width=3),
        dbc.Col(_card("1D Return", f"{ret_1d:.2%}", f"{ret_1d:.2%}", C["green"] if ret_1d >= 0 else C["red"]), width=3),
        dbc.Col(_card("5D Return", f"{ret_5d:.2%}", f"{ret_5d:.2%}", C["green"] if ret_5d >= 0 else C["red"]), width=3),
    ], className="g-3")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.58, 0.18, 0.24], vertical_spacing=0.03,
        subplot_titles=(f"{selected} — Candlestick + SMA", "Volume", "RSI (14)"))

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price",
        increasing_line_color=C["green"], decreasing_line_color=C["red"],
        increasing_fillcolor=C["green"], decreasing_fillcolor=C["red"],
    ), row=1, col=1)
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"],
        line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"],
        line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(255,255,255,0.03)", showlegend=False), row=1, col=1)
    # SMA lines
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], name="SMA 20",
        line=dict(color=C["amber"], width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50",
        line=dict(color=C["purple"], width=1.4)), row=1, col=1)
    # Buy / Sell markers
    buys  = df[signals == 1]
    sells = df[signals == -1]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["low"] * 0.985, mode="markers", name="Buy",
        marker=dict(color=C["green"], size=9, symbol="triangle-up")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells["high"] * 1.015, mode="markers", name="Sell",
        marker=dict(color=C["red"], size=9, symbol="triangle-down")), row=1, col=1)
    # Volume
    vol_colors = [C["green"] if c >= o else C["red"] for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=vol_colors,
        opacity=0.7, showlegend=False), row=2, col=1)
    # RSI neutral zone + line
    fig.add_trace(go.Scatter(
        x=list(df.index) + list(df.index[::-1]),
        y=[70] * len(df) + [30] * len(df),
        fill="toself", fillcolor="rgba(255,255,255,0.04)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
        line=dict(color=C["purple"], width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.5)",  row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.5)", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot",  line_color="rgba(255,255,255,0.12)", row=3, col=1)

    fig.update_layout(**_dark(height=680, xaxis_rangeslider_visible=False))
    _axes(fig)
    return [metrics, fig]


# ── Callback: MACD ────────────────────────────────────────────
@app.callback(
    Output("chart-macd", "figure"),
    Input("dd-asset",    "value"),
)
def cb_macd(selected):
    df = load_ohlcv(selected)
    df = add_indicators(df)
    hist = df["macd"] - df["macd_signal"]
    hcol = [C["green"] if v >= 0 else C["red"] for v in hist]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.45], vertical_spacing=0.05,
        subplot_titles=("MACD Line & Signal", "Histogram"))
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
        line=dict(color=C["blue"], width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
        line=dict(color=C["orange"], width=1.4, dash="dash")), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=hist, marker_color=hcol,
        opacity=0.85, showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)
    fig.update_layout(**_dark(height=380))
    _axes(fig)
    return fig


# ── Callback: Backtest ────────────────────────────────────────
@app.callback(
    Output("bt-title",       "children"),
    Output("bt-metrics",     "children"),
    Output("chart-backtest", "figure"),
    Input("dd-asset",        "value"),
    Input("dd-strategy",     "value"),
)
def cb_backtest(selected, strategy_name):
    df = load_ohlcv(selected)
    df = add_indicators(df)
    signals = _get_strategy(strategy_name).generate_signals(df)
    res = backtest(df["close"], signals)

    eq  = res["equity_curve"]
    bh  = 10000 * (df["close"] / df["close"].iloc[0])
    dd  = (eq - eq.cummax()) / eq.cummax() * 100
    tr  = res["total_return"]
    bhr = res["buy_and_hold"]
    sr  = res["sharpe_ratio"]
    mdd = res["max_drawdown"]

    metrics = dbc.Row([
        dbc.Col(_card("Strategy Return", f"{tr:+.1%}", f"{tr - bhr:+.1%} vs B&H", C["green"] if tr >= bhr else C["red"]), width=3),
        dbc.Col(_card("Buy & Hold",      f"{bhr:+.1%}"), width=3),
        dbc.Col(_card("Sharpe Ratio",    f"{sr:.2f}", "Good" if sr > 1 else "Low", C["green"] if sr > 1 else C["red"]), width=3),
        dbc.Col(_card("Max Drawdown",    f"{mdd:.1%}"), width=3),
    ], className="g-3")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.04,
        subplot_titles=("Portfolio Value ($)", "Drawdown (%)"))
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy",
        line=dict(color=C["blue"], width=2),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.07)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bh.index, y=bh.values, name="Buy & Hold",
        line=dict(color=C["amber"], width=1.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
        line=dict(color=C["red"], width=1.5),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.15)", showlegend=False), row=2, col=1)
    fig.update_layout(**_dark(height=480))
    _axes(fig)
    return [f"📈 {selected} — Backtest ({strategy_name})", metrics, fig]


# ── Callback: Strategy comparison ────────────────────────────
@app.callback(
    Output("cmp-title",        "children"),
    Output("cmp-table",        "children"),
    Output("chart-comparison", "figure"),
    Input("dd-asset",          "value"),
)
def cb_comparison(selected):
    df = load_ohlcv(selected)
    df = add_indicators(df)
    rows, returns = [], []
    for name, strat in [
        ("Momentum",       MomentumStrategy(params={"short_window": 20, "long_window": 50})),
        ("Mean Reversion", MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5})),
        ("MACD Crossover", MACDCrossoverStrategy(params={})),
    ]:
        sig = strat.generate_signals(df)
        res = backtest(df["close"], sig)
        rows.append({
            "Strategy": name,
            "Return":   res["total_return"],
            "B&H":      res["buy_and_hold"],
            "Sharpe":   res["sharpe_ratio"],
            "Max DD":   res["max_drawdown"],
            "Trades":   int(res["n_trades"]),
        })
        returns.append(res["total_return"])

    thead = html.Thead(html.Tr([
        html.Th(c, style={"color": C["grey"], "fontSize": "0.82rem"})
        for c in ["Strategy", "Return", "B&H", "Sharpe", "Max DD", "Trades"]
    ]))
    tbody = html.Tbody([
        html.Tr([
            html.Td(r["Strategy"]),
            html.Td(f"{r['Return']:+.1%}", style={"color": C["green"] if r["Return"] >= 0 else C["red"]}),
            html.Td(f"{r['B&H']:+.1%}"),
            html.Td(f"{r['Sharpe']:.2f}"),
            html.Td(f"{r['Max DD']:.1%}", style={"color": C["red"]}),
            html.Td(str(r["Trades"])),
        ]) for r in rows
    ])
    table = dbc.Table([thead, tbody], bordered=False, hover=True,
                      style={"fontSize": "0.9rem", "color": "#e0e0e0"})

    names = [r["Strategy"] for r in rows]
    bar_colors = [C["green"] if v >= 0 else C["red"] for v in returns]
    fig = go.Figure(go.Bar(
        x=names, y=[v * 100 for v in returns],
        marker_color=bar_colors,
        text=[f"{v:+.1%}" for v in returns], textposition="outside",
    ))
    fig.add_hline(y=rows[0]["B&H"] * 100, line_dash="dash", line_color=C["amber"],
                  annotation_text="B&H", annotation_position="top right")
    fig.update_layout(**_dark(height=280, showlegend=False,
        yaxis_title="Return (%)", margin=dict(l=0, r=0, t=30, b=40)))
    _axes(fig)
    return [f"⚖️ {selected} — Strategy Comparison", table, fig]


# ── Callback: Portfolio weights ───────────────────────────────
@app.callback(
    Output("chart-donut", "figure"),
    Output("chart-wbar",  "figure"),
    Input("dd-asset",     "value"),   # just to trigger on load
)
def cb_portfolio(_):
    try:
        prices  = pd.DataFrame({t: load_ohlcv(t)["close"] for t in ASSETS})
        weights = compute_weights(prices)
    except Exception as e:
        err = dbc.Alert(f"Portfolio optimization failed: {e}", color="danger")
        return err, err
    names   = list(weights.keys())
    vals    = list(weights.values())

    fig_donut = go.Figure(go.Pie(
        values=vals, labels=names, hole=0.45,
        marker=dict(colors=COLORS, line=dict(color=C["bg"], width=2)),
        textinfo="label+percent", insidetextorientation="radial",
    ))
    fig_donut.update_layout(**_dark(height=320, margin=dict(l=0, r=0, t=10, b=0),
        annotations=[dict(text="Weights", x=0.5, y=0.5,
                          font_size=13, showarrow=False, font_color="#e0e0e0")]))

    fig_bar = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=COLORS,
        text=[f"{v:.1%}" for v in vals], textposition="outside",
    ))
    fig_bar.update_layout(**_dark(height=320, showlegend=False,
        xaxis_tickformat=".0%", margin=dict(l=0, r=60, t=10, b=0)))
    _axes(fig_bar)
    return [fig_donut, fig_bar]


# ── Callback: Consensus ───────────────────────────────────────
@app.callback(
    Output("con-title",       "children"),
    Output("con-metrics",     "children"),
    Output("con-banner",      "children"),
    Output("chart-consensus", "figure"),
    Input("dd-asset",         "value"),
)
def cb_consensus(selected):
    df = load_ohlcv(selected)
    df = add_indicators(df)
    sig_mom  = MomentumStrategy(params={"short_window": 20, "long_window": 50}).generate_signals(df)
    sig_mr   = MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5}).generate_signals(df)
    sig_macd = MACDCrossoverStrategy(params={}).generate_signals(df)
    consensus = sig_mom + sig_mr + sig_macd
    latest = int(consensus.iloc[-1])

    if   latest >= 2:  label, color = "🟢 STRONG BUY",  C["green"]
    elif latest == 1:  label, color = "🟡 WEAK BUY",    C["amber"]
    elif latest == 0:  label, color = "⚪ HOLD",          C["grey"]
    elif latest == -1: label, color = "🟠 WEAK SELL",   C["orange"]
    else:              label, color = "🔴 STRONG SELL", C["red"]

    metrics = dbc.Row([
        dbc.Col(_card("Momentum",       str(int(sig_mom.iloc[-1]))),  width=3),
        dbc.Col(_card("Mean Reversion", str(int(sig_mr.iloc[-1]))),   width=3),
        dbc.Col(_card("MACD Crossover", str(int(sig_macd.iloc[-1]))), width=3),
        dbc.Col(_card("Consensus Score", str(latest)),                 width=3),
    ], className="g-3")

    banner = html.Div(
        html.Span(f"Current Signal: {label}",
                  style={"fontSize": "1.3rem", "fontWeight": 700, "color": color}),
        style={"background": f"{color}22", "borderLeft": f"4px solid {color}",
               "borderRadius": 8, "padding": "14px 22px"},
    )

    bar_col = [C["green"] if v > 0 else (C["red"] if v < 0 else C["grey"])
               for v in consensus.values]
    fig = go.Figure(go.Bar(x=consensus.index, y=consensus.values,
        marker_color=bar_col, opacity=0.85, showlegend=False))
    for lvl, col, lbl in [(2, C["green"], "STRONG BUY"), (-2, C["red"], "STRONG SELL")]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=col,
            annotation_text=lbl, annotation_position="top right",
            annotation_font_color=col)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
    fig.update_layout(**_dark(height=290,
        yaxis=dict(tickvals=[-3, -2, -1, 0, 1, 2, 3], title="Consensus Score")))
    _axes(fig)
    return [f"🎯 {selected} — Consensus Signal (Ensemble)", metrics, banner, fig]


# ── Callback: AI Analysis ─────────────────────────────────────
@app.callback(
    Output("ai-title", "children"),
    Output("ai-out",   "children"),
    Input("ai-btn",    "n_clicks"),
    State("dd-asset",     "value"),
    State("dd-strategy",  "value"),
    prevent_initial_call=True,
)
def cb_ai(_, selected, strategy_name):
    df = load_ohlcv(selected)
    df = add_indicators(df)
    sig_mom  = MomentumStrategy(params={"short_window": 20, "long_window": 50}).generate_signals(df)
    sig_mr   = MeanReversionStrategy(params={"window": 20, "z_threshold": 1.5}).generate_signals(df)
    sig_macd = MACDCrossoverStrategy(params={}).generate_signals(df)
    consensus = sig_mom + sig_mr + sig_macd
    latest = int(consensus.iloc[-1])
    if   latest >= 2:  signal_label = "🟢 STRONG BUY"
    elif latest == 1:  signal_label = "🟡 WEAK BUY"
    elif latest == 0:  signal_label = "⚪ HOLD"
    elif latest == -1: signal_label = "🟠 WEAK SELL"
    else:              signal_label = "🔴 STRONG SELL"

    recent = df.tail(10)[["close", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower"]].round(2)
    prompt = f"""You are a quantitative financial analyst. Analyze {selected} ETF.

Market data (last 10 days):
{recent.to_string()}

Metrics: Close=${df['close'].iloc[-1]:.2f}, RSI={df['rsi'].iloc[-1]:.1f}, 1D={df['return_1d'].iloc[-1]:.2%}, 5D={df['return_5d'].iloc[-1]:.2%}
Consensus: {latest}/3, Signal: {signal_label}
Momentum return: {backtest(df['close'], sig_mom)['total_return']:+.1%}
Mean Reversion return: {backtest(df['close'], sig_mr)['total_return']:+.1%}
MACD return: {backtest(df['close'], sig_macd)['total_return']:+.1%}

Provide: 1) Market conditions, 2) Signal interpretation, 3) Risk considerations, 4) Outlook. Max 200 words."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ["🤖 AI Market Analysis",
                dbc.Alert("GROQ_API_KEY not set in environment.", color="danger")]
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile", max_tokens=1024,
            messages=[{"role": "user", "content": prompt}])
        content = resp.choices[0].message.content
    except Exception as e:
        return ["🤖 AI Market Analysis",
                dbc.Alert(f"AI request failed: {e}", color="danger")]

    return [f"🤖 {selected} — AI Market Analysis",
            dbc.Alert(content, color="info")]


if __name__ == "__main__":
    app.run(debug=True, port=8502)
