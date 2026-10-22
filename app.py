"""SV Equity Research Analyzer — live tracking, buy/sell signals and NIFTY comparison.

Run:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analysis as A
import data as D
import patterns as P
import scalp as SC
import symbols as S

APP_NAME = "SV Equity Research Analyzer"
st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")

UP, DOWN, NEUTRAL = "#16a34a", "#dc2626", "#64748b"


# ----------------------------------------------------------------------------
# Cached data access
# ----------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def get_history(sym, period):
    return D.history(sym, period=period)


@st.cache_data(ttl=20, show_spinner=False)
def get_intraday(sym, interval, days):
    return D.intraday(sym, interval, days)


@st.cache_data(ttl=1, show_spinner=False)
def get_quote(sym):
    return D.live_quote(sym)


FREQ = {"1m": "1min", "5m": "5min", "15m": "15min"}


def apply_tick(idf: pd.DataFrame, price: float, interval: str) -> pd.DataFrame:
    """Push the latest live price into the intraday candles so the chart and the
    signal move every tick instead of waiting for the next Yahoo candle."""
    if idf.empty or not D.market_status()[0]:
        return idf
    df = idf.copy()
    bar = pd.Timestamp.now(tz="Asia/Kolkata").floor(FREQ[interval])
    if bar > df.index[-1]:
        df.loc[bar] = [price, price, price, price, 0.0]
    else:
        last = df.index[-1]
        df.loc[last, "Close"] = price
        df.loc[last, "High"] = max(df.loc[last, "High"], price)
        df.loc[last, "Low"] = min(df.loc[last, "Low"], price)
    return df


def live_daily(df: pd.DataFrame, q: dict | None) -> pd.DataFrame:
    """Overwrite (or add) today's daily candle with the live quote so every daily
    indicator, signal, level and pattern moves with the market."""
    if df.empty or not q or not D.market_status()[0]:
        return df
    if not all(q.get(k) for k in ("last", "open", "day_high", "day_low")):
        return df
    out = df.copy()
    today = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
    vals = [float(q["open"]), float(q["day_high"]), float(q["day_low"]), float(q["last"])]
    if out.index[-1].normalize() == today:
        out.loc[out.index[-1], ["Open", "High", "Low", "Close"]] = vals
    else:
        out.loc[today] = vals + [0.0]
    return out


def show_chart(fig, key: str):
    """Plot that keeps the viewer's zoom/pan while the data refreshes underneath."""
    fig.update_layout(uirevision=f"{key}-{symbol}")
    st.plotly_chart(fig, use_container_width=True, key=key)


def stamp(prefix: str = "Updated"):
    st.caption(f"🔄 {prefix} {pd.Timestamp.now(tz='Asia/Kolkata'):%H:%M:%S} IST")


# Colour scheme: green = buy, red = sell, grey = wait
SIGNAL_BG = {"STRONG BUY": "#047857", "BUY": "#16a34a", "HOLD / WAIT": "#64748b",
             "SELL": "#dc2626", "STRONG SELL": "#991b1b"}

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700;9..144,800&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
/* Brand typography: Fraunces (wordmark/headings), Manrope (UI), JetBrains Mono (prices) */
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select,
[data-testid="stMetricLabel"], [data-baseweb="tab"] {
  font-family: 'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}
h1, h2, h3, .sv-word {
  font-family: 'Fraunces', Georgia, 'Times New Roman', serif !important; letter-spacing: -.01em;
}
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  font-variant-numeric: tabular-nums; }
.sv-brand { display:flex; align-items:center; gap:14px; flex-wrap:wrap;
  padding:14px 18px; margin:-6px 0 12px; border-radius:14px;
  background: linear-gradient(120deg, #0b1f17 0%, #11402c 55%, #0b1f17 100%); color:#f4f1e8; }
.sv-mono { flex:none; width:52px; height:52px; border-radius:12px; display:grid; place-items:center;
  background:#f4f1e8; color:#0b1f17; font-family:'Fraunces',Georgia,serif; font-weight:800;
  font-size:24px; letter-spacing:-.04em; box-shadow: inset 0 -3px 0 #16a34a; }
.sv-word { font-size: clamp(1.35rem, 3.2vw, 2.1rem); font-weight:700; line-height:1.05; margin:0; }
.sv-word em { font-style: italic; font-weight:500; color:#86efac; }
.sv-tag { font-size:.78rem; letter-spacing:.14em; text-transform:uppercase; opacity:.75; margin-top:4px; }
.sv-side { font-family:'Fraunces',Georgia,serif; font-weight:700; font-size:1.25rem; line-height:1.15; }
.sv-side small { display:block; font-family:'Manrope',sans-serif; font-weight:600; font-size:.68rem;
  letter-spacing:.14em; text-transform:uppercase; opacity:.65; margin-top:2px; }
.sig-badge{border-radius:14px;padding:14px 20px;color:#fff;margin-bottom:8px}
.sig-badge .lbl{font-size:.85em;opacity:.9}
.sig-badge .big{font-size:2.3em;font-weight:800;letter-spacing:.5px;line-height:1.15}
.tick-up{animation:flashG 1s ease-out}
.tick-down{animation:flashR 1s ease-out}
@keyframes flashG{from{background:rgba(22,163,74,.35)}to{background:transparent}}
@keyframes flashR{from{background:rgba(220,38,38,.35)}to{background:transparent}}
/* seamless live updates: don't fade old content or show the running indicator */
[data-stale="true"], .stale-element, [data-testid="stale-element"] { opacity: 1 !important; filter: none !important;
  transition: none !important; }
[data-testid="stStatusWidget"] { visibility: hidden; }
.hold-row{display:grid;grid-template-columns:1.1fr 1fr 1.2fr 1fr;gap:6px;padding:7px 10px;align-items:center;
  border-bottom:1px solid rgba(100,116,139,.15);font-variant-numeric:tabular-nums}
.hold-row.head{font-size:.75em;text-transform:uppercase;letter-spacing:.08em;opacity:.65}
@media (max-width: 640px){ .hold-row{grid-template-columns:1fr 1fr;} }
.live-dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#16a34a;
  margin-right:6px;animation:pulse 1.2s infinite}
@keyframes pulse{0%{opacity:1}50%{opacity:.25}100%{opacity:1}}
</style>
""", unsafe_allow_html=True)


def signal_badge(verdict: str, score: float, label: str, conf: float | None = None) -> str:
    arrow = "▲" if "BUY" in verdict else "▼" if "SELL" in verdict else "■"
    text = verdict if verdict != "HOLD / WAIT" else "WAIT"
    sub = (f"Confidence <b>{conf:.0f}%</b>" if conf is not None else f"Score <b>{score:+.0f}</b>")
    if verdict == "HOLD / WAIT":
        sub += " · no strong signal, don't trade"
    return (f"<div class='sig-badge' style='background:{SIGNAL_BG[verdict]}'>"
            f"<div class='lbl'>{label}</div><div class='big'>{arrow} {text}</div>"
            f"<div>{sub}</div></div>")


@st.cache_data(ttl=600, show_spinner=False)
def get_many(symbols, period):
    return D.history_many(list(symbols), period)


@st.cache_data(ttl=86400, show_spinner="Loading NSE stock list…")
def get_stock_list():
    return S.load_all()


@st.cache_data(ttl=600, show_spinner=False)
def yahoo_search(q):
    return S.search_yahoo(q)


@st.cache_data(ttl=3600, show_spinner=False)
def long_history(symbols):
    return D.history_many(list(symbols), "1y")


def merged_history(symbols):
    """1-year history (cached 1 h) + the last 5 days re-downloaded now, so today's live candle is current."""
    long_ = long_history(symbols)
    try:
        fresh = D.history_many(list(symbols), "5d")
    except Exception:
        fresh = {}
    out = {}
    for s, df in long_.items():
        f = fresh.get(s)
        if f is not None and len(f):
            df = pd.concat([df[df.index.normalize() < f.index[0].normalize()], f])
        out[s] = df
    return out


@st.cache_data(ttl=55, show_spinner=False)
def scan_chunk(symbols, fresh_days, vol_mult):
    """Download one batch of stocks and return breakout rows for it."""
    rows = []
    for s, sdf in merged_history(symbols).items():
        if len(sdf) < 60:
            continue
        try:
            sd = A.add_indicators(sdf)
            summ = A.summarize_breakouts(A.detect_breakouts(sdf, fresh_days, vol_mult, d=sd))
        except Exception:
            continue
        if summ:
            summ.update(symbol=s, price=float(sdf.Close.iloc[-1]),
                        day_pct=float(sdf.Close.pct_change().iloc[-1] * 100),
                        score=float(A.score_series(sd).iloc[-1]), asof=sdf.index[-1])
            cs = A.confidence_series(sd).iloc[-1]
            px, stop = summ["price"], summ["stop"]
            r_ = abs(px - stop)
            sign = 1 if summ["side"] == "BUY" else -1
            summ.update(confidence=float(cs["bull"] if summ["side"] == "BUY" else cs["bear"]),
                        t1=px + sign * r_, t2=px + sign * 2 * r_, t3=px + sign * 3 * r_)
            rows.append(summ)
    return rows


@st.cache_data(ttl=110, show_spinner=False)
def pattern_scan_chunk(symbols, min_conf):
    rows = []
    for s, sdf in merged_history(symbols).items():
        if len(sdf) < 60:
            continue
        try:
            sd = A.add_indicators(sdf)
            cs = A.confidence_series(sd).iloc[-1]
            r = P.all_patterns(sdf, candle_lookback=1)
        except Exception:
            continue
        for p in r["chart"] + r["harmonic"] + r["candles"]:
            confirmed = p["kind"] == "Candlestick" or "✅" in p["status"]
            if p["bias"] == "Neutral" or not confirmed:
                continue
            if p["kind"] == "Candlestick" and p["reliability"] != "High":
                continue
            conf = float(cs["bull"] if p["bias"] == "Bullish" else cs["bear"])
            if p["kind"] == "Candlestick":  # stop beyond the pattern candle, target 2× risk
                bar_, at_ = sdf.loc[p["when"]], float(sd["ATR"].iloc[-1])
                px_ = float(sdf.Close.iloc[-1])
                stop_ = (max(float(bar_["Low"]) - 0.25 * at_, px_ - 3 * at_) if p["bias"] == "Bullish"
                         else min(float(bar_["High"]) + 0.25 * at_, px_ + 3 * at_))
                p = {**p, "stop": stop_, "target": px_ + 2 * (px_ - stop_)}
            rows.append({"symbol": s, "pattern": p["name"], "kind": p["kind"], "bias": p["bias"],
                         "status": p["status"], "reliability": p["reliability"], "confidence": conf,
                         "price": float(sdf.Close.iloc[-1]), "stop": p.get("stop"), "target": p.get("target"),
                         "when": p["when"]})
    return rows


@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentals(sym):
    return D.fundamentals(sym)


def pct(x, digits=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.{digits}f}%"


def inr(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"₹{x:,.2f}"


def verdict_color(v: str) -> str:
    return UP if "BUY" in v else DOWN if "SELL" in v else NEUTRAL


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sv-side'>SV Equity Research<br>Analyzer<small>NSE · Live</small></div>",
                unsafe_allow_html=True)
    period = st.selectbox("History", ["6mo", "1y", "2y", "5y", "max"], index=2)
    bench_name = st.selectbox("Main benchmark", list(D.BENCHMARKS), index=0)
    refresh = st.select_slider("Live refresh", options=[0, 1, 2, 3, 5, 10, 30, 60], value=2,
                               format_func=lambda s: "Off" if s == 0 else f"every {s}s",
                               help="1–2 s is near-live. If NSE starts blocking, use 3–5 s.")
    live_iv = st.selectbox("Live signal candle", ["1m", "5m", "15m"], index=1,
                           help="Timeframe for the live BUY/SELL badge at the top")
    st.divider()
    st.subheader("Risk settings")
    capital = st.number_input("Capital (₹)", 1000, 100_000_000, 100_000, step=10_000)
    risk_pct = st.slider("Risk per trade (% of capital)", 0.25, 5.0, 1.0, 0.25)
    min_conf = st.slider("Only show BUY/SELL above confidence", 50, 90, 60, 5, format="%d%%",
                         help="Confidence = how much of the indicator weight agrees with the signal. "
                              "Below this the app says WAIT.")
    rf = st.number_input("Risk-free rate (%/yr)", 0.0, 15.0, 6.5, 0.25) / 100
    st.divider()
    is_open, status = D.market_status()
    st.caption(f"{'🟢' if is_open else '🔴'} {status} · {pd.Timestamp.now(tz='Asia/Kolkata'):%d %b %Y %H:%M} IST")
    st.caption("⚠️ Educational tool, not investment advice. Signals are rule-based and can be wrong.")

bench_sym = D.BENCHMARKS[bench_name]
# Fast refresh while NSE is open; once a minute when closed (nothing changes then)
LIVE_EVERY = (refresh if D.market_status()[0] else max(refresh, 60)) if refresh else None


# ----------------------------------------------------------------------------
# Stock search (type a few letters of the name or symbol)
# ----------------------------------------------------------------------------
stocks, list_source = get_stock_list()
labels = (stocks["SYMBOL"] + "  ·  " + stocks["NAME"]).tolist()
sym_of = dict(zip(labels, stocks["SYMBOL"]))
if "symbol" not in st.session_state:
    st.session_state.symbol = "RELIANCE"
extra = st.session_state.get("extra_label")
if extra and extra not in sym_of:
    labels.insert(0, extra)
    sym_of[extra] = extra.split("  ·  ")[0]
current = next((l for l, s_ in sym_of.items() if s_ == st.session_state.symbol), labels[0])

pick = st.selectbox("🔍 Search NSE stock", labels, index=labels.index(current),
                    placeholder="Type name or symbol, e.g. 'tata', 'hdfc', 'infy'…",
                    help=f"{list_source}. Start typing to filter.", key="stock_pick")
st.session_state.symbol = sym_of[pick]
with st.expander("Can't find it? Search by company name online"):
    q = st.text_input("Company name", key="ysearch", placeholder="e.g. mazagon dock")
    if q.strip():
        hits = yahoo_search(q.strip())
        if not hits:
            st.caption("No NSE matches found.")
        for hs, hn in hits:
            if st.button(f"{hs} · {hn}", key=f"hit_{hs}"):
                st.session_state.extra_label = f"{hs}  ·  {hn}"
                st.session_state.symbol = hs
                st.session_state.pop("stock_pick", None)
                st.rerun()
symbol = st.session_state.symbol


# ----------------------------------------------------------------------------
# Live quote header (auto-refreshing fragment)
# ----------------------------------------------------------------------------
@st.fragment(run_every=LIVE_EVERY)
def live_header():
    cols = st.columns([2.2, 1.3, 1, 1, 1])
    try:
        q = get_quote(symbol)
    except Exception as e:
        cols[0].error(f"Could not get a quote for **{symbol}**: {e}")
        return
    # Tick direction vs the previous refresh -> green/red flash
    key = f"last_px_{symbol}"
    prev = st.session_state.get(key)
    st.session_state[key] = q["last"]
    tick = "" if prev is None or prev == q["last"] else ("tick-up" if q["last"] > prev else "tick-down")
    arrow = "" if not tick else ("▲" if tick == "tick-up" else "▼")
    day_color = UP if q["change"] >= 0 else DOWN
    px_color = UP if tick == "tick-up" else DOWN if tick == "tick-down" else day_color
    is_open = D.market_status()[0]
    live = "<span class='live-dot'></span>LIVE" if (is_open and refresh) else "⏸ market closed" if not is_open else "auto-refresh off"
    cols[0].markdown(
        f"<div class='{tick}' style='border-radius:10px;padding:4px 8px'>"
        f"<b style='font-size:1.3em'>{q['name']}</b> <span style='color:{NEUTRAL};font-size:0.8em'>NSE: {symbol}</span><br>"
        f"<span style='font-size:2.3em;font-weight:800;color:{px_color}'>{inr(q['last'])} {arrow}</span> "
        f"<span style='background:{day_color};color:#fff;border-radius:6px;padding:2px 8px;font-weight:700'>"
        f"{q['change']:+,.2f} ({q['pchange']:+.2f}%)</span><br>"
        f"<span style='font-size:0.8em;color:{NEUTRAL}'>{live} · {q['source']} · {q['time']}</span></div>",
        unsafe_allow_html=True,
    )
    # Live signal: intraday candles + the tick just received
    try:
        idf = apply_tick(get_intraday(symbol, live_iv, 5), q["last"], live_iv)
        if len(idf) >= 30:
            lp = A.trade_plan(A.add_indicators(idf, intraday=True), capital, risk_pct, min_conf)
            cols[1].markdown(signal_badge(lp["verdict"], lp["score"], f"Live signal ({live_iv})", lp["confidence"]),
                             unsafe_allow_html=True)
    except Exception:
        cols[1].caption("Live signal unavailable")
    cols[2].metric("Day range", f"{q['day_low']:,.1f} – {q['day_high']:,.1f}",
                   f"Open {q['open']:,.1f}", delta_color="off")
    cols[3].metric("52W range", f"{q['low_52w']:,.0f} – {q['high_52w']:,.0f}",
                   f"{(q['last'] / q['high_52w'] - 1) * 100:.1f}% from high", delta_color="off")
    try:
        n = get_quote(bench_sym)
        cols[4].metric(bench_name, f"{n['last']:,.2f}", f"{n['pchange']:+.2f}%")
    except Exception:
        cols[4].metric(bench_name, "—")


st.markdown(
    "<div class='sv-brand'><div class='sv-mono'>SV</div><div>"
    "<div class='sv-word'>SV Equity Research <em>Analyzer</em></div>"
    "<div class='sv-tag'>NSE live signals · breakouts · patterns · scalping</div></div></div>",
    unsafe_allow_html=True)
live_header()

# ----------------------------------------------------------------------------
# Load daily data
# ----------------------------------------------------------------------------
with st.spinner(f"Loading {symbol} history…"):
    raw = get_history(symbol, period)
    bench_raw = get_history(bench_sym, period)
if raw.empty:
    st.error(f"No data for **{symbol}**. Check the NSE symbol (e.g. RELIANCE, TATASTEEL, HDFCBANK).")
    st.stop()

d = A.add_indicators(raw)
plan = A.trade_plan(d, capital, risk_pct, min_conf)

tab_sig, tab_pos, tab_brk, tab_pat, tab_scalp, tab_intra, tab_cmp, tab_scan, tab_bt, tab_fund = st.tabs(
    ["🎯 Signal & Chart", "💼 My Positions", "🚀 Breakouts", "🕯️ Patterns", "⚡ Scalping", "⏱️ Intraday Live", "⚖️ vs NIFTY & Peers", "🔎 Scanner", "🧪 Backtest", "🏢 Fundamentals"]
)


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def price_chart(df: pd.DataFrame, plan: dict | None, intraday: bool = False, height: int = 860):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.55, 0.13, 0.16, 0.16])
    fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close,
                                 name="Price", increasing_line_color=UP, decreasing_line_color=DOWN), 1, 1)
    lines = [("SMA20", "#f59e0b", 1), ("SMA50", "#3b82f6", 1.3), ("SMA200", "#a855f7", 1.6)]
    if intraday:
        lines = [("EMA9", "#f59e0b", 1), ("EMA21", "#3b82f6", 1.3), ("VWAP", "#0ea5e9", 2)]
    for col, c, w in lines:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=c, width=w)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df.BB_up, name="BB upper", line=dict(color="#94a3b8", width=0.8, dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df.BB_low, name="BB lower", line=dict(color="#94a3b8", width=0.8, dash="dot"),
                             fill="tonexty", fillcolor="rgba(148,163,184,0.08)"), 1, 1)
    st_up = df.ST.where(df.ST_dir == 1)
    st_dn = df.ST.where(df.ST_dir == -1)
    fig.add_trace(go.Scatter(x=df.index, y=st_up, name="Supertrend ↑", line=dict(color=UP, width=1.2)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=st_dn, name="Supertrend ↓", line=dict(color=DOWN, width=1.2)), 1, 1)

    # Buy / sell markers where the composite score crosses the thresholds
    sc = A.score_series(df)
    buys = df.index[(sc >= 15) & (sc.shift() < 15)]
    sells = df.index[(sc <= -15) & (sc.shift() > -15)]
    fig.add_trace(go.Scatter(x=buys, y=df.Low.loc[buys] * 0.98, mode="markers+text", name="BUY signal",
                             text=["BUY"] * len(buys), textposition="bottom center",
                             textfont=dict(color=UP, size=11), marker=dict(symbol="triangle-up", size=14, color=UP)), 1, 1)
    fig.add_trace(go.Scatter(x=sells, y=df.High.loc[sells] * 1.02, mode="markers+text", name="SELL signal",
                             text=["SELL"] * len(sells), textposition="top center",
                             textfont=dict(color=DOWN, size=11), marker=dict(symbol="triangle-down", size=14, color=DOWN)), 1, 1)

    # Background shading: green while the signal is BUY, red while it is SELL
    regime = pd.Series(np.where(sc >= 15, 1, np.where(sc <= -15, -1, 0)), index=df.index)
    block = (regime != regime.shift()).cumsum()
    for _, seg in regime.groupby(block):
        if seg.iloc[0] == 0:
            continue
        fig.add_vrect(x0=seg.index[0], x1=seg.index[-1], row=1, col=1, line_width=0, layer="below",
                      fillcolor=UP if seg.iloc[0] > 0 else DOWN, opacity=0.08)

    if plan:
        for lvl, txt, c in [(plan["stop_loss"], "Stop-loss", DOWN), (plan["target1"], "Target 1", UP),
                            (plan["target2"], "Target 2", UP), (plan["target3"], "Target 3", UP)]:
            fig.add_hline(y=lvl, line=dict(color=c, width=1, dash="dash"), row=1, col=1,
                          annotation_text=f"{txt} {lvl:,.1f}", annotation_position="right")
        for lv, n in plan["supports"][:2]:
            fig.add_hline(y=lv, line=dict(color="#22c55e", width=0.8, dash="dot"), row=1, col=1,
                          annotation_text=f"S {lv:,.0f}", annotation_position="left")
        for lv, n in plan["resistances"][:2]:
            fig.add_hline(y=lv, line=dict(color="#ef4444", width=0.8, dash="dot"), row=1, col=1,
                          annotation_text=f"R {lv:,.0f}", annotation_position="left")

    vcol = np.where(df.Close >= df.Open, UP, DOWN)
    fig.add_trace(go.Bar(x=df.index, y=df.Volume, name="Volume", marker_color=vcol, opacity=0.6), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df.RSI, name="RSI", line=dict(color="#8b5cf6")), 3, 1)
    for y, c in [(70, DOWN), (30, UP)]:
        fig.add_hline(y=y, line=dict(color=c, width=0.8, dash="dash"), row=3, col=1)
    hcol = np.where(df.MACD_hist >= 0, UP, DOWN)
    fig.add_trace(go.Bar(x=df.index, y=df.MACD_hist, name="MACD hist", marker_color=hcol), 4, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df.MACD, name="MACD", line=dict(color="#3b82f6", width=1)), 4, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df.MACD_signal, name="Signal", line=dict(color="#f59e0b", width=1)), 4, 1)

    fig.update_layout(height=height, xaxis_rangeslider_visible=False, hovermode="x unified",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.02, x=0))
    for i, t in enumerate(["Price ₹", "Volume", "RSI", "MACD"], start=1):
        fig.update_yaxes(title_text=t, row=i, col=1)
    if intraday:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[15.5, 9.25], pattern="hour")])
    else:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def plan_panel(plan: dict, label: str):
    v = plan["verdict"]
    st.markdown(signal_badge(v, plan["score"], f"{label} signal", plan["confidence"]), unsafe_allow_html=True)
    st.progress(int(plan["confidence"]), text=f"Confidence {plan['confidence']:.0f}% "
                f"(needs {plan['min_conf']:.0f}% for a BUY/SELL call)")
    lo, hi = plan["entry_zone"]
    buy_side = plan["score"] >= 0
    trade = v != "HOLD / WAIT"
    zc = (UP if buy_side else DOWN) if trade else NEUTRAL
    zone = ("🟢 BUY zone" if buy_side else "🔴 SELL / exit zone") if trade else "⏸ No trade — levels below are for reference"
    st.markdown(f"<div style='background:{zc}1a;border-left:5px solid {zc};padding:8px 12px;border-radius:6px'>"
                f"<b style='color:{zc}'>{zone}</b>"
                + (f": {inr(lo)} – {inr(hi)}" if trade else "") + "</div>", unsafe_allow_html=True)
    m = st.columns(4)
    m[0].metric("🛑 Stop-loss", inr(plan["stop_loss"]), pct(plan["stop_loss"] / plan["price"] - 1), delta_color="off")
    for i in (1, 2, 3):
        m[i].metric(f"🎯 Target {i}", inr(plan[f"target{i}"]), pct(plan[f"target{i}"] / plan["price"] - 1), delta_color="off")
    st.caption("Targets are 1×, 2× and 3× the risk (entry to stop). Book part profit at T1 and move the stop to entry.")
    if trade and buy_side:
        st.markdown(
            f"**Position size:** {plan['qty']:,} shares (≈ {inr(plan['capital_needed'])}) · "
            f"max loss at stop ≈ {inr(plan['max_loss'])} ({risk_pct}% of capital)"
        )
    elif trade:
        st.markdown("**If you hold it:** consider exiting or tightening your stop. "
                    f"Short-side stop (for F&O traders) at {inr(plan['stop_loss'])}.")


KIND_STYLE = {
    "price": ("#0f172a", "#ffffff", "●"), "resistance": ("rgba(220,38,38,.10)", DOWN, "R"),
    "support": ("rgba(22,163,74,.10)", UP, "S"), "target": ("rgba(4,120,87,.20)", "#047857", "🎯"),
    "stop": ("rgba(153,27,27,.20)", "#991b1b", "🛑"), "info": ("transparent", NEUTRAL, "·"),
}


def levels_ladder(levels: pd.DataFrame):
    """Price ladder: every key level sorted high to low, coloured by role."""
    html = ["<div style='font-variant-numeric:tabular-nums;border:1px solid rgba(100,116,139,.25);"
            "border-radius:10px;overflow:hidden;font-size:.92em'>"]
    for _, r in levels.iterrows():
        bg, fg, icon = KIND_STYLE[r["kind"]]
        bold = "font-weight:800;font-size:1.05em" if r["kind"] in ("price", "target", "stop") else "font-weight:600"
        dist = "" if r["kind"] == "price" else f"{r['Distance %']:+.2f}%"
        note = f"<div style='font-size:.78em;opacity:.75'>{r['note']}</div>" if r["note"] else ""
        html.append(
            f"<div style='display:flex;align-items:center;gap:10px;padding:6px 12px;background:{bg};"
            f"color:{fg if r['kind'] == 'price' else 'inherit'};border-bottom:1px solid rgba(100,116,139,.15)'>"
            f"<span style='width:1.6em;text-align:center;color:{fg};font-weight:800'>{icon}</span>"
            f"<div style='flex:1;min-width:0'><span style='{bold};color:{fg}'>{r['Level']}</span>{note}</div>"
            f"<span style='{bold}'>₹{r['Price']:,.2f}</span>"
            f"<span style='width:4.5em;text-align:right;opacity:.8'>{dist}</span></div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _row_colour(row):
    v = row["_r"]
    bg = "background-color: rgba(22,163,74,.18)" if v > 0 else "background-color: rgba(220,38,38,.18)" if v < 0 else ""
    return [bg] * len(row)


def reasons_table(plan: dict):
    rows = pd.DataFrame(plan["reasons"])
    rows["Signal"] = rows["reading"].map(lambda v: "🟢 Bullish" if v > 0 else "🔴 Bearish" if v < 0 else "⚪ Neutral")
    t = rows[["factor", "Signal", "points", "weight", "detail", "reading"]].rename(
        columns={"factor": "Factor", "points": "Points", "weight": "Max", "detail": "Reading", "reading": "_r"})
    styled = t.style.apply(_row_colour, axis=1).format({"Points": "{:+.1f}"}).hide(axis="columns", subset=["_r"])
    st.dataframe(styled, hide_index=True, use_container_width=True)


def fmt_days(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    x = max(1, round(x))
    return f"{x} day" if x == 1 else f"{x} days"


def hold_panel(d_: pd.DataFrame, plan_: dict):
    est = A.holding_estimate(d_, plan_, min_conf)
    v = plan_["verdict"]
    buy = est["side"] > 0
    col = UP if buy else DOWN
    if v == "HOLD / WAIT":
        head = ("⏸ <b>No new trade now.</b> If you already hold this stock, add it in 💼 My Positions "
                "to get live HOLD / SELL advice.")
        col = NEUTRAL
    elif buy:
        head = (f"⏳ Hold for about <b>{fmt_days(est['hold_min'])} to {fmt_days(est['hold_max'])}</b> "
                f"(Target 1 → Target 2). Sell if Target 1 isn't reached within "
                f"<b>{est['time_stop_days']} trading days</b> "
                f"(by {A._bdays_from_today(est['time_stop_days']):%a %d %b}).")
    else:
        head = ("🔻 <b>Sell / exit now if you hold it.</b> For F&O short trades: expected "
                f"{fmt_days(est['hold_min'])} to {fmt_days(est['hold_max'])} to targets, "
                f"time stop {est['time_stop_days']} trading days.")
    rows = ["<div class='hold-row head'><span>Target</span><span>Typical time</span><span>Expected by</span>"
            "<span>Reached in the past</span></div>"]
    for k in (1, 2, 3):
        lo, hi = est[f"t{k}_range"]
        rate = est[f"t{k}_rate"]
        rows.append(
            f"<div class='hold-row'><span>🎯 T{k} <b>₹{plan_[f'target{k}']:,.2f}</b></span>"
            f"<span><b>{fmt_days(est[f't{k}_days'])}</b> <span style='opacity:.65;font-size:.85em'>"
            f"({max(1, round(lo))}–{max(1, round(hi))})</span></span>"
            f"<span>{est[f't{k}_date']:%a %d %b %Y}</span>"
            f"<span>{'—' if not np.isfinite(rate) else f'{rate:.0f}% of past signals'}</span></div>")
    basis = (f"Based on <b>{est['n_signals']}</b> past {'BUY' if buy else 'SELL'} signals on this stock "
             f"(stop hit first in {est['stop_rate']:.0f}%; a signal usually lasts "
             f"{fmt_days(est['signal_days'])})." if est["method"] == "history" else
             "Not enough past signals on this stock: times are estimated from its daily volatility.")
    st.markdown(
        f"<div style='border:2px solid {col};border-radius:12px;overflow:hidden;margin:6px 0 10px'>"
        f"<div style='background:{col}1f;padding:10px 12px'>{head}</div>{''.join(rows)}"
        f"<div style='padding:7px 10px;font-size:.8em;opacity:.75'>{basis} Counted in trading days "
        f"(Mon–Fri). Updates live.</div></div>", unsafe_allow_html=True)
    return est


# ----------------------------------------------------------------------------
# Tab: Signal & Chart (daily)
# ----------------------------------------------------------------------------
with tab_sig:
    @st.fragment(run_every=LIVE_EVERY)
    def signal_view():
        try:
            q_now = get_quote(symbol)
        except Exception:
            q_now = None
        is_open_now = D.market_status()[0]
        raw_l = live_daily(raw, q_now)
        d_l = A.add_indicators(raw_l)
        plan_l = A.trade_plan(d_l, capital, risk_pct, min_conf)
        stamp("Live signal updated" if is_open_now else "Market closed · last update")
        left, right = st.columns([1, 1])
        with left:
            plan_panel(plan_l, "Daily (swing)")
            st.markdown("**How long to hold**")
            hold_panel(d_l, plan_l)
            st.markdown("**Key information**")
            stats = A.key_stats(d_l, plan_l, q_now)
            kc = st.columns(2)
            for i, (k_, v_) in enumerate(stats.items()):
                kc[i % 2].markdown(f"<div style='font-size:.8em;opacity:.7'>{k_}</div>"
                                   f"<div style='font-weight:700;margin-bottom:8px'>{v_}</div>", unsafe_allow_html=True)
        with right:
            st.markdown("**Key levels** · 🔴 resistance above · 🟢 support below")
            levels_ladder(A.key_levels(d_l, plan_l, q_now, is_open_now))
            st.caption("Day high/low are live. Pivots use the previous session" +
                       ("" if is_open_now else " (shown for the next session)") +
                       ". Zones are ±¼ ATR around levels price has reversed from before.")
        try:
            pnow = P.all_patterns(raw_l, candle_lookback=3)
            chips = []
            for p in pnow["chart"] + pnow["harmonic"] + pnow["candles"]:
                if p["kind"] != "Candlestick" and "✅" not in p["status"]:
                    continue
                col_ = UP if p["bias"] == "Bullish" else DOWN if p["bias"] == "Bearish" else NEUTRAL
                chips.append(f"<span style='background:{col_};color:#fff;border-radius:12px;padding:2px 10px;"
                             f"margin:2px;display:inline-block;font-size:.85em'>{p['name']}</span>")
            if chips:
                st.markdown("**Patterns now:** " + "".join(chips[:8]) + " <span style='font-size:.8em;opacity:.7'>"
                            "(details in 🕯️ Patterns tab)</span>", unsafe_allow_html=True)
        except Exception:
            pass
        with st.expander("Why this signal? (indicator breakdown)"):
            reasons_table(plan_l)
        lookback = st.select_slider("Chart window", ["3M", "6M", "1Y", "2Y", "All"], value="1Y", key="sig_win")
        n = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "All": len(d_l)}[lookback]
        show_chart(price_chart(d_l.tail(n), plan_l), "chart1")

    signal_view()


# ----------------------------------------------------------------------------
# Tab: My Positions (live HOLD / BOOK / SELL advice)
# ----------------------------------------------------------------------------
import json as _json
import os as _os

POS_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "my_positions.json")


def load_positions() -> list:
    if "positions" not in st.session_state:
        try:
            with open(POS_FILE) as fh:
                st.session_state.positions = _json.load(fh)
        except Exception:
            st.session_state.positions = []
    return st.session_state.positions


def save_positions(pos_list: list):
    st.session_state.positions = pos_list
    try:
        with open(POS_FILE, "w") as fh:
            _json.dump(pos_list, fh)
    except Exception:
        pass


with tab_pos:
    positions = load_positions()
    with st.expander("➕ Add a stock you bought", expanded=not positions):
        with st.form("add_pos", clear_on_submit=False):
            f1, f2 = st.columns(2)
            p_label = f1.selectbox("Stock", labels, index=labels.index(current), key="pos_stock")
            try:
                _px = float(get_quote(sym_of[p_label])["last"])
            except Exception:
                _px = 0.0
            p_price = f2.number_input("Buy price (₹)", min_value=0.0, value=round(_px, 2), step=0.05, key="pos_price")
            f3, f4, f5 = st.columns(3)
            p_qty = f3.number_input("Quantity", min_value=1, value=1, step=1, key="pos_qty")
            p_date = f4.date_input("Buy date", value=pd.Timestamp.now(tz="Asia/Kolkata").date(), key="pos_date")
            p_stop = f5.number_input("Your stop-loss (₹, 0 = auto)", min_value=0.0, value=0.0, step=0.05, key="pos_stop")
            if st.form_submit_button("Add position", type="primary"):
                if p_price > 0:
                    positions = positions + [{"symbol": sym_of[p_label], "price": float(p_price), "qty": int(p_qty),
                                              "date": str(p_date), "stop": float(p_stop) or None}]
                    save_positions(positions)
                    st.success(f"Added {sym_of[p_label]}. Live advice is below.")
                else:
                    st.error("Enter the price you bought at.")

    @st.fragment(run_every=LIVE_EVERY)
    def positions_view():
        pos_list = load_positions()
        if not pos_list:
            st.info("No positions yet. Add a stock you hold above and the app will tell you, live, "
                    "whether to HOLD, BOOK PROFIT or SELL, and how many days are left.")
            return
        stamp("Positions updated")
        total_cost = total_val = 0.0
        cards = []
        for idx_, p in enumerate(pos_list):
            try:
                hist = get_history(p["symbol"], "2y")
                q = get_quote(p["symbol"])
                hd = A.add_indicators(live_daily(hist, q))
                pl = A.trade_plan(hd, capital, risk_pct, min_conf)
                est = A.holding_estimate(hd, pl, min_conf)
                adv = A.position_advice(hd, pl, p["price"], p["date"], q["last"], p.get("stop"),
                                        est["time_stop_days"], min_conf)
            except Exception as e:
                cards.append((idx_, p, None, str(e)))
                continue
            total_cost += p["price"] * p["qty"]
            total_val += adv["price"] * p["qty"]
            cards.append((idx_, p, adv, est))
        if total_cost:
            pnl = total_val - total_cost
            pc = UP if pnl >= 0 else DOWN
            st.markdown(
                f"<div style='display:flex;gap:18px;flex-wrap:wrap;align-items:baseline;margin-bottom:8px'>"
                f"<span style='font-size:.85em;opacity:.7'>Invested</span><b>₹{total_cost:,.0f}</b>"
                f"<span style='font-size:.85em;opacity:.7'>Now</span><b>₹{total_val:,.0f}</b>"
                f"<span style='font-size:.85em;opacity:.7'>P&L</span>"
                f"<b style='color:{pc};font-size:1.3em'>{pnl:+,.0f} ({pnl / total_cost * 100:+.2f}%)</b></div>",
                unsafe_allow_html=True)
        for idx_, p, adv, est in cards:
            if adv is None:
                st.warning(f"{p['symbol']}: could not get live data ({est}).")
                continue
            pc = UP if adv["pnl_pct"] >= 0 else DOWN
            ticks = "".join(
                f"<span style='margin-right:12px'>{'✅' if adv['high_since'] >= adv[t] else '🎯'} "
                f"{t.upper()} <b>₹{adv[t]:,.2f}</b></span>" for t in ("t1", "t2", "t3"))
            days_txt = (f"Held <b>{adv['days_held']}</b> of ~{est['time_stop_days']} trading days · "
                        + (f"T1 usually in {fmt_days(est['t1_days'])}" if adv["high_since"] < adv["t1"]
                           else f"T2 usually by day {round(est['t2_days'])}"))
            st.markdown(
                f"<div style='border:2px solid {adv['color']};border-radius:14px;overflow:hidden;margin-bottom:10px'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;"
                f"background:{adv['color']};color:#fff;padding:10px 14px'>"
                f"<div><b style='font-size:1.15em'>{p['symbol']}</b> · {p['qty']} @ ₹{p['price']:,.2f} "
                f"<span style='opacity:.85;font-size:.85em'>(bought {p['date']})</span></div>"
                f"<div style='font-size:1.4em;font-weight:800;letter-spacing:.3px'>{adv['action']}</div></div>"
                f"<div style='padding:10px 14px;display:grid;gap:6px'>"
                f"<div>{adv['why']}</div>"
                f"<div style='font-variant-numeric:tabular-nums'>Live <b>₹{adv['price']:,.2f}</b> · P&L "
                f"<b style='color:{pc}'>{(adv['price'] - p['price']) * p['qty']:+,.2f} ({adv['pnl_pct']:+.2f}%)</b>"
                f" · 🛑 Stop now <b style='color:{DOWN}'>₹{adv['stop']:,.2f}</b>"
                f"{' (trailing)' if adv['stop'] > adv['init_stop'] else ''}</div>"
                f"<div style='font-variant-numeric:tabular-nums'>{ticks}</div>"
                f"<div style='font-size:.85em;opacity:.8'>⏳ {days_txt} · {adv['stage']}</div></div></div>",
                unsafe_allow_html=True)
            if st.button(f"Remove {p['symbol']}", key=f"rm_{idx_}_{p['symbol']}"):
                save_positions([x for j, x in enumerate(pos_list) if j != idx_])
                st.rerun()
        st.caption("Rules: sell at the stop · sell ⅓ at Target 1 and move the stop to your buy price · sell ⅓ at "
                   "Target 2 and move the stop to Target 1 · sell the rest at Target 3 or when the trend turns to "
                   "SELL · sell if Target 1 hasn't come within the usual time for this stock. Positions are saved on "
                   "the device running the app.")

    positions_view()


# ----------------------------------------------------------------------------
# Tab: Breakouts (automatic scan)
# ----------------------------------------------------------------------------
with tab_brk:
    st.markdown("Stocks that **just broke** a resistance or support level. "
                "🟢 **BUY** = breakout upward · 🔴 **SELL** = breakdown. Updates automatically.")
    b1, b2, b3, b4 = st.columns([1.4, 1, 1, 1])
    universe_name = b1.selectbox("Stocks to scan", ["Major stocks (~110)", "Scanner watchlist",
                                                    "All NSE stocks (slow, few minutes)"], key="brk_univ")
    fresh_days = b2.selectbox("Broke within", [1, 2, 3, 5], index=2,
                              format_func=lambda n: "today" if n == 1 else f"last {n} days", key="brk_fresh")
    show = b3.selectbox("Show", ["BUY + SELL", "BUY only", "SELL only"], key="brk_show")
    major_only = b4.toggle("Major only", value=False, key="brk_major",
                           help="Strong volume, 52-week levels or several breakouts at once")
    t1_, t2_ = st.columns(2)
    rescan_min = t1_.selectbox("Auto re-scan", [1, 2, 5, 0], index=0, key="brk_every",
                               format_func=lambda m: "Off" if m == 0 else f"every {m} min",
                               help="Every minute during market hours. 'All NSE stocks' needs 5 min.")
    brk_strong = t2_.toggle(f"Only strong (≥ {min_conf}% confidence)", value=True, key="brk_strong",
                            help="Hide breakouts that the trend indicators don't back up")

    if universe_name.startswith("Major"):
        universe = sorted(S.BUILTIN)
    elif universe_name.startswith("Scanner"):
        universe = list(D.DEFAULT_WATCHLIST)
    else:
        universe = stocks["SYMBOL"].tolist()

    brk_every = None
    if rescan_min:
        brk_every = max(rescan_min, 5 if universe_name.startswith("All") else 0) * 60
        if not D.market_status()[0]:
            brk_every = max(brk_every, 900)

    @st.fragment(run_every=brk_every)
    def breakout_view():
        chunks = [tuple(universe[i:i + 100]) for i in range(0, len(universe), 100)]
        rows = []
        bar = st.progress(0, text=f"Scanning {len(universe):,} stocks…") if len(chunks) > 1 else None
        for i, ch in enumerate(chunks):
            rows += scan_chunk(ch, fresh_days, 1.5)
            if bar:
                bar.progress((i + 1) / len(chunks), text=f"Scanned {min((i + 1) * 100, len(universe)):,} / {len(universe):,}")
        if bar:
            bar.empty()
        res = pd.DataFrame(rows)
        if res.empty:
            st.info("No fresh breakouts or breakdowns right now in this list. Try 'last 5 days' or a bigger list.")
            return
        n_buy, n_sell = int((res.side == "BUY").sum()), int((res.side == "SELL").sum())
        st.markdown(
            f"<div style='display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 10px'>"
            f"<span style='background:{UP};color:#fff;padding:6px 14px;border-radius:20px;font-weight:700'>▲ {n_buy} BUY breakouts</span>"
            f"<span style='background:{DOWN};color:#fff;padding:6px 14px;border-radius:20px;font-weight:700'>▼ {n_sell} SELL breakdowns</span>"
            f"<span style='color:{NEUTRAL};align-self:center;font-size:.85em'>scanned {pd.Timestamp.now(tz='Asia/Kolkata'):%H:%M} IST · "
            f"{int(res.major.sum())} major</span></div>", unsafe_allow_html=True)
        if show == "BUY only":
            res = res[res.side == "BUY"]
        elif show == "SELL only":
            res = res[res.side == "SELL"]
        if major_only:
            res = res[res.major]
        if brk_strong:
            res = res[res.confidence >= min_conf]
        if res.empty:
            st.info("Nothing matches these filters.")
            return
        res = res.sort_values(["major", "confidence", "strength"], ascending=False)
        out = pd.DataFrame({
            "Signal": [("🟢 " if s == "BUY" else "🔴 ") + ("MAJOR " if m else "") + s for s, m in zip(res.side, res.major)],
            "Stock": res.symbol,
            "Price ₹": res.price.round(2),
            "Today %": res.day_pct.round(2),
            "What happened": res.types,
            "Level broken ₹": res.level.round(2),
            "Beyond level %": res.beyond_pct.round(2),
            "Volume × avg": res.vol_x.round(1),
            "When": res.days_ago.map(lambda n: "today" if n == 0 else f"{n} day{'s' if n > 1 else ''} ago"),
            "Confidence": res.confidence.round(0),
            "Stop-loss ₹": res.stop.round(2),
            "Target 1 ₹": res.t1.round(2),
            "Target 2 ₹": res.t2.round(2),
            "Target 3 ₹": res.t3.round(2),
        })

        def colour_signal(v):
            bg = SIGNAL_BG["STRONG BUY" if "MAJOR BUY" in v else "BUY" if "BUY" in v else
                           "STRONG SELL" if "MAJOR SELL" in v else "SELL"]
            return f"background-color:{bg};color:white;font-weight:700"

        def colour_num(v):
            return f"color:{UP if v > 0 else DOWN if v < 0 else NEUTRAL};font-weight:600"

        styled = (out.style.map(colour_signal, subset=["Signal"])
                  .map(colour_num, subset=["Today %", "Beyond level %"])
                  .map(lambda v: f"color:{UP if v >= 75 else '#ca8a04' if v >= 60 else NEUTRAL};font-weight:700",
                       subset=["Confidence"])
                  .map(lambda v: f"color:{DOWN};font-weight:700", subset=["Stop-loss ₹"])
                  .map(lambda v: f"color:{UP};font-weight:600", subset=["Target 1 ₹", "Target 2 ₹", "Target 3 ₹"])
                  .format({"Price ₹": "{:,.2f}", "Level broken ₹": "{:,.2f}", "Stop-loss ₹": "{:,.2f}",
                           "Target 1 ₹": "{:,.2f}", "Target 2 ₹": "{:,.2f}", "Target 3 ₹": "{:,.2f}",
                           "Today %": "{:+.2f}", "Beyond level %": "{:+.2f}",
                           "Volume × avg": "{:.1f}×", "Confidence": "{:.0f}%"}))
        st.dataframe(styled, hide_index=True, use_container_width=True, height=min(640, 38 + 35 * len(out)))

        o1, o2 = st.columns([3, 1])
        pick_b = o1.selectbox("Open a stock in full analysis", out["Stock"].tolist(), key="brk_open")
        if o2.button("Open ➜", type="primary", key="brk_go"):
            name = stocks.loc[stocks.SYMBOL == pick_b, "NAME"]
            st.session_state.extra_label = f"{pick_b}  ·  {name.iloc[0] if len(name) else pick_b}"
            st.session_state.symbol = pick_b
            st.session_state.pop("stock_pick", None)
            st.rerun()
        st.caption("Confidence = how much of the 9 trend indicators agree with the breakout direction. "
                   "During market hours today's candle is live, so a breakout can still fail by the close.")

    breakout_view()


# ----------------------------------------------------------------------------
# Tab: Patterns
# ----------------------------------------------------------------------------
ABBR = {"Bullish Engulfing": "BuE", "Bearish Engulfing": "BeE", "Bullish Harami": "BuH", "Bearish Harami": "BeH",
        "Hammer": "Ham", "Shooting Star": "SS", "Morning Star": "MS", "Evening Star": "ES", "Doji": "Dj",
        "Bullish Elephant Bar": "🐘↑", "Bearish Elephant Bar": "🐘↓"}


def pattern_chart(df: pd.DataFrame, res: dict, bars: int = 150):
    view = df.tail(bars)
    start = view.index[0]
    fig = go.Figure(go.Candlestick(x=view.index, open=view.Open, high=view.High, low=view.Low, close=view.Close,
                                   name="Price", increasing_line_color=UP, decreasing_line_color=DOWN))
    zz = [(df.index[i], p) for i, p, _ in P.zigzag(df, 2.0) if df.index[i] >= start]
    if zz:
        fig.add_trace(go.Scatter(x=[z[0] for z in zz], y=[z[1] for z in zz], mode="lines",
                                 line=dict(color="#94a3b8", width=1, dash="dot"), name="Swings"))
    for p in res["chart"] + res["harmonic"]:
        col = UP if p["bias"] == "Bullish" else DOWN if p["bias"] == "Bearish" else "#ca8a04"
        pts = [(t, v) for t, v in p["points"] if t >= start]
        if len(pts) >= 2:
            fig.add_trace(go.Scatter(x=[t for t, _ in pts], y=[v for _, v in pts], mode="lines+markers+text",
                                     text=p.get("labels", [""] * len(pts))[-len(pts):], textposition="top center",
                                     line=dict(color=col, width=2.5), marker=dict(size=7), name=p["name"]))
        for ln in p.get("lines", []):
            fig.add_trace(go.Scatter(x=[t for t, _ in ln], y=[v for _, v in ln], mode="lines", showlegend=False,
                                     line=dict(color=col, width=1.5, dash="dash")))
        if "d_zone" in p:
            fig.add_hrect(y0=p["d_zone"][0], y1=p["d_zone"][1], fillcolor=col, opacity=0.12, line_width=0,
                          annotation_text=f"D zone ({p['name'].replace('Potential ', '')})")
    cs = [c for c in res["candles"] if c["when"] >= start]
    for c in cs:
        up = c["bias"] == "Bullish"
        y = view.Low.loc[c["when"]] * 0.99 if up else view.High.loc[c["when"]] * 1.01
        fig.add_annotation(x=c["when"], y=y, text=ABBR.get(c["name"], c["name"][:3]), showarrow=True, arrowhead=2,
                           ay=25 if up else -25, font=dict(size=10, color=UP if up else DOWN if c["bias"] == "Bearish" else NEUTRAL))
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.02, x=0), hovermode="x unified")
    return fig


def pattern_card(p: dict):
    col = UP if p["bias"] == "Bullish" else DOWN if p["bias"] == "Bearish" else "#ca8a04"
    arrow = "▲ BUY setup" if p["bias"] == "Bullish" else "▼ SELL setup" if p["bias"] == "Bearish" else "◆ Wait for breakout"
    lv = ""
    if "trigger" in p:
        lv = (f"<div style='display:flex;gap:14px;flex-wrap:wrap;margin-top:6px;font-variant-numeric:tabular-nums'>"
              f"<span>Trigger <b>₹{p['trigger']:,.2f}</b></span>"
              f"<span style='color:{DOWN}'>🛑 Stop <b>₹{p['stop']:,.2f}</b></span>"
              f"<span style='color:{UP}'>🎯 Target <b>₹{p['target']:,.2f}</b></span></div>")
    ratios = ""
    if p.get("ratios"):
        ratios = "<div style='font-size:.78em;opacity:.7;margin-top:4px'>Ratios: " + \
                 ", ".join(f"{k}={v}" for k, v in p["ratios"].items()) + "</div>"
    st.markdown(
        f"<div style='border-left:6px solid {col};background:{col}14;border-radius:8px;padding:10px 14px;margin-bottom:8px'>"
        f"<div style='display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap'>"
        f"<b style='font-size:1.08em;color:{col}'>{p['name']}</b>"
        f"<span style='background:{col};color:#fff;border-radius:12px;padding:1px 10px;font-weight:700;font-size:.85em'>{arrow}</span></div>"
        f"<div style='font-size:.85em;margin-top:2px'>{p['status']} · reliability {p['reliability']} · "
        f"{p['when']:%d %b %H:%M}</div>"
        f"<div style='font-size:.9em;margin-top:4px'>{p['meaning']}</div>{lv}{ratios}</div>",
        unsafe_allow_html=True)


with tab_pat:
    pat_tf = st.selectbox("Timeframe", ["Daily", "15 min", "5 min"], key="pat_tf")
    PAT_EVERY = max(LIVE_EVERY, 10) if LIVE_EVERY else None

    @st.fragment(run_every=PAT_EVERY)
    def pattern_view():
        try:
            q_p = get_quote(symbol)
        except Exception:
            q_p = None
        if pat_tf == "Daily":
            pdf = live_daily(raw, q_p)
        else:
            iv_ = "15m" if pat_tf == "15 min" else "5m"
            try:
                pdf = get_intraday(symbol, iv_, 30 if pat_tf == "15 min" else 10)
                if q_p:
                    pdf = apply_tick(pdf, q_p["last"], iv_)
            except Exception:
                pdf = pd.DataFrame()
        stamp("Patterns re-read")
        if pdf.empty or len(pdf) < 40:
            st.info("Not enough candles for pattern reading on this timeframe.")
        else:
            pres = P.all_patterns(pdf, candle_lookback=5)
            lean_col = UP if pres["lean"] == "Bullish" else DOWN if pres["lean"] == "Bearish" else NEUTRAL
            st.markdown(
                f"<div style='background:{lean_col};color:#fff;border-radius:12px;padding:10px 16px;margin-top:4px'>"
                f"<b style='font-size:1.3em'>Patterns lean {pres['lean'].upper()}</b><br>"
                f"Bullish weight {pres['bull_pts']} vs bearish {pres['bear_pts']}</div>", unsafe_allow_html=True)
            show_chart(pattern_chart(pdf, pres), "chart2")
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Chart & harmonic patterns**")
                big_pats = pres["chart"] + pres["harmonic"]
                if not big_pats:
                    st.caption("No chart or harmonic pattern on the latest swings.")
                for p in sorted(big_pats, key=lambda x: ("✅" not in x["status"], x["reliability"] != "High")):
                    pattern_card(p)
            with g2:
                st.markdown("**Candlestick patterns (last 5 candles)**")
                if not pres["candles"]:
                    st.caption("No candlestick pattern in the last 5 candles.")
                else:
                    ct = pd.DataFrame(pres["candles"]).sort_values("when", ascending=False)
                    ct = pd.DataFrame({"When": ct["when"].dt.strftime("%d %b %H:%M"), "Pattern": ct["name"],
                                       "Signal": ct["bias"].map({"Bullish": "🟢 Bullish", "Bearish": "🔴 Bearish", "Neutral": "⚪ Neutral"}),
                                       "Reliability": ct["reliability"], "Meaning": ct["meaning"]})
                    st.dataframe(ct.style.map(lambda v: f"color:{UP if 'Bull' in v else DOWN if 'Bear' in v else NEUTRAL};font-weight:700",
                                              subset=["Signal"]), hide_index=True, use_container_width=True)
            st.caption("Patterns are shapes, not guarantees: they also appear by chance. Trust a pattern more when it is "
                       "✅ confirmed, has High reliability, and the 🎯 Signal tab agrees with ≥ "
                       f"{min_conf}% confidence. Covered: 32 candlestick patterns incl. Elephant Bar, Head & Shoulders, "
                       "double/triple tops & bottoms, triangles, wedges, channels, rectangles, flags, and harmonic "
                       "Gartley, Bat, Alt Bat, Butterfly, Crab, Deep Crab, Cypher, Shark and AB=CD.")

    pattern_view()
    st.divider()
    st.markdown("**🔎 Find stocks with confirmed patterns today**")
    ps1, ps2 = st.columns([2, 1])
    pat_univ = ps1.selectbox("Stocks", ["Major stocks (~110)", "All NSE stocks (slow)"], key="pat_univ")
    pat_strong = ps2.toggle(f"Only ≥ {min_conf}% confidence", value=True, key="pat_strong")
    PAT_SCAN_EVERY = (120 if D.market_status()[0] else None) if pat_univ.startswith("Major") else None

    @st.fragment(run_every=PAT_SCAN_EVERY)
    def pattern_scan_view():
        if not (pat_univ.startswith("Major") or st.button("Scan all NSE stocks", type="primary", key="pat_go")):
            return
        stamp("Pattern scan (auto every 2 min for major stocks)")
        uni = sorted(S.BUILTIN) if pat_univ.startswith("Major") else stocks["SYMBOL"].tolist()
        rows, chunks = [], [tuple(uni[i:i + 100]) for i in range(0, len(uni), 100)]
        bar = st.progress(0, text="Scanning…")
        for i, ch in enumerate(chunks):
            rows += pattern_scan_chunk(ch, min_conf)
            bar.progress((i + 1) / len(chunks), text=f"Scanned {min((i + 1) * 100, len(uni)):,} / {len(uni):,}")
        bar.empty()
        pr = pd.DataFrame(rows)
        if not pr.empty and pat_strong:
            pr = pr[pr.confidence >= min_conf]
        if pr.empty:
            st.info("No confirmed patterns matching the filters right now.")
        else:
            pr = pr.sort_values(["confidence"], ascending=False)
            show_t = pd.DataFrame({
                "Signal": np.where(pr.bias == "Bullish", "🟢 BUY", "🔴 SELL"), "Stock": pr.symbol,
                "Pattern": pr.pattern, "Type": pr.kind, "Status": pr.status, "Confidence": pr.confidence.round(0),
                "Price ₹": pr.price.round(2), "Stop ₹": pr.stop.astype(float).round(2), "Target ₹": pr.target.astype(float).round(2),
            })
            st.dataframe(show_t.style.map(lambda v: f"background-color:{UP if 'BUY' in v else DOWN};color:white;font-weight:700",
                                          subset=["Signal"]).format({"Confidence": "{:.0f}%", "Price ₹": "{:,.2f}",
                                                                     "Stop ₹": "{:,.2f}", "Target ₹": "{:,.2f}"}, na_rep="—"),
                         hide_index=True, use_container_width=True)

    pattern_scan_view()


# ----------------------------------------------------------------------------
# Tab: Scalping
# ----------------------------------------------------------------------------
with tab_scalp:
    sc1, sc2 = st.columns(2)
    scalp_tf = sc1.selectbox("Candle", ["1 min", "3 min", "5 min"], key="scalp_tf")
    scalp_conf = sc2.slider("Minimum confidence for a scalp call", 50, 90, max(70, min_conf), 5,
                            format="%d%%", key="scalp_conf",
                            help="Scalping is noisy; 70%+ cuts the number of weak calls a lot.")

    @st.fragment(run_every=LIVE_EVERY)
    def scalp_view():
        try:
            base = get_intraday(symbol, "5m" if scalp_tf == "5 min" else "1m", 5 if scalp_tf != "5 min" else 10)
        except Exception as e:
            st.warning(f"1-minute data unavailable: {e}")
            return
        if base.empty or len(base) < 60:
            st.info("Not enough 1-minute candles yet.")
            return
        try:
            base = apply_tick(base, get_quote(symbol)["last"], "5m" if scalp_tf == "5 min" else "1m")
        except Exception:
            pass
        sdf = SC.resample(base, 3) if scalp_tf == "3 min" else base
        r = SC.scalp_analysis(sdf, scalp_conf)
        tf_min = {"1 min": 1, "3 min": 3, "5 min": 5}[scalp_tf]
        bg = SIGNAL_BG["BUY"] if r["side"] > 0 else SIGNAL_BG["SELL"] if r["side"] < 0 else SIGNAL_BG["HOLD / WAIT"]
        arrow = "▲" if r["side"] > 0 else "▼" if r["side"] < 0 else "■"
        k1, k2 = st.columns([1, 1.2])
        with k1:
            st.markdown(
                f"<div class='sig-badge' style='background:{bg}'><div class='lbl'>Scalp signal ({scalp_tf})</div>"
                f"<div class='big'>{arrow} {r['call']}</div><div>Confidence <b>{r['confidence']:.0f}%</b>"
                f"{' · ' + r['wait_reason'] if r['wait_reason'] else ''}</div></div>", unsafe_allow_html=True)
            side_word = "BUY" if (r["side"] or r["lead"]) > 0 else "SELL"
            note = "" if r["side"] else " (reference only)"
            m = st.columns(4)
            m[0].metric(f"Entry{note}", f"{r['price']:,.2f}")
            m[1].metric("🛑 Stop", f"{r['stop']:,.2f}", f"{(r['stop'] / r['price'] - 1) * 100:+.2f}%", delta_color="off")
            m[2].metric("🎯 T1 (1R)", f"{r['targets'][0]:,.2f}", f"{(r['targets'][0] / r['price'] - 1) * 100:+.2f}%", delta_color="off")
            m[3].metric("🎯 T2 / T3", f"{r['targets'][1]:,.1f} / {r['targets'][2]:,.1f}")
            sa, st_ = r["stats_all"], r["stats_today"]
            wr = "—" if np.isnan(sa["win_rate"]) else f"{sa['win_rate']:.0f}%"
            wr_t = "—" if np.isnan(st_["win_rate"]) else f"{st_['win_rate']:.0f}%"
            good = not np.isnan(sa["win_rate"]) and sa["win_rate"] >= 50
            st.markdown(
                f"<div style='border:1px solid {UP if good else DOWN};border-radius:10px;padding:8px 12px;margin-top:6px'>"
                f"<b>How these scalp calls did on {symbol}</b><br>"
                f"Last 5 days: <b>{sa['signals']}</b> calls · reached T1 before stop <b style='color:{UP if good else DOWN}'>{wr}</b>"
                f" · avg {sa['avg_r']:+.2f}R<br>Today: <b>{st_['signals']}</b> calls · T1 first {wr_t}"
                + (f"<br>⏳ Typical hold: <b>~{sa['win_bars'] * tf_min:.0f} min</b> to T1 · exit if T1 hasn't come in "
                   f"<b>{max(sa['win_bars_p75'] * tf_min * 1.5, tf_min * 3):.0f} min</b>"
                   if np.isfinite(sa.get('win_bars', np.nan)) else "") + "</div>",
                unsafe_allow_html=True)
            st.caption(f"Rules: {side_word} side scored on VWAP, EMA 9/21, RSI(7), volume, 15-bar breakout and candle close. "
                       "Stop behind the last 5 candles; T1/T2/T3 = 1×/1.5×/2× risk. Exit by 3:15 pm. "
                       "If the hit rate above is under 50%, scalping this stock with these rules has been losing.")
        with k2:
            rt = pd.DataFrame(r["reasons"])
            st.dataframe(rt.style.map(lambda v: f"color:{UP if 'Buy' in v else DOWN if 'Sell' in v else NEUTRAL};font-weight:700",
                                      subset=["Favours"]), hide_index=True, use_container_width=True)
        dv = r["data"].tail(150)
        fig = go.Figure(go.Candlestick(x=dv.index, open=dv.Open, high=dv.High, low=dv.Low, close=dv.Close, name="Price",
                                       increasing_line_color=UP, decreasing_line_color=DOWN))
        for col_, c_ in (("VWAP", "#0ea5e9"), ("EMA9", "#f59e0b"), ("EMA21", "#3b82f6")):
            fig.add_trace(go.Scatter(x=dv.index, y=dv[col_], name=col_, line=dict(color=c_, width=1.5)))
        ev = r["events"]
        if len(ev):
            ev = ev[ev.time >= dv.index[0]]
            for sd_, sym_, col_ in (("BUY", "triangle-up", UP), ("SELL", "triangle-down", DOWN)):
                e = ev[ev.side == sd_]
                fig.add_trace(go.Scatter(x=e.time, y=e.entry, mode="markers", name=f"Scalp {sd_}",
                                         marker=dict(symbol=sym_, size=12, color=col_, line=dict(width=1, color="white")),
                                         text=e.outcome, hovertemplate="%{text}<extra></extra>"))
        if r["side"]:
            fig.add_hline(y=r["stop"], line=dict(color=DOWN, dash="dash"), annotation_text="Stop")
            for j, t in enumerate(r["targets"], 1):
                fig.add_hline(y=t, line=dict(color=UP, dash="dot"), annotation_text=f"T{j}")
        fig.update_layout(height=520, xaxis_rangeslider_visible=False, hovermode="x unified",
                          margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.02, x=0))
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[15.5, 9.25], pattern="hour")])
        show_chart(fig, "chart3")
        if len(r["events"]):
            with st.expander("Recent scalp calls and what happened"):
                e = r["events"].tail(25).iloc[::-1].copy()
                e["time"] = e["time"].dt.strftime("%d %b %H:%M")
                st.dataframe(e.round(2).style.map(lambda v: f"background-color:{UP if v == 'BUY' else DOWN};color:white;font-weight:700",
                                                  subset=["side"]), hide_index=True, use_container_width=True)

    scalp_view()


# ----------------------------------------------------------------------------
# Tab: Intraday live
# ----------------------------------------------------------------------------
with tab_intra:
    c1, c2 = st.columns(2)
    interval = c1.selectbox("Candle", ["1m", "5m", "15m"], index=1)
    days = c2.selectbox("Days", [1, 2, 5], index=2)

    @st.fragment(run_every=LIVE_EVERY)
    def intraday_view():
        try:
            idf = get_intraday(symbol, interval, days)
        except Exception as e:
            st.warning(f"Intraday data unavailable: {e}")
            return
        if idf.empty or len(idf) < 30:
            st.info("Not enough intraday candles yet (market closed or just opened).")
            return
        try:
            idf = apply_tick(idf, get_quote(symbol)["last"], interval)
        except Exception:
            pass
        di = A.add_indicators(idf, intraday=True)
        ip = A.trade_plan(di, capital, risk_pct, min_conf)
        a, b = st.columns([1, 1.15])
        with a:
            plan_panel(ip, f"Intraday ({interval})")
            last = di.iloc[-1]
            above = last.Close > last.VWAP
            st.markdown(f"Price is **{'above' if above else 'below'} VWAP** ({inr(last.VWAP)}) — "
                        f"{'buyers in control today' if above else 'sellers in control today'}.")
        with b:
            reasons_table(ip)
        show_chart(price_chart(di, ip, intraday=True, height=760), "chart4")
        st.caption(f"Last candle {di.index[-1]:%d %b %H:%M} IST · refreshes "
                   f"{'every %ss' % refresh if refresh else 'manually'} · Yahoo intraday can lag NSE by a minute or two.")

    intraday_view()


# ----------------------------------------------------------------------------
# Tab: Comparison
# ----------------------------------------------------------------------------
with tab_cmp:
    c1, c2 = st.columns([1, 1])
    benches = c1.multiselect("Compare with indices", list(D.BENCHMARKS), default=[bench_name])
    peers_txt = c2.text_input("…and peer stocks (comma separated)", "TCS, INFY" if symbol != "TCS" else "INFY, HCLTECH")
    peers = [D.to_nse(p) for p in peers_txt.split(",") if p.strip()]

    series = {symbol: raw.Close}
    for b in benches:
        bd = bench_raw if b == bench_name else get_history(D.BENCHMARKS[b], period)
        if not bd.empty:
            series[b] = bd.Close
    if peers:
        for p, pdf in get_many(tuple(peers), period).items():
            series[p] = pdf.Close
    closes = pd.DataFrame(series)
    closes.index = closes.index.normalize()
    closes = closes.groupby(level=0).last().ffill().dropna()

    if closes.empty or bench_raw.empty:
        st.warning("Not enough overlapping data to compare.")
    else:
        rebased = closes / closes.iloc[0] * 100
        fig = go.Figure()
        for col in rebased:
            fig.add_trace(go.Scatter(x=rebased.index, y=rebased[col], name=col,
                                     line=dict(width=3 if col == symbol else 1.6)))
        fig.update_layout(title="Performance (rebased to 100)", height=420, hovermode="x unified",
                          margin=dict(l=10, r=10, t=40, b=10))
        show_chart(fig, "chart5")

        b_close = bench_raw.Close.copy()
        b_close.index = b_close.index.normalize()
        s_close = raw.Close.copy()
        s_close.index = s_close.index.normalize()
        cm = A.compare(s_close, b_close.groupby(level=0).last(), rf)

        st.subheader(f"{symbol} vs {bench_name}")
        k = st.columns(6)
        k[0].metric("Beta", f"{cm['beta']:.2f}", help="1.0 = moves with index; >1 more volatile")
        k[1].metric("Correlation", f"{cm['correlation']:.2f}")
        k[2].metric("Alpha / yr", pct(cm["alpha"]), help="Jensen's alpha: return above what beta explains")
        k[3].metric("Up capture", f"{cm['up_capture'] * 100:.0f}%", help="Avg move on index up-days vs index")
        k[4].metric("Down capture", f"{cm['down_capture'] * 100:.0f}%", help="Lower is better")
        k[5].metric("Info ratio", f"{cm['info_ratio']:.2f}")

        stats = pd.DataFrame({
            symbol: [cm["cagr_stock"], cm["vol_stock"], cm["sharpe_stock"], cm["maxdd_stock"]],
            bench_name: [cm["cagr_bench"], cm["vol_bench"], cm["sharpe_bench"], cm["maxdd_bench"]],
        }, index=["CAGR", "Volatility (ann.)", "Sharpe", "Max drawdown"])
        disp = stats.copy().astype(object)
        for r in ["CAGR", "Volatility (ann.)", "Max drawdown"]:
            disp.loc[r] = [pct(v) for v in stats.loc[r]]
        disp.loc["Sharpe"] = [f"{v:.2f}" for v in stats.loc["Sharpe"]]

        rets = pd.DataFrame({c: A.period_returns(closes[c]) for c in closes}).T
        g1, g2 = st.columns([1, 2])
        g1.markdown("**Risk / return**")
        g1.dataframe(disp, use_container_width=True)
        g2.markdown("**Returns by period**")
        g2.dataframe(rets.map(pct), use_container_width=True)

        h1, h2 = st.columns(2)
        rs = cm["relative_strength"]
        f2 = go.Figure(go.Scatter(x=rs.index, y=rs, line=dict(color="#3b82f6"), name="RS"))
        f2.add_hline(y=100, line=dict(dash="dash", color=NEUTRAL))
        f2.update_layout(title=f"Relative strength vs {bench_name} (rising = outperforming)", height=320,
                         margin=dict(l=10, r=10, t=40, b=10))
        h1.plotly_chart(f2, use_container_width=True)
        rb = cm["rolling_beta"].dropna()
        f3 = go.Figure(go.Scatter(x=rb.index, y=rb, line=dict(color="#a855f7"), name="Beta"))
        f3.add_hline(y=1, line=dict(dash="dash", color=NEUTRAL))
        f3.update_layout(title="Rolling 60-day beta", height=320, margin=dict(l=10, r=10, t=40, b=10))
        h2.plotly_chart(f3, use_container_width=True)

        if closes.shape[1] > 2:
            corr = closes.pct_change().corr()
            f4 = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu",
                                      zmin=-1, zmax=1, text=corr.round(2).values, texttemplate="%{text}"))
            f4.update_layout(title="Correlation of daily returns", height=380, margin=dict(l=10, r=10, t=40, b=10))
            show_chart(f4, "chart6")

        verdict_rs = rs.iloc[-1] - rs.iloc[-min(len(rs), 63)]
        st.info(
            f"**Takeaway:** over the last ~3 months {symbol} has "
            f"{'**outperformed**' if verdict_rs > 0 else '**underperformed**'} {bench_name} "
            f"(RS change {verdict_rs:+.1f}). Beta {cm['beta']:.2f} means a 1% move in {bench_name} "
            f"has typically come with a {abs(cm['beta']):.2f}% move in {symbol}."
        )


# ----------------------------------------------------------------------------
# Tab: Scanner
# ----------------------------------------------------------------------------
with tab_scan:
    st.markdown("Scan a watchlist with the same signal engine. Strongest (highest confidence) first.")
    wl = st.text_area("Watchlist (NSE symbols, comma or newline separated)", ", ".join(D.DEFAULT_WATCHLIST), height=90)
    strong_only = st.toggle(f"Only strong signals (≥ {min_conf}% confidence)", value=True, key="scan_strong")
    AUTO_SCAN_EVERY = 60 if D.market_status()[0] else None

    @st.fragment(run_every=AUTO_SCAN_EVERY)
    def scanner_view():
        stamp("Scanned (re-scans every minute while the market is open)")
        syms = tuple(dict.fromkeys(D.to_nse(s) for s in wl.replace("\n", ",").split(",") if s.strip()))
        with st.spinner(f"Scanning {len(syms)} stocks…"):
            many = merged_history(syms)
            b_ret = A.period_returns(bench_raw.Close) if not bench_raw.empty else {}
            rows = []
            for s, sdf in many.items():
                if len(sdf) < 60:
                    continue
                sd = A.add_indicators(sdf)
                sp = A.trade_plan(sd, capital, risk_pct, min_conf)
                pr = A.period_returns(sdf.Close)
                rows.append({
                    "Symbol": s, "Price": sp["price"], "1D %": sdf.Close.pct_change().iloc[-1] * 100,
                    "Confidence %": sp["confidence"], "Signal": sp["verdict"], "RSI": sd.RSI.iloc[-1],
                    f"1M vs {bench_name} (pp)": (pr["1M"] - b_ret.get("1M", np.nan)) * 100,
                    "Stop-loss": sp["stop_loss"], "Target 1": sp["target1"], "Target 2": sp["target2"],
                    "Target 3": sp["target3"],
                })
        if rows:
            res = pd.DataFrame(rows).sort_values("Confidence %", ascending=False)
            if strong_only:
                res = res[res.Signal != "HOLD / WAIT"]
            if res.empty:
                st.info(f"No stock in this list has a BUY/SELL signal above {min_conf}% confidence right now.")

            def _sig_colour(v):
                return f"background-color:{SIGNAL_BG.get(v, NEUTRAL)};color:white;font-weight:700"

            def _pn(v):
                return f"color:{UP if v > 0 else DOWN if v < 0 else NEUTRAL};font-weight:600"

            pn_cols = ["1D %", f"1M vs {bench_name} (pp)"]
            show_res = not res.empty
            res = res.style.map(_sig_colour, subset=["Signal"]).map(_pn, subset=pn_cols)
            if show_res:
              st.dataframe(
                res, hide_index=True, use_container_width=True,
                column_config={
                    "Confidence %": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.0f%%"),
                    "Target 2": st.column_config.NumberColumn(format="₹%.2f"),
                    "Target 3": st.column_config.NumberColumn(format="₹%.2f"),
                    "Price": st.column_config.NumberColumn(format="₹%.2f"),
                    "Stop-loss": st.column_config.NumberColumn(format="₹%.2f"),
                    "Target 1": st.column_config.NumberColumn(format="₹%.2f"),
                    "1D %": st.column_config.NumberColumn(format="%+.2f"),
                    "RSI": st.column_config.NumberColumn(format="%.0f"),
                    f"1M vs {bench_name} (pp)": st.column_config.NumberColumn(format="%+.1f"),
                },
            )
            missing = set(syms) - set(many)
            if missing:
                st.caption("No data for: " + ", ".join(sorted(missing)))
        else:
            st.warning("No data returned for the watchlist.")

    scanner_view()


# ----------------------------------------------------------------------------
# Tab: Backtest
# ----------------------------------------------------------------------------
with tab_bt:
    st.markdown("How would the signal engine have done on this stock? Long-only: buy when the score "
                "crosses the buy level, sell when it falls to the sell level.")
    c1, c2, c3 = st.columns(3)
    buy_th = c1.slider("Buy when score ≥", 0, 80, 15, 5)
    sell_th = c2.slider("Sell when score ≤", -80, 0, -15, 5)
    cost = c3.number_input("Cost per side (%)", 0.0, 1.0, 0.1, 0.05, help="Brokerage + STT + slippage")
    bt = A.backtest(d, buy_th, sell_th, cost)
    k = st.columns(6)
    k[0].metric("Strategy return", pct(bt["total_return"], 1))
    k[1].metric("Buy & hold", pct(bt["bh_return"], 1))
    k[2].metric("Strategy CAGR", pct(bt["cagr"], 1))
    k[3].metric("Max drawdown", pct(bt["max_dd"], 1), f"B&H {pct(bt['bh_max_dd'], 1)}", delta_color="off")
    k[4].metric("Trades", bt["n_trades"])
    k[5].metric("Win rate", "—" if np.isnan(bt["win_rate"]) else f"{bt['win_rate'] * 100:.0f}%")
    f = go.Figure()
    f.add_trace(go.Scatter(x=bt["equity"].index, y=bt["equity"] * 100, name="Signal strategy", line=dict(width=2.5)))
    f.add_trace(go.Scatter(x=bt["buy_hold"].index, y=bt["buy_hold"] * 100, name="Buy & hold", line=dict(color=NEUTRAL)))
    f.update_layout(height=400, hovermode="x unified", title="Growth of ₹100", margin=dict(l=10, r=10, t=40, b=10))
    show_chart(f, "chart7")
    if len(bt["trades"]):
        t = bt["trades"].copy()
        t["entry_date"] = t["entry_date"].dt.date
        t["exit_date"] = t["exit_date"].dt.date
        t["return"] = (t["return"] * 100).round(2)
        st.dataframe(t.rename(columns={"return": "return %"}), hide_index=True, use_container_width=True)
    st.caption("Past performance does not guarantee future results. Indicators need ~200 bars to warm up.")


# ----------------------------------------------------------------------------
# Tab: Fundamentals
# ----------------------------------------------------------------------------
with tab_fund:
    f = get_fundamentals(symbol)
    if not f:
        st.info("Fundamentals unavailable for this symbol right now.")
    else:
        about = f.pop("About", None)
        st.subheader(f.pop("Company", symbol))
        st.caption(f"{f.pop('Sector', '')} · {f.pop('Industry', '')}")
        pct_keys = {"Dividend yield", "ROE", "Net margin", "Revenue growth (YoY)", "Earnings growth (YoY)",
                    "Promoter/insider holding", "Institutional holding"}
        cols = st.columns(4)
        for i, (k, v) in enumerate(f.items()):
            if k == "Market cap":
                txt = f"₹{v / 1e7:,.0f} Cr"
            elif k == "Dividend yield":
                txt = f"{v:.2f}%" if v > 1 else f"{v * 100:.2f}%"  # yfinance changed units over time
            elif k in pct_keys:
                txt = f"{v * 100:.1f}%"
            elif isinstance(v, (int, float)):
                txt = f"{v:,.2f}"
            else:
                txt = str(v)
            cols[i % 4].metric(k, txt)
        if about:
            with st.expander("About the company"):
                st.write(about)
