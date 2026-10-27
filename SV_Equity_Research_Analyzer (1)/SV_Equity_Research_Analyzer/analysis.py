"""Technical analysis, buy/sell signal engine, benchmark comparison and backtest.

Pure pandas/numpy - no network access here, so everything is testable offline.
Input DataFrames have columns: Open, High, Low, Close, Volume (DatetimeIndex).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def _wilder(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0), n)
    loss = _wilder(-delta.clip(upper=0), n)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100).where(loss.notna())


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift()
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1
    ).max(axis=1)
    return _wilder(tr, n)


def adx(df: pd.DataFrame, n: int = 14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    a = atr(df, n)
    plus_di = 100 * _wilder(pd.Series(plus_dm, index=df.index), n) / a
    minus_di = 100 * _wilder(pd.Series(minus_dm, index=df.index), n) / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx, n), plus_di, minus_di


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0):
    """Returns (line, direction) where direction is +1 (up-trend) or -1."""
    a = atr(df, n).to_numpy()
    hl2 = ((df["High"] + df["Low"]) / 2).to_numpy()
    close = df["Close"].to_numpy()
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    fu, fl = upper.copy(), lower.copy()
    line = np.full(len(df), np.nan)
    direction = np.zeros(len(df))
    for i in range(len(df)):
        if np.isnan(a[i]):
            continue
        if i == 0 or np.isnan(line[i - 1]):
            fu[i], fl[i] = upper[i], lower[i]
            direction[i] = 1 if close[i] >= hl2[i] else -1
        else:
            fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
            fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
            if direction[i - 1] == 1:
                direction[i] = -1 if close[i] < fl[i] else 1
            else:
                direction[i] = 1 if close[i] > fu[i] else -1
        line[i] = fl[i] if direction[i] == 1 else fu[i]
    return pd.Series(line, index=df.index), pd.Series(direction, index=df.index)


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP that resets each calendar day (for intraday bars)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    day = df.index.date
    pv = (tp * df["Volume"]).groupby(day).cumsum()
    vol = df["Volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv / vol


def add_indicators(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    for n in (20, 50, 200):
        d[f"SMA{n}"] = c.rolling(n).mean()
    d["EMA9"] = c.ewm(span=9, adjust=False).mean()
    d["EMA21"] = c.ewm(span=21, adjust=False).mean()
    d["RSI"] = rsi(c)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["BB_mid"], d["BB_up"], d["BB_low"] = mid, mid + 2 * sd, mid - 2 * sd
    d["BB_pctB"] = (c - d["BB_low"]) / (d["BB_up"] - d["BB_low"])
    d["ATR"] = atr(d)
    d["ADX"], d["+DI"], d["-DI"] = adx(d)
    d["ST"], d["ST_dir"] = supertrend(d)
    d["VolSMA20"] = d["Volume"].rolling(20).mean()
    d["OBV"] = (np.sign(c.diff()).fillna(0) * d["Volume"]).cumsum()
    if intraday:
        d["VWAP"] = vwap(d)
    return d


# ----------------------------------------------------------------------------
# Support / resistance
# ----------------------------------------------------------------------------
def pivot_points(df: pd.DataFrame, use_last: bool = False) -> dict:
    """Classic floor pivots + CPR from the last completed bar (use_last=True when the
    latest bar is already complete, e.g. after market close -> levels for next session)."""
    bar = df.iloc[-1] if (use_last or len(df) < 2) else df.iloc[-2]
    h, l, c = bar["High"], bar["Low"], bar["Close"]
    p = (h + l + c) / 3
    return {k: float(v) for k, v in {
        "R3": h + 2 * (p - l), "R2": p + (h - l), "R1": 2 * p - l, "Pivot": p,
        "S1": 2 * p - h, "S2": p - (h - l), "S3": l - 2 * (h - p),
        "CPR top": max((h + l) / 2, 2 * p - (h + l) / 2), "CPR bottom": min((h + l) / 2, 2 * p - (h + l) / 2),
    }.items()}


def swing_levels(df: pd.DataFrame, window: int = 5, lookback: int = 250, tol: float = 0.015):
    """Swing highs/lows clustered into price levels. Returns (supports, resistances)."""
    d = df.tail(lookback)
    hi, lo = d["High"], d["Low"]
    span = 2 * window + 1
    is_hi = hi == hi.rolling(span, center=True).max()
    is_lo = lo == lo.rolling(span, center=True).min()
    raw = sorted(list(hi[is_hi].values) + list(lo[is_lo].values))
    levels: list[list[float]] = []
    for p in raw:
        if levels and abs(p - np.mean(levels[-1])) / np.mean(levels[-1]) <= tol:
            levels[-1].append(p)
        else:
            levels.append([p])
    clustered = [(float(np.mean(g)), len(g)) for g in levels]
    price = float(df["Close"].iloc[-1])
    supports = sorted([lv for lv in clustered if lv[0] < price], key=lambda x: -x[0])
    resistances = sorted([lv for lv in clustered if lv[0] > price], key=lambda x: x[0])
    return supports, resistances


# ----------------------------------------------------------------------------
# Signal engine
# ----------------------------------------------------------------------------
WEIGHTS = {
    "Long-term trend (price vs 200 SMA)": 15,
    "Golden/Death cross (50 vs 200 SMA)": 10,
    "Short-term trend (EMA 9 vs 21)": 10,
    "MACD momentum": 10,
    "RSI": 15,
    "Bollinger Bands": 10,
    "Supertrend": 15,
    "ADX trend strength": 10,
    "Volume confirmation": 5,
}


def component_scores(d: pd.DataFrame) -> pd.DataFrame:
    """Each component in [-1, 1]; +1 = bullish, -1 = bearish. Vectorised over all bars."""
    c = d["Close"]
    out = pd.DataFrame(index=d.index)
    out["Long-term trend (price vs 200 SMA)"] = np.sign(c - d["SMA200"])
    out["Golden/Death cross (50 vs 200 SMA)"] = np.sign(d["SMA50"] - d["SMA200"])
    out["Short-term trend (EMA 9 vs 21)"] = np.sign(d["EMA9"] - d["EMA21"])
    h = d["MACD_hist"]
    rising = h.diff() > 0
    out["MACD momentum"] = np.where(h > 0, np.where(rising, 1.0, 0.5), np.where(rising, -0.5, -1.0))
    r = d["RSI"]
    out["RSI"] = np.where(r < 30, 1.0, np.where(r > 70, -1.0, ((r - 50) / 20 * 0.5).clip(-0.5, 0.5)))
    pb = d["BB_pctB"]
    out["Bollinger Bands"] = np.where(pb < 0, 1.0, np.where(pb > 1, -1.0, np.where(pb < 0.2, 0.5, np.where(pb > 0.8, -0.5, 0.0))))
    out["Supertrend"] = d["ST_dir"].replace(0, np.nan)
    out["ADX trend strength"] = np.where(d["ADX"] > 20, np.sign(d["+DI"] - d["-DI"]), 0.0)
    big_vol = d["Volume"] > 1.5 * d["VolSMA20"]
    out["Volume confirmation"] = np.where(big_vol, np.sign(c - d["Open"]), 0.0)
    # Bars without enough history for an indicator contribute 0
    na_mask = d[["SMA200", "SMA50", "RSI", "BB_pctB", "ATR", "ADX"]].isna()
    out.loc[na_mask["SMA200"], ["Long-term trend (price vs 200 SMA)", "Golden/Death cross (50 vs 200 SMA)"]] = 0
    out.loc[na_mask["RSI"], "RSI"] = 0
    out.loc[na_mask["BB_pctB"], "Bollinger Bands"] = 0
    out.loc[na_mask["ADX"], "ADX trend strength"] = 0
    return out.fillna(0.0)


def score_series(d: pd.DataFrame) -> pd.Series:
    comp = component_scores(d)
    total = sum(comp[k] * w for k, w in WEIGHTS.items())
    return total.clip(-100, 100)


def confidence_series(d: pd.DataFrame) -> pd.DataFrame:
    """Share of indicator weight (0-100%) that is bullish and that is bearish, per bar.
    This measures how strongly the indicators AGREE - it is not a probability of profit."""
    comp = component_scores(d)
    bull = sum(comp[k].clip(lower=0) * w for k, w in WEIGHTS.items())
    bear = sum((-comp[k]).clip(lower=0) * w for k, w in WEIGHTS.items())
    return pd.DataFrame({"bull": bull, "bear": bear})


def verdict(score: float, confidence: float | None = None, min_conf: float = 60) -> str:
    """BUY/SELL only when the indicators agree at least `min_conf`% in that direction."""
    if confidence is None:  # legacy score-only mode
        confidence = 100 if abs(score) >= 15 else 0
    if abs(score) < 15 or confidence < min_conf:
        return "HOLD / WAIT"
    strong = confidence >= max(75, min_conf + 10)
    if score > 0:
        return "STRONG BUY" if strong else "BUY"
    return "STRONG SELL" if strong else "SELL"


def _explain(name: str, val: float, row: pd.Series) -> str:
    c = row["Close"]
    texts = {
        "Long-term trend (price vs 200 SMA)": f"Price ₹{c:,.2f} is {'above' if val > 0 else 'below'} 200-SMA ₹{row['SMA200']:,.2f}",
        "Golden/Death cross (50 vs 200 SMA)": f"50-SMA is {'above' if val > 0 else 'below'} 200-SMA ({'golden' if val > 0 else 'death'} cross regime)",
        "Short-term trend (EMA 9 vs 21)": f"EMA9 is {'above' if val > 0 else 'below'} EMA21",
        "MACD momentum": f"MACD histogram {row['MACD_hist']:+.2f}, momentum {'building' if val in (1, -0.5) else 'fading'}",
        "RSI": f"RSI {row['RSI']:.1f}" + (" — oversold" if row["RSI"] < 30 else " — overbought" if row["RSI"] > 70 else ""),
        "Bollinger Bands": f"%B {row['BB_pctB']:.2f}" + (" — below lower band" if row["BB_pctB"] < 0 else " — above upper band" if row["BB_pctB"] > 1 else ""),
        "Supertrend": f"Supertrend is {'UP' if val > 0 else 'DOWN'} (line ₹{row['ST']:,.2f})",
        "ADX trend strength": f"ADX {row['ADX']:.1f}" + (f", {'+DI' if val > 0 else '-DI'} leads" if row["ADX"] > 20 else " — weak/no trend"),
        "Volume confirmation": f"Volume {row['Volume'] / row['VolSMA20']:.1f}× 20-day avg" if row["VolSMA20"] else "Volume n/a",
    }
    return texts.get(name, name)


def trade_plan(d: pd.DataFrame, capital: float = 100000, risk_pct: float = 1.0, min_conf: float = 60) -> dict:
    """Latest signal plus entry zone, stop-loss, targets and position size."""
    comp = component_scores(d)
    score = float(score_series(d).iloc[-1])
    cs = confidence_series(d).iloc[-1]
    confidence = float(cs["bull"] if score >= 0 else cs["bear"])
    row = d.iloc[-1]
    price, a = float(row["Close"]), float(row["ATR"]) if pd.notna(row["ATR"]) else float(row["Close"] * 0.02)
    supports, resistances = swing_levels(d)
    sup = supports[0][0] if supports else None
    res = resistances[0][0] if resistances else None

    reasons = []
    for k, w in WEIGHTS.items():
        v = float(comp[k].iloc[-1])
        reasons.append({"factor": k, "weight": w, "reading": v, "points": round(v * w, 1),
                        "detail": _explain(k, v, row) if v != 0 or k in ("RSI", "Bollinger Bands", "ADX trend strength") else "Neutral / not enough data"})

    long_side = score >= 0
    if long_side:
        stop = price - 2 * a
        if sup is not None and price - sup < 2 * a:
            stop = sup - 0.5 * a
        risk = price - stop
        t1, t2, t3 = price + risk, price + 2 * risk, price + 3 * risk
        entry_low = max(stop + 0.5 * risk, price - 0.5 * a)
        entry = (entry_low, price)
    else:
        stop = price + 2 * a
        if res is not None and res - price < 2 * a:
            stop = res + 0.5 * a
        risk = stop - price
        t1, t2, t3 = price - risk, price - 2 * risk, price - 3 * risk
        entry = (price, min(stop - 0.5 * risk, price + 0.5 * a))

    qty = int((capital * risk_pct / 100) // risk) if risk > 0 else 0
    qty = min(qty, int(capital // price)) if price > 0 else 0
    return {
        "score": round(score, 1), "confidence": round(confidence, 0), "min_conf": min_conf,
        "verdict": verdict(score, confidence, min_conf), "price": price, "atr": a,
        "side": "LONG (buy)" if long_side else "EXIT / SHORT (sell)",
        "entry_zone": entry, "stop_loss": stop, "target1": t1, "target2": t2, "target3": t3,
        "risk_per_share": risk, "qty": qty, "capital_needed": qty * price,
        "max_loss": qty * risk, "nearest_support": sup, "nearest_resistance": res,
        "supports": supports[:4], "resistances": resistances[:4],
        "pivots": pivot_points(d), "reasons": reasons,
    }


# ----------------------------------------------------------------------------
# Comparison with benchmark (e.g. NIFTY 50)
# ----------------------------------------------------------------------------
def _max_drawdown(prices: pd.Series) -> float:
    return float((prices / prices.cummax() - 1).min())


def period_returns(close: pd.Series) -> dict:
    close = close.dropna()
    end = close.index[-1]
    out = {}
    for label, off in [("1W", pd.DateOffset(weeks=1)), ("1M", pd.DateOffset(months=1)),
                       ("3M", pd.DateOffset(months=3)), ("6M", pd.DateOffset(months=6)),
                       ("1Y", pd.DateOffset(years=1)), ("3Y", pd.DateOffset(years=3))]:
        past = close[close.index <= end - off]
        out[label] = float(close.iloc[-1] / past.iloc[-1] - 1) if len(past) else np.nan
    ytd = close[close.index < pd.Timestamp(year=end.year, month=1, day=1, tz=end.tz)]
    out["YTD"] = float(close.iloc[-1] / ytd.iloc[-1] - 1) if len(ytd) else np.nan
    return out


def compare(stock: pd.Series, bench: pd.Series, rf: float = 0.065) -> dict:
    """Risk/return statistics of `stock` vs `bench` (daily close series)."""
    df = pd.concat({"s": stock, "b": bench}, axis=1).dropna()
    r = df.pct_change().dropna()
    var_b = r["b"].var()
    beta = float(r["s"].cov(r["b"]) / var_b) if var_b else np.nan
    corr = float(r["s"].corr(r["b"]))
    years = max(len(r) / TRADING_DAYS, 1e-9)
    cagr_s = float((df["s"].iloc[-1] / df["s"].iloc[0]) ** (1 / years) - 1)
    cagr_b = float((df["b"].iloc[-1] / df["b"].iloc[0]) ** (1 / years) - 1)
    vol_s = float(r["s"].std() * np.sqrt(TRADING_DAYS))
    vol_b = float(r["b"].std() * np.sqrt(TRADING_DAYS))
    alpha = cagr_s - (rf + beta * (cagr_b - rf))
    up, dn = r["b"] > 0, r["b"] < 0
    up_cap = float(r["s"][up].mean() / r["b"][up].mean()) if up.any() else np.nan
    dn_cap = float(r["s"][dn].mean() / r["b"][dn].mean()) if dn.any() else np.nan
    active = r["s"] - r["b"]
    te = float(active.std() * np.sqrt(TRADING_DAYS))
    return {
        "beta": beta, "correlation": corr, "alpha": alpha,
        "cagr_stock": cagr_s, "cagr_bench": cagr_b,
        "vol_stock": vol_s, "vol_bench": vol_b,
        "sharpe_stock": (cagr_s - rf) / vol_s if vol_s else np.nan,
        "sharpe_bench": (cagr_b - rf) / vol_b if vol_b else np.nan,
        "maxdd_stock": _max_drawdown(df["s"]), "maxdd_bench": _max_drawdown(df["b"]),
        "up_capture": up_cap, "down_capture": dn_cap,
        "info_ratio": float(active.mean() * TRADING_DAYS / te) if te else np.nan,
        "relative_strength": df["s"] / df["b"] / (df["s"].iloc[0] / df["b"].iloc[0]) * 100,
        "rolling_beta": r["s"].rolling(60).cov(r["b"]) / r["b"].rolling(60).var(),
    }


# ----------------------------------------------------------------------------
# Backtest of the signal engine
# ----------------------------------------------------------------------------
def backtest(d: pd.DataFrame, buy_th: float = 15, sell_th: float = -15, cost_pct: float = 0.1) -> dict:
    """Long-only: go long when score >= buy_th, go flat when score <= sell_th.
    The signal is computed at a bar's close and the trade is filled at that same close;
    returns are earned from the following bar on. cost_pct is charged per side."""
    score = score_series(d)
    pos = pd.Series(np.nan, index=d.index)
    pos[score >= buy_th] = 1.0
    pos[score <= sell_th] = 0.0
    pos = pos.ffill().fillna(0.0).shift(1).fillna(0.0)
    ret = d["Close"].pct_change().fillna(0.0)
    trades = pos.diff().abs().fillna(0.0)
    strat = pos * ret - trades * cost_pct / 100
    equity = (1 + strat).cumprod()
    bh = (1 + ret).cumprod()

    # individual round trips
    entries = d.index[(pos.diff() == 1)]
    exits = d.index[(pos.diff() == -1)]
    trips = []
    for e in entries:
        x = exits[exits > e]
        xe = x[0] if len(x) else d.index[-1]
        pe, px = d["Close"].shift(1).loc[e], d["Close"].shift(1).loc[xe] if len(x) else d["Close"].iloc[-1]
        trips.append({"entry_date": e, "entry": pe, "exit_date": xe, "exit": px,
                      "return": px / pe - 1 - 2 * cost_pct / 100, "open": not len(x)})
    trips_df = pd.DataFrame(trips)
    years = max(len(d) / TRADING_DAYS, 1e-9)
    return {
        "equity": equity, "buy_hold": bh, "position": pos, "score": score, "trades": trips_df,
        "total_return": float(equity.iloc[-1] - 1), "bh_return": float(bh.iloc[-1] - 1),
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1),
        "max_dd": _max_drawdown(equity), "bh_max_dd": _max_drawdown(bh),
        "n_trades": len(trips_df),
        "win_rate": float((trips_df["return"] > 0).mean()) if len(trips_df) else np.nan,
        "exposure": float(pos.mean()),
    }


# ----------------------------------------------------------------------------
# Breakout / breakdown detection
# ----------------------------------------------------------------------------
def _fresh_cross(close: pd.Series, level: float, up: bool, fresh_days: int) -> int | None:
    """Bars ago (0 = today) when close crossed `level` within the last `fresh_days`
    bars and is still beyond it; None if no such fresh cross."""
    c = close.to_numpy()
    if (up and c[-1] <= level) or (not up and c[-1] >= level):
        return None
    for ago in range(0, min(fresh_days, len(c) - 1)):
        i = len(c) - 1 - ago
        if (up and c[i - 1] <= level < c[i]) or (not up and c[i - 1] >= level > c[i]):
            return ago
    return None


def detect_breakouts(df: pd.DataFrame, fresh_days: int = 3, vol_mult: float = 1.5,
                     d: pd.DataFrame | None = None) -> list[dict]:
    """Find fresh breakouts (BUY) and breakdowns (SELL) on daily bars.

    Checks: swing resistance/support levels, 52-week high/low, 20-day range (Donchian),
    Bollinger squeeze release and big gaps. Levels come only from bars BEFORE the
    breakout window, so today's move can't create its own level."""
    if len(df) < 60:
        return []
    d = add_indicators(df) if d is None else d
    close, last = d["Close"], d.iloc[-1]
    a = float(last["ATR"]) if pd.notna(last["ATR"]) else float(last["Close"]) * 0.02
    vol_x = float(last["Volume"] / last["VolSMA20"]) if last["VolSMA20"] else np.nan
    # strongest volume in the breakout window (the break day may not be today)
    win_vol_x = float((d["Volume"] / d["VolSMA20"]).iloc[-fresh_days:].max())
    hist = d.iloc[:-fresh_days]
    found = []

    def add(kind, side, level, ago, base_strength, needs_volume=True):
        # Most breakouts on normal volume fail, so require volume unless it's a 52-week level
        if needs_volume and not (win_vol_x >= vol_mult):
            return
        level = float(level)
        px = float(last["Close"])
        beyond = (px / level - 1) * 100
        # stop just beyond the broken level, but never more than 3 ATR away
        if side == "BUY":
            stop = max(min(level - 0.5 * a, px - a), px - 3 * a)
            target = px + 2 * (px - stop)
        else:
            stop = min(max(level + 0.5 * a, px + a), px + 3 * a)
            target = px - 2 * (stop - px)
        strength = base_strength + (2 if win_vol_x >= 2 else 1 if win_vol_x >= vol_mult else 0)
        found.append({"type": kind, "side": side, "level": level, "beyond_pct": beyond,
                      "days_ago": ago, "vol_x": win_vol_x, "stop": stop, "target": target,
                      "strength": strength})

    # 1. Swing resistance / support: a real barrier = touched 2+ times AND price stayed
    #    on one side of it for the 15 bars before the breakout window; break must be clear (0.3 ATR)
    sup, res = swing_levels(hist, lookback=250)
    before = hist["Close"].tail(15)
    margin = 0.3 * a
    best_up = best_dn = None
    for lv, touches in sup + res:
        if touches < 2:
            continue
        if before.max() < lv:
            ago = _fresh_cross(close, lv + margin, True, fresh_days)
            if ago is not None and (best_up is None or (touches, lv) > (best_up[1], best_up[0])):
                best_up = (lv, touches, ago)
        if before.min() > lv:
            ago = _fresh_cross(close, lv - margin, False, fresh_days)
            if ago is not None and (best_dn is None or (touches, -lv) > (best_dn[1], -best_dn[0])):
                best_dn = (lv, touches, ago)
    if best_up:
        add(f"Resistance breakout ({best_up[1]} touches)", "BUY", best_up[0], best_up[2], 1 + min(best_up[1], 3))
    if best_dn:
        add(f"Support breakdown ({best_dn[1]} touches)", "SELL", best_dn[0], best_dn[2], 1 + min(best_dn[1], 3))

    # 2. 52-week high / low
    yr = hist.tail(250)
    if len(yr) >= 120:
        ago = _fresh_cross(close, yr["High"].max(), True, fresh_days)
        if ago is not None:
            add("52-week high breakout", "BUY", yr["High"].max(), ago, 4, needs_volume=False)
        ago = _fresh_cross(close, yr["Low"].min(), False, fresh_days)
        if ago is not None:
            add("52-week low breakdown", "SELL", yr["Low"].min(), ago, 4, needs_volume=False)

    # 3. 20-day range (Donchian channel)
    rng = hist.tail(20)
    ago = _fresh_cross(close, rng["High"].max(), True, fresh_days)
    if ago is not None:
        add("20-day range breakout", "BUY", rng["High"].max(), ago, 1)
    ago = _fresh_cross(close, rng["Low"].min(), False, fresh_days)
    if ago is not None:
        add("20-day range breakdown", "SELL", rng["Low"].min(), ago, 1)

    # 4. Bollinger squeeze release: bands were the tightest in 6 months, price now outside
    bw = (d["BB_up"] - d["BB_low"]) / d["BB_mid"]
    if bw.iloc[-fresh_days - 5:-fresh_days].min() <= bw.iloc[-130:].quantile(0.1):
        if last["Close"] > last["BB_up"]:
            add("Bollinger squeeze breakout", "BUY", last["BB_up"], 0, 2)
        elif last["Close"] < last["BB_low"]:
            add("Bollinger squeeze breakdown", "SELL", last["BB_low"], 0, 2)

    # 5. Big gap with volume
    gap = (last["Open"] / d["Close"].iloc[-2] - 1) * 100
    if abs(gap) >= 3 and vol_x >= vol_mult:
        up = gap > 0 and last["Close"] >= last["Open"]
        dn = gap < 0 and last["Close"] <= last["Open"]
        if up or dn:
            add(f"Gap {'up' if up else 'down'} {gap:+.1f}% on volume", "BUY" if up else "SELL",
                d["Close"].iloc[-2], 0, 2)
    return found


def summarize_breakouts(found: list[dict]) -> dict | None:
    """Combine all breakouts of one stock into one row (majority side wins)."""
    if not found:
        return None
    buy = [f for f in found if f["side"] == "BUY"]
    sell = [f for f in found if f["side"] == "SELL"]
    side_list = buy if sum(f["strength"] for f in buy) >= sum(f["strength"] for f in sell) else sell
    main = max(side_list, key=lambda f: f["strength"])
    strength = sum(f["strength"] for f in side_list)
    major = (main["vol_x"] >= 2 and strength >= 5) or ("52-week" in " ".join(f["type"] for f in side_list)
                                                     and main["vol_x"] >= 1.5)
    return {
        "side": main["side"], "major": major, "strength": strength,
        "types": " + ".join(f["type"] for f in side_list), "level": main["level"],
        "beyond_pct": main["beyond_pct"], "days_ago": min(f["days_ago"] for f in side_list),
        "vol_x": main["vol_x"], "stop": main["stop"], "target": main["target"],
    }


# ----------------------------------------------------------------------------
# Key levels ladder
# ----------------------------------------------------------------------------
def key_levels(d: pd.DataFrame, plan: dict, quote: dict | None = None, session_open: bool = True) -> pd.DataFrame:
    """All important price levels for the stock, sorted high -> low.
    kind: price | resistance | support | target | stop | entry | info"""
    last, a = d.iloc[-1], plan["atr"]
    price = float(quote["last"]) if quote else float(last["Close"])
    rows = []

    def add(name, value, kind, note=""):
        if value is None or not np.isfinite(value):
            return
        rows.append({"Level": name, "Price": float(value), "kind": kind, "note": note})

    add("CURRENT PRICE", price, "price")
    # Today's session (live quote preferred)
    if quote:
        add("Day high", quote["day_high"], "resistance", "today")
        add("Day low", quote["day_low"], "support", "today")
        add("Open", quote["open"], "info", "today")
        add("Previous close", quote["prev_close"], "info")
        if quote.get("vwap"):
            add("VWAP", quote["vwap"], "info", "today's average price")
    else:
        add("Day high", last["High"], "resistance", "latest session")
        add("Day low", last["Low"], "support", "latest session")
        add("Open", last["Open"], "info")
    prev = d.iloc[-2] if session_open and len(d) > 1 else d.iloc[-1]
    if session_open and len(d) > 1:
        add("Prev day high", prev["High"], "resistance")
        add("Prev day low", prev["Low"], "support")
    # Pivots / CPR
    pv = pivot_points(d, use_last=not session_open)
    for k in ("R2", "R1", "S1", "S2"):
        add(f"Pivot {k}", pv[k], "resistance" if k.startswith("R") else "support",
            "" if session_open else "next session")
    add("Pivot (P)", pv["Pivot"], "info", "CPR " + f"{pv['CPR bottom']:,.1f}–{pv['CPR top']:,.1f}")
    # Swing zones
    for lv, t in plan["resistances"][:3]:
        add(f"Resistance zone ({t}× tested)", lv, "resistance", f"{lv - 0.25 * a:,.1f} – {lv + 0.25 * a:,.1f}")
    for lv, t in plan["supports"][:3]:
        add(f"Support zone ({t}× tested)", lv, "support", f"{lv - 0.25 * a:,.1f} – {lv + 0.25 * a:,.1f}")
    # Moving averages & 52 week
    for col in ("SMA50", "SMA200"):
        if pd.notna(last.get(col)):
            v = float(last[col])
            add(col.replace("SMA", "") + "-day average", v, "resistance" if v > price else "support", "dynamic level")
    yr = d.tail(252)
    hi52 = quote["high_52w"] if quote else yr["High"].max()
    lo52 = quote["low_52w"] if quote else yr["Low"].min()
    add("52-week high", hi52, "resistance")
    add("52-week low", lo52, "support")
    # Trade plan
    tag = "" if plan["verdict"] != "HOLD / WAIT" else " (reference only — no strong signal)"
    add("Stop-loss", plan["stop_loss"], "stop", "exit if price closes beyond" + tag)
    for i in (1, 2, 3):
        add(f"Target {i}", plan[f"target{i}"], "target", f"{i}× risk" + tag)
    df = pd.DataFrame(rows)
    # A broken resistance becomes support and vice versa: classify by side of current price
    sr = df["kind"].isin(["resistance", "support"])
    df.loc[sr, "kind"] = np.where(df.loc[sr, "Price"] > price, "resistance", "support")
    # Ignore far-away clutter (more than ~12 ATR away), keep plan + 52-week + price
    keep = df["kind"].isin(["price", "stop", "target"]) | df["Level"].str.startswith("52-week") | \
        ((df["Price"] - price).abs() <= 12 * a)
    df = df[keep].copy()
    df["Distance %"] = (df["Price"] / price - 1) * 100
    return df.sort_values("Price", ascending=False).reset_index(drop=True)


def key_stats(d: pd.DataFrame, plan: dict, quote: dict | None = None) -> dict:
    last = d.iloc[-1]
    price = float(quote["last"]) if quote else float(last["Close"])
    rr = abs(plan["target2"] - price) / plan["risk_per_share"] if plan["risk_per_share"] else np.nan
    hi52 = quote["high_52w"] if quote else d.tail(252)["High"].max()
    lo52 = quote["low_52w"] if quote else d.tail(252)["Low"].min()
    return {
        "Expected daily move (ATR)": f"±₹{plan['atr']:,.2f} ({plan['atr'] / price * 100:.1f}%)",
        "Volume vs 20-day avg": f"{last['Volume'] / last['VolSMA20']:.1f}×" if last["VolSMA20"] else "—",
        "RSI (14)": f"{last['RSI']:.0f}" + (" overbought" if last["RSI"] > 70 else " oversold" if last["RSI"] < 30 else ""),
        "Trend": ("Up" if last["Close"] > last["SMA200"] else "Down") + "trend (vs 200-day avg)" if pd.notna(last["SMA200"]) else "—",
        "52-week position": f"{(price - lo52) / (hi52 - lo52) * 100:.0f}% of range" if hi52 > lo52 else "—",
        "Risk : reward (to T2)": f"1 : {rr:.1f}" if np.isfinite(rr) else "—",
        "Stop distance": f"{plan['risk_per_share'] / price * 100:.1f}%",
    }


# ----------------------------------------------------------------------------
# How long to hold: learned from this stock's own past signals
# ----------------------------------------------------------------------------
def _bdays_from_today(n: float) -> pd.Timestamp:
    return (pd.Timestamp.now(tz="Asia/Kolkata").normalize() + pd.offsets.BDay(int(np.ceil(n))))


def holding_estimate(d: pd.DataFrame, plan: dict, min_conf: float = 60, horizon: int = 60) -> dict:
    """For past signals in the same direction as the current plan, measure how many
    trading days it took to reach 1R/2R/3R targets (stop = 2 ATR), and how long the
    signal itself lasted. Falls back to a volatility estimate if history is thin."""
    side = 1 if plan["score"] >= 0 else -1
    score = score_series(d)
    cs = confidence_series(d)
    conf = cs["bull"] if side > 0 else cs["bear"]
    active = ((score * side) >= 15) & (conf >= min_conf)
    starts = np.where(active & ~active.shift(1, fill_value=False))[0]
    c, h, l = d["Close"].to_numpy(), d["High"].to_numpy(), d["Low"].to_numpy()
    a = d["ATR"].to_numpy()
    act = active.to_numpy()
    reach = {1: [], 2: [], 3: []}
    stopped, decided, durations = 0, 0, []
    for i in starts:
        if np.isnan(a[i]) or i >= len(d) - 1:
            continue
        entry, risk = c[i], 2 * a[i]
        stop = entry - side * risk
        got = {}
        end = min(i + 1 + horizon, len(d))
        for j in range(i + 1, end):
            if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
                stopped += 1
                break
            for k in (1, 2, 3):
                tgt = entry + side * k * risk
                if k not in got and ((side > 0 and h[j] >= tgt) or (side < 0 and l[j] <= tgt)):
                    got[k] = j - i
            if 3 in got:
                break
        if got or j == end - 1 or stopped:
            decided += 1
        for k, v in got.items():
            reach[k].append(v)
        # signal duration
        j = i
        while j + 1 < len(d) and act[j + 1]:
            j += 1
        if j + 1 < len(d):
            durations.append(j - i + 1)
    n = len([i for i in starts if not np.isnan(a[i]) and i < len(d) - 1])
    out = {"side": side, "n_signals": n, "method": "history" if n >= 5 else "volatility"}
    price, atr_now, risk_now = plan["price"], plan["atr"], plan["risk_per_share"]
    for k in (1, 2, 3):
        arr = np.array(reach[k])
        if n >= 5 and len(arr) >= 2:
            out[f"t{k}_days"] = float(np.median(arr))
            out[f"t{k}_range"] = (float(np.percentile(arr, 25)), float(np.percentile(arr, 75)))
            out[f"t{k}_rate"] = len(arr) / max(n, 1) * 100
        else:  # random-walk estimate: time ~ (distance / daily move)^2
            est = min(((k * risk_now) / (0.8 * atr_now)) ** 2, 250) if atr_now else np.nan
            out[f"t{k}_days"] = float(est)
            out[f"t{k}_range"] = (float(est * 0.5), float(est * 1.5))
            out[f"t{k}_rate"] = np.nan
        out[f"t{k}_date"] = _bdays_from_today(out[f"t{k}_days"])
    out["stop_rate"] = stopped / max(n, 1) * 100 if n else np.nan
    out["signal_days"] = float(np.median(durations)) if durations else np.nan
    t1_hi = out["t1_range"][1]
    out["time_stop_days"] = int(max(3, np.ceil(t1_hi * 1.25))) if np.isfinite(t1_hi) else 10
    out["hold_min"], out["hold_max"] = out["t1_days"], out["t2_days"]
    return out


def position_advice(d: pd.DataFrame, plan: dict, entry: float, entry_date, live_price: float | None = None,
                    stop: float | None = None, time_stop_days: int = 10, min_conf: float = 60) -> dict:
    """Live HOLD / BOOK PROFIT / SELL decision for a long position bought at `entry`."""
    price = float(live_price if live_price else d["Close"].iloc[-1])
    a = float(d["ATR"].iloc[-1])
    entry_ts = pd.Timestamp(entry_date)
    if entry_ts.tz is None:
        entry_ts = entry_ts.tz_localize("Asia/Kolkata")
    since = d[d.index.normalize() >= entry_ts.normalize()]
    if len(since) and pd.notna(since["ATR"].iloc[0]):
        a = float(since["ATR"].iloc[0])  # volatility at the time you bought
    high_since = max(float(since["High"].max()) if len(since) else price, price)
    days_held = int(np.busday_count(entry_ts.date(), pd.Timestamp.now(tz="Asia/Kolkata").date()))
    init_stop = float(stop) if stop else entry - 2 * a
    risk = max(entry - init_stop, 1e-9)
    t1, t2, t3 = entry + risk, entry + 2 * risk, entry + 3 * risk
    trail = init_stop
    stage = "Before Target 1"
    if high_since >= t1:
        trail, stage = max(trail, entry), "Target 1 reached"
    if high_since >= t2:
        trail, stage = max(trail, t1), "Target 2 reached"
    last = d.iloc[-1]
    if last.get("ST_dir", 0) == 1 and pd.notna(last.get("ST")) and high_since >= t1:
        trail = max(trail, float(last["ST"]))
    pnl_pct = (price / entry - 1) * 100
    conf_bull = float(confidence_series(d)["bull"].iloc[-1])
    v = plan["verdict"]
    if price <= trail:
        action, color, why = "SELL NOW", "#991b1b", (
            "Price is below your stop-loss." if trail == init_stop else "Price fell back to the trailing stop. Lock in what is left.")
    elif price >= t3:
        action, color, why = "BOOK FULL PROFIT", "#047857", "Final target (3× risk) reached."
    elif "SELL" in v:
        action, color, why = "SELL", "#dc2626", f"Trend has turned down ({v}, {plan['confidence']:.0f}% confidence)."
    elif price >= t2:
        action, color, why = "BOOK MORE PROFIT", "#047857", "Target 2 reached: sell another third, keep stop at Target 1."
    elif price >= t1:
        action, color, why = "BOOK PARTIAL PROFIT", "#16a34a", "Target 1 reached: sell a third and move your stop to your buy price."
    elif days_held > time_stop_days and high_since < t1:
        action, color, why = "SELL (TIME STOP)", "#b45309", (
            f"Held {days_held} trading days without reaching Target 1; this stock usually gets there faster. Free the money.")
    elif conf_bull < 40:
        action, color, why = "HOLD – CAUTION", "#ca8a04", f"Uptrend support is weak ({conf_bull:.0f}% bullish). Keep the stop tight."
    else:
        action, color, why = "HOLD", "#16a34a", f"Trend still supports the trade ({conf_bull:.0f}% bullish)."
    return {"price": price, "pnl_pct": pnl_pct, "days_held": days_held, "high_since": high_since,
            "stop": trail, "init_stop": init_stop, "t1": t1, "t2": t2, "t3": t3, "stage": stage,
            "action": action, "color": color, "why": why, "time_stop_days": time_stop_days,
            "days_left": max(time_stop_days - days_held, 0)}
