"""Scalping signals on 1-minute (or 3/5-minute) candles.

Long setup points (100 max, short side mirrors it):
  price above VWAP 20 · EMA9 above EMA21 15 · EMA9 rising 10 · fresh EMA cross 10
  RSI(7) momentum 15 · volume burst 10 · micro-breakout of 15-bar high 10 · strong close 10
Signal = side with >= min_conf points (and more than the other side).
Stop = beyond the last 5 bars' extreme (0.5–2 ATR); targets 1R / 1.5R / 2R.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import atr as _atr, rsi as _rsi, vwap as _vwap

FACTORS = {
    "Price vs VWAP": 20, "EMA 9 vs 21": 15, "EMA 9 direction": 10, "Fresh EMA cross": 10,
    "RSI(7) momentum": 15, "Volume burst": 10, "15-bar breakout": 10, "Strong candle close": 10,
}


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes <= 1:
        return df
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df.resample(f"{minutes}min", origin="start_day", offset="15min").agg(agg)
    return out.dropna(subset=["Close"])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    d["EMA9"] = c.ewm(span=9, adjust=False).mean()
    d["EMA21"] = c.ewm(span=21, adjust=False).mean()
    d["VWAP"] = _vwap(d)
    d["RSI7"] = _rsi(c, 7)
    d["ATR"] = _atr(d)
    d["VolAvg"] = d["Volume"].rolling(20, min_periods=5).mean()
    d["Hi15"] = d["High"].shift(1).rolling(15).max()
    d["Lo15"] = d["Low"].shift(1).rolling(15).min()
    return d


def factor_points(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-bar points for the long side and the short side."""
    c, o, h, l = d["Close"], d["Open"], d["High"], d["Low"]
    rng = (h - l).replace(0, np.nan)
    up_cross = (d["EMA9"] > d["EMA21"]) & (d["EMA9"].shift() <= d["EMA21"].shift())
    dn_cross = (d["EMA9"] < d["EMA21"]) & (d["EMA9"].shift() >= d["EMA21"].shift())
    vol_burst = d["Volume"] >= 1.5 * d["VolAvg"]
    r = d["RSI7"]
    L, S = pd.DataFrame(index=d.index), pd.DataFrame(index=d.index)
    L["Price vs VWAP"], S["Price vs VWAP"] = (c > d["VWAP"]) * 20, (c < d["VWAP"]) * 20
    L["EMA 9 vs 21"], S["EMA 9 vs 21"] = (d["EMA9"] > d["EMA21"]) * 15, (d["EMA9"] < d["EMA21"]) * 15
    L["EMA 9 direction"] = (d["EMA9"] > d["EMA9"].shift(3)) * 10
    S["EMA 9 direction"] = (d["EMA9"] < d["EMA9"].shift(3)) * 10
    L["Fresh EMA cross"] = up_cross.rolling(5, min_periods=1).max().fillna(0) * 10
    S["Fresh EMA cross"] = dn_cross.rolling(5, min_periods=1).max().fillna(0) * 10
    L["RSI(7) momentum"] = np.where((r > 55) & (r < 85), 15, np.where((r < 25) & (r > r.shift()), 8, 0))
    S["RSI(7) momentum"] = np.where((r < 45) & (r > 15), 15, np.where((r > 75) & (r < r.shift()), 8, 0))
    L["Volume burst"], S["Volume burst"] = (vol_burst & (c > o)) * 10, (vol_burst & (c < o)) * 10
    L["15-bar breakout"], S["15-bar breakout"] = (c > d["Hi15"]) * 10, (c < d["Lo15"]) * 10
    pos = (c - l) / rng
    L["Strong candle close"] = ((c > o) & (pos >= 0.7)) * 10
    S["Strong candle close"] = ((c < o) & (pos <= 0.3)) * 10
    return L.fillna(0).astype(float), S.fillna(0).astype(float)


def _in_session(ts: pd.Timestamp) -> bool:
    t = ts.hour * 60 + ts.minute
    return 9 * 60 + 20 <= t <= 15 * 60 + 15  # skip first 5 min and last 15 min


def _levels(d: pd.DataFrame, i: int, side: int):
    price = float(d["Close"].iloc[i])
    a = float(d["ATR"].iloc[i]) if pd.notna(d["ATR"].iloc[i]) else price * 0.002
    if side > 0:
        swing = float(d["Low"].iloc[max(0, i - 4):i + 1].min()) - 0.1 * a
        stop = min(max(swing, price - 2 * a), price - 0.5 * a)
    else:
        swing = float(d["High"].iloc[max(0, i - 4):i + 1].max()) + 0.1 * a
        stop = max(min(swing, price + 2 * a), price + 0.5 * a)
    risk = abs(price - stop)
    return price, stop, [price + side * m * risk for m in (1, 1.5, 2)], a


def scalp_analysis(df: pd.DataFrame, min_conf: float = 60, cooldown: int = 5, horizon: int = 30) -> dict:
    """Current scalp call + every past signal with its outcome (did T1 come before the stop?)."""
    d = prepare(df)
    L, S = factor_points(d)
    lp, sp = L.sum(axis=1), S.sum(axis=1)
    side = np.where((lp >= min_conf) & (lp > sp), 1, np.where((sp >= min_conf) & (sp > lp), -1, 0))
    side = pd.Series(side, index=d.index)
    sess = pd.Series([_in_session(t) for t in d.index], index=d.index)
    side = side.where(sess, 0)

    # signal events: side turns on (with cooldown)
    events, last_i = [], -10 ** 9
    hi, lo = d["High"].to_numpy(), d["Low"].to_numpy()
    days = np.array([t.date() for t in d.index])
    for i in range(1, len(d)):
        s_ = side.iloc[i]
        if s_ == 0 or (side.iloc[i - 1] == s_) or i - last_i < cooldown:
            continue
        last_i = i
        price, stop, tgts, _ = _levels(d, i, s_)
        outcome, r_mult, bars_held = "Open", 0.0, np.nan
        for j in range(i + 1, min(i + 1 + horizon, len(d))):
            if days[j] != days[i]:
                outcome = "Closed (day end)"
                r_mult = (float(d["Close"].iloc[j - 1]) - price) * s_ / abs(price - stop)
                break
            hit_stop = lo[j] <= stop if s_ > 0 else hi[j] >= stop
            hit_t1 = hi[j] >= tgts[0] if s_ > 0 else lo[j] <= tgts[0]
            if hit_stop:  # stop first if both in same bar (conservative)
                outcome, r_mult, bars_held = "Stop hit ❌", -1.0, j - i
                break
            if hit_t1:
                outcome, r_mult, bars_held = "Target 1 hit ✅", 1.0, j - i
                break
        else:
            if i + horizon < len(d):
                outcome = "No result in 30 bars"
        events.append({"time": d.index[i], "side": "BUY" if s_ > 0 else "SELL",
                       "confidence": float(lp.iloc[i] if s_ > 0 else sp.iloc[i]),
                       "entry": price, "stop": stop, "t1": tgts[0], "outcome": outcome, "r": r_mult,
                       "bars": bars_held})
    ev = pd.DataFrame(events)

    # current state
    i = len(d) - 1
    cur_side = int(side.iloc[i])
    lead = 1 if lp.iloc[i] >= sp.iloc[i] else -1
    price, stop, tgts, a = _levels(d, i, cur_side or lead)
    conf = float(lp.iloc[i] if (cur_side or lead) > 0 else sp.iloc[i])
    reasons = []
    for k, w in FACTORS.items():
        lv, sv = float(L[k].iloc[i]), float(S[k].iloc[i])
        reasons.append({"Factor": k, "Max": w, "Long pts": lv, "Short pts": sv,
                        "Favours": "🟢 Buy" if lv > sv else "🔴 Sell" if sv > lv else "⚪ —"})
    if cur_side > 0:
        call = "SCALP BUY"
    elif cur_side < 0:
        call = "SCALP SELL"
    else:
        call = "NO SCALP"
    reason_wait = ""
    if cur_side == 0:
        if not sess.iloc[i]:
            reason_wait = "Outside scalping hours (9:20–15:15)."
        else:
            reason_wait = f"Best side has only {conf:.0f}% (needs {min_conf:.0f}%)."

    today = ev[ev["time"].dt.date == d.index[-1].date()] if len(ev) else ev
    def stats(e):
        done = e[e["outcome"].isin(["Target 1 hit ✅", "Stop hit ❌"])] if len(e) else e
        wins = e[e["outcome"] == "Target 1 hit ✅"] if len(e) else e
        return {"signals": len(e), "decided": len(done),
                "win_bars": float(wins["bars"].median()) if len(wins) else np.nan,
                "win_bars_p75": float(wins["bars"].quantile(0.75)) if len(wins) else np.nan,
                "win_rate": float((done["outcome"] == "Target 1 hit ✅").mean() * 100) if len(done) else np.nan,
                "avg_r": float(e["r"].mean()) if len(e) else np.nan}
    return {
        "call": call, "side": cur_side, "lead": lead, "confidence": conf, "min_conf": min_conf,
        "price": price, "stop": stop, "targets": tgts, "atr": a, "reasons": reasons, "wait_reason": reason_wait,
        "events": ev, "stats_today": stats(today), "stats_all": stats(ev), "data": d,
        "long_pts": lp, "short_pts": sp,
    }
