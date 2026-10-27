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


# Smart filters learned from the 23 Oct 2026 live report (786 calls):
#   against market direction 41% right vs 52% with it · right after a stop-loss 42% vs 55%
#   11:00-13:30 chop 38-44% · calls 11-20 in the same stock 40% · stops < 0.15% often hit by noise
DEFAULT_FILTERS = {
    "market_filter": True,       # no BUY while NIFTY falls (30 min), no SELL while it rises
    "mkt_threshold": 0.05,       # % NIFTY move in 30 min that counts as rising / falling
    "skip_midday": True,         # no new scalp 11:00-13:30
    "cooldown_after_loss": 30,   # minutes to wait in a stock after its stop-loss is hit
    "no_flip_minutes": 15,       # no opposite call within this many minutes of the last one
    "max_per_stock": 8,          # max scalp calls per stock per day
    "min_stop_pct": 0.15,        # stop-loss at least this % away from entry
    "max_stretch": 1.5,          # don't chase if the stock already moved this % our way today
    "one_at_a_time": True,       # no new call while the previous one in the stock is still open
    # High-Accuracy Mode: only A+ setups where every condition lines up
    "a_plus": False,
    "a_plus_min_conf": 80,       # signal confidence needed
    "a_plus_max_per_stock": 3,   # calls per stock per day
    "a_plus_min_stop_pct": 0.25, # wider stop, out of the 1-minute noise
    "target_ratio": 1.0,         # Target 1 = this many times the stop distance (1, 0.75 or 0.5)
}
NO_FILTERS = {**DEFAULT_FILTERS, "a_plus": False, "target_ratio": 1.0, "market_filter": False, "skip_midday": False, "cooldown_after_loss": 0,
              "no_flip_minutes": 0, "max_per_stock": 10 ** 6, "min_stop_pct": 0.0, "max_stretch": 1e9,
              "one_at_a_time": False}


def _levels(d: pd.DataFrame, i: int, side: int, min_stop_pct: float = 0.0, target_ratio: float = 1.0):
    price = float(d["Close"].iloc[i])
    a = float(d["ATR"].iloc[i]) if pd.notna(d["ATR"].iloc[i]) else price * 0.002
    if side > 0:
        swing = float(d["Low"].iloc[max(0, i - 4):i + 1].min()) - 0.1 * a
        stop = min(max(swing, price - 2 * a), price - 0.5 * a)
    else:
        swing = float(d["High"].iloc[max(0, i - 4):i + 1].max()) + 0.1 * a
        stop = max(min(swing, price + 2 * a), price + 0.5 * a)
    risk = max(abs(price - stop), price * min_stop_pct / 100)
    stop = price - side * risk
    return price, stop, [price + side * m * risk * target_ratio for m in (1, 1.5, 2)], a


def add_context(d: pd.DataFrame, mkt: pd.Series | None = None) -> pd.DataFrame:
    """Buyer/seller pressure, VWAP side, today's move and market direction at every bar."""
    day = pd.Series([t.date() for t in d.index], index=d.index)
    day_open = d["Open"].groupby(day).transform("first")
    d["day_move"] = (d["Close"] / day_open - 1) * 100
    up_vol = d["Volume"].where(d["Close"] > d["Open"], 0.0)
    dn_vol = d["Volume"].where(d["Close"] < d["Open"], 0.0)
    tot = (up_vol + dn_vol).rolling(15, min_periods=5).sum().replace(0, np.nan)
    d["buy_pressure"] = (up_vol.rolling(15, min_periods=5).sum() / tot * 100).fillna(50)
    d["vwap_side"] = np.sign(d["Close"] - d["VWAP"])
    if mkt is not None and len(mkt) > 5:
        m = mkt.sort_index()
        m = m[~m.index.duplicated(keep="last")]
        now_v = m.reindex(d.index, method="ffill")
        past_v = m.reindex(d.index - pd.Timedelta(minutes=30), method="ffill")
        d["mkt30"] = (now_v.to_numpy() / past_v.to_numpy() - 1) * 100
    else:
        d["mkt30"] = np.nan
    return d


def _stop_pct(f):
    return max(f["min_stop_pct"], f["a_plus_min_stop_pct"]) if f.get("a_plus") else f["min_stop_pct"]


def _block_reasons(t, sgn, row, state, f, conf=None):
    r = []
    if f.get("a_plus"):
        hm_ = t.hour * 60 + t.minute
        m30_ = row["mkt30"]
        if conf is not None and conf < f["a_plus_min_conf"]:
            r.append(f"A+: confidence {conf:.0f}% < {f['a_plus_min_conf']:.0f}%")
        if pd.isna(m30_) or m30_ * sgn <= f["mkt_threshold"]:
            r.append("A+: market not clearly moving our way")
        if pd.isna(row["vwap_side"]) or row["vwap_side"] * sgn <= 0:
            r.append("A+: price on the wrong side of VWAP")
        bp_ = row["buy_pressure"]
        if (sgn > 0 and bp_ < 55) or (sgn < 0 and bp_ > 45):
            r.append("A+: volume not with us")
        dm_ = row["day_move"]
        if pd.notna(dm_) and not (-0.5 <= dm_ * sgn <= 1.0):
            r.append("A+: not an early move")
        if not (9 * 60 + 45 <= hm_ < 11 * 60 or 13 * 60 + 30 <= hm_ < 14 * 60 + 45):
            r.append("A+: outside 9:45–11:00 / 1:30–2:45")
        if state["count"] >= f["a_plus_max_per_stock"]:
            r.append(f"A+: already {state['count']} calls in this stock")
    m30 = row["mkt30"]
    if f["market_filter"] and pd.notna(m30):
        if sgn > 0 and m30 < -f["mkt_threshold"]:
            r.append(f"Market falling (NIFTY {m30:+.2f}% in 30 min)")
        if sgn < 0 and m30 > f["mkt_threshold"]:
            r.append(f"Market rising (NIFTY {m30:+.2f}% in 30 min)")
    hm = t.hour * 60 + t.minute
    if f["skip_midday"] and 11 * 60 <= hm < 13 * 60 + 30:
        r.append("Midday chop 11:00–13:30")
    if state["last_loss"] is not None and state["last_loss"] <= t and \
            (t - state["last_loss"]).total_seconds() / 60 < f["cooldown_after_loss"]:
        r.append(f"Stop-loss hit here {int((t - state['last_loss']).total_seconds() // 60)} min ago")
    if state["last_time"] is not None and state["last_side"] != sgn and \
            (t - state["last_time"]).total_seconds() / 60 < f["no_flip_minutes"]:
        r.append("Flip-flop (opposite call just before)")
    if state["count"] >= f["max_per_stock"]:
        r.append(f"Already {state['count']} calls in this stock today")
    if pd.notna(row["day_move"]) and row["day_move"] * sgn > f["max_stretch"]:
        r.append(f"Chasing: already {row['day_move']:+.1f}% today")
    if f["one_at_a_time"] and state["open_until"] is not None and t < state["open_until"]:
        r.append("Previous call still running")
    return r


def scalp_analysis(df: pd.DataFrame, min_conf: float = 60, cooldown: int = 5, horizon: int = 30,
                   filters: dict | None = None, mkt: pd.Series | None = None) -> dict:
    """Current scalp call + every past signal with its outcome (did T1 come before the stop?).
    With `filters`, each signal is marked taken or blocked (with reasons) by the smart filters."""
    f = {**NO_FILTERS, **(filters or {})} if filters is not None else NO_FILTERS
    d = add_context(prepare(df), mkt)
    L, S = factor_points(d)
    lp, sp = L.sum(axis=1), S.sum(axis=1)
    side = np.where((lp >= min_conf) & (lp > sp), 1, np.where((sp >= min_conf) & (sp > lp), -1, 0))
    side = pd.Series(side, index=d.index)
    sess = pd.Series([_in_session(t) for t in d.index], index=d.index)
    side = side.where(sess, 0)
    step = d.index.to_series().diff().median()
    step = step if pd.notna(step) and step > pd.Timedelta(0) else pd.Timedelta(minutes=1)

    events, last_i = [], -10 ** 9
    hi, lo = d["High"].to_numpy(), d["Low"].to_numpy()
    days = np.array([t.date() for t in d.index])
    state, cur_day = None, None

    def fresh():
        return {"last_loss": None, "last_time": None, "last_side": 0, "count": 0, "open_until": None}

    for i in range(1, len(d)):
        if days[i] != cur_day:
            cur_day, state = days[i], fresh()
        s_ = side.iloc[i]
        if s_ == 0 or (side.iloc[i - 1] == s_) or i - last_i < cooldown:
            continue
        last_i = i
        price, stop, tgts, _ = _levels(d, i, s_, _stop_pct(f), f.get("target_ratio", 1.0))
        outcome, r_mult, bars_held, end_j = "Open", 0.0, np.nan, None
        for j in range(i + 1, min(i + 1 + horizon, len(d))):
            if days[j] != days[i]:
                outcome = "Closed (day end)"
                r_mult = (float(d["Close"].iloc[j - 1]) - price) * s_ / abs(price - stop)
                end_j = j - 1
                break
            hit_stop = lo[j] <= stop if s_ > 0 else hi[j] >= stop
            hit_t1 = hi[j] >= tgts[0] if s_ > 0 else lo[j] <= tgts[0]
            if hit_stop:  # stop first if both in same bar (conservative)
                outcome, r_mult, bars_held, end_j = "Stop hit ❌", -1.0, j - i, j
                break
            if hit_t1:
                outcome, r_mult, bars_held, end_j = "Target 1 hit ✅", float(f.get("target_ratio", 1.0)), j - i, j
                break
        else:
            if i + horizon < len(d):
                outcome = "No result in 30 bars"
                end_j = i + horizon
        t = d.index[i]
        row = d.iloc[i]
        reasons = _block_reasons(t, s_, row, state, f, float(lp.iloc[i] if s_ > 0 else sp.iloc[i]))
        taken = not reasons
        if taken:
            end_t = d.index[end_j] if end_j is not None else d.index[-1] + step
            state.update(last_time=t, last_side=s_, count=state["count"] + 1, open_until=end_t)
            if outcome == "Stop hit ❌":
                state["last_loss"] = end_t
        events.append({"time": t, "side": "BUY" if s_ > 0 else "SELL",
                       "confidence": float(lp.iloc[i] if s_ > 0 else sp.iloc[i]),
                       "entry": price, "stop": stop, "t1": tgts[0], "outcome": outcome, "r": r_mult,
                       "bars": bars_held, "taken": taken, "blocked_by": "; ".join(reasons),
                       "mkt30": row["mkt30"], "buy_pressure": row["buy_pressure"],
                       "vwap_side": row["vwap_side"], "day_move": row["day_move"]})
    ev = pd.DataFrame(events)

    # current state
    i = len(d) - 1
    cur_side = int(side.iloc[i])
    lead = 1 if lp.iloc[i] >= sp.iloc[i] else -1
    price, stop, tgts, a = _levels(d, i, cur_side or lead, _stop_pct(f), f.get("target_ratio", 1.0))
    conf = float(lp.iloc[i] if (cur_side or lead) > 0 else sp.iloc[i])
    reasons = []
    for k, w in FACTORS.items():
        lv, sv = float(L[k].iloc[i]), float(S[k].iloc[i])
        reasons.append({"Factor": k, "Max": w, "Long pts": lv, "Short pts": sv,
                        "Favours": "🟢 Buy" if lv > sv else "🔴 Sell" if sv > lv else "⚪ —"})
    blocked_now = []
    if cur_side != 0 and state is not None and days[i] == cur_day:
        blocked_now = _block_reasons(d.index[i], cur_side, d.iloc[i], state, f, conf)
    reason_wait = ""
    if cur_side > 0 and not blocked_now:
        call = "SCALP BUY"
    elif cur_side < 0 and not blocked_now:
        call = "SCALP SELL"
    else:
        call = "NO SCALP"
        if blocked_now:
            reason_wait = (f"{'BUY' if cur_side > 0 else 'SELL'} signal blocked: " + "; ".join(blocked_now))
        elif not sess.iloc[i]:
            reason_wait = "Outside scalping hours (9:20–15:15)."
        else:
            reason_wait = f"Best side has only {conf:.0f}% (needs {min_conf:.0f}%)."

    taken_ev = ev[ev["taken"]] if len(ev) else ev
    blocked_ev = ev[~ev["taken"]] if len(ev) else ev
    today_d = d.index[-1].date()

    def stats(e):
        done = e[e["outcome"].isin(["Target 1 hit ✅", "Stop hit ❌"])] if len(e) else e
        wins = e[e["outcome"] == "Target 1 hit ✅"] if len(e) else e
        return {"signals": len(e), "decided": len(done),
                "win_bars": float(wins["bars"].median()) if len(wins) else np.nan,
                "win_bars_p75": float(wins["bars"].quantile(0.75)) if len(wins) else np.nan,
                "win_rate": float((done["outcome"] == "Target 1 hit ✅").mean() * 100) if len(done) else np.nan,
                "avg_r": float(e["r"].mean()) if len(e) else np.nan}

    def today(e):
        return e[e["time"].dt.date == today_d] if len(e) else e
    last = d.iloc[-1]
    return {
        "call": call, "side": cur_side if not blocked_now else 0, "signal_side": cur_side, "lead": lead,
        "confidence": conf, "min_conf": min_conf, "price": price, "stop": stop, "targets": tgts, "atr": a,
        "reasons": reasons, "wait_reason": reason_wait, "blocked_now": blocked_now,
        "events": taken_ev, "all_events": ev, "stats_today": stats(today(taken_ev)), "stats_all": stats(taken_ev),
        "stats_blocked": stats(blocked_ev), "stats_unfiltered": stats(ev), "data": d,
        "long_pts": lp, "short_pts": sp,
        "buy_pressure": float(last["buy_pressure"]), "vwap_side": int(last["vwap_side"]) if pd.notna(last["vwap_side"]) else 0,
        "mkt30": float(last["mkt30"]) if pd.notna(last["mkt30"]) else np.nan,
        "day_move": float(last["day_move"]),
    }
