"""Pattern recognition: candlestick patterns, chart patterns and harmonic patterns.

Pure pandas/numpy. Input: OHLCV DataFrame (Open, High, Low, Close, Volume).
Every pattern returns a dict with at least:
    name, bias ("Bullish"/"Bearish"/"Neutral"), when (timestamp), status, reliability, meaning
Chart/harmonic patterns also carry key levels: entry/trigger, stop, target, and points for drawing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import atr as _atr


# ============================================================================
# 1. Candlestick patterns
# ============================================================================
CANDLE_INFO = {
    # name: (bias, reliability, meaning)
    "Bullish Engulfing": ("Bullish", "High", "Buyers fully overpowered the previous red candle after a fall."),
    "Bearish Engulfing": ("Bearish", "High", "Sellers fully overpowered the previous green candle after a rise."),
    "Bullish Harami": ("Bullish", "Medium", "Small green candle inside a big red one: selling is drying up."),
    "Bearish Harami": ("Bearish", "Medium", "Small red candle inside a big green one: buying is drying up."),
    "Bullish Harami Cross": ("Bullish", "Medium", "Doji inside a big red candle after a fall: indecision, possible bottom."),
    "Bearish Harami Cross": ("Bearish", "Medium", "Doji inside a big green candle after a rise: indecision, possible top."),
    "Hammer": ("Bullish", "Medium", "Long lower wick after a fall: sellers pushed down, buyers took it back."),
    "Inverted Hammer": ("Bullish", "Low", "Long upper wick after a fall: first sign of buyers; needs confirmation."),
    "Hanging Man": ("Bearish", "Low", "Hammer shape after a rise: selling appeared; needs confirmation."),
    "Shooting Star": ("Bearish", "Medium", "Long upper wick after a rise: buyers rejected at the top."),
    "Doji": ("Neutral", "Low", "Open ≈ close: indecision. Watch the next candle."),
    "Dragonfly Doji": ("Bullish", "Medium", "Doji with a long lower wick after a fall: rejection of lower prices."),
    "Gravestone Doji": ("Bearish", "Medium", "Doji with a long upper wick after a rise: rejection of higher prices."),
    "Morning Star": ("Bullish", "High", "Big red, small candle, big green: a 3-candle bottom reversal."),
    "Evening Star": ("Bearish", "High", "Big green, small candle, big red: a 3-candle top reversal."),
    "Three White Soldiers": ("Bullish", "High", "Three strong green candles in a row: steady buying."),
    "Three Black Crows": ("Bearish", "High", "Three strong red candles in a row: steady selling."),
    "Piercing Line": ("Bullish", "Medium", "Green candle closes above the middle of the previous red candle."),
    "Dark Cloud Cover": ("Bearish", "Medium", "Red candle closes below the middle of the previous green candle."),
    "Tweezer Bottom": ("Bullish", "Medium", "Two candles with the same low after a fall: support held twice."),
    "Tweezer Top": ("Bearish", "Medium", "Two candles with the same high after a rise: resistance held twice."),
    "Bullish Marubozu": ("Bullish", "Medium", "Full green body with almost no wicks: buyers in control all session."),
    "Bearish Marubozu": ("Bearish", "Medium", "Full red body with almost no wicks: sellers in control all session."),
    "Bullish Kicker": ("Bullish", "High", "Gap up and green right after a red candle: sudden shift to buyers."),
    "Bearish Kicker": ("Bearish", "High", "Gap down and red right after a green candle: sudden shift to sellers."),
    "Three Inside Up": ("Bullish", "High", "Bullish harami confirmed by a close above the first candle."),
    "Three Inside Down": ("Bearish", "High", "Bearish harami confirmed by a close below the first candle."),
    "Three Outside Up": ("Bullish", "High", "Bullish engulfing confirmed by another higher close."),
    "Three Outside Down": ("Bearish", "High", "Bearish engulfing confirmed by another lower close."),
    "Spinning Top": ("Neutral", "Low", "Small body, wicks on both sides: balance between buyers and sellers."),
    "Bullish Elephant Bar": ("Bullish", "Medium", "Huge green candle (≥2× normal size): strong momentum entering."),
    "Bearish Elephant Bar": ("Bearish", "Medium", "Huge red candle (≥2× normal size): strong momentum leaving."),
}


def candlestick_patterns(df: pd.DataFrame, lookback: int = 10) -> list[dict]:
    """Detect candlestick patterns in the last `lookback` bars (needs ~30 bars of history)."""
    if len(df) < 25:
        return []
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-9)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    green, red = c > o, c < o
    avg_body = pd.Series(body).rolling(20, min_periods=10).mean().to_numpy()
    a = _atr(df).to_numpy()
    a = np.where(np.isnan(a), pd.Series(rng).rolling(14, min_periods=1).mean().to_numpy(), a)
    ema = pd.Series(c).ewm(span=10, adjust=False).mean().to_numpy()

    def downtrend(i):  # prior trend into bar i
        return i >= 6 and c[i - 1] < ema[i - 1] and c[i - 1] < c[i - 6]

    def uptrend(i):
        return i >= 6 and c[i - 1] > ema[i - 1] and c[i - 1] > c[i - 6]

    def big(i):
        return body[i] >= 1.0 * avg_body[i] if not np.isnan(avg_body[i]) else body[i] >= 0.6 * rng[i]

    def small(i):
        return body[i] <= 0.5 * avg_body[i] if not np.isnan(avg_body[i]) else body[i] <= 0.3 * rng[i]

    def doji(i):
        return body[i] <= 0.1 * rng[i]

    found = []
    n = len(df)
    for i in range(max(3, n - lookback), n):
        hits = []
        dt, ut = downtrend(i), uptrend(i)
        p = i - 1
        # --- single candle
        if doji(i):
            if lower[i] >= 0.6 * rng[i] and upper[i] <= 0.1 * rng[i] and dt:
                hits.append("Dragonfly Doji")
            elif upper[i] >= 0.6 * rng[i] and lower[i] <= 0.1 * rng[i] and ut:
                hits.append("Gravestone Doji")
            elif rng[i] >= 0.5 * a[i]:
                hits.append("Doji")
        else:
            hammer_shape = lower[i] >= 2 * body[i] and upper[i] <= 0.35 * body[i] + 0.05 * rng[i]
            inv_shape = upper[i] >= 2 * body[i] and lower[i] <= 0.35 * body[i] + 0.05 * rng[i]
            if hammer_shape and dt:
                hits.append("Hammer")
            elif hammer_shape and ut:
                hits.append("Hanging Man")
            if inv_shape and dt:
                hits.append("Inverted Hammer")
            elif inv_shape and ut:
                hits.append("Shooting Star")
            if body[i] >= 0.9 * rng[i] and big(i):
                hits.append("Bullish Marubozu" if green[i] else "Bearish Marubozu")
            if small(i) and upper[i] >= body[i] and lower[i] >= body[i] and not (hammer_shape or inv_shape):
                hits.append("Spinning Top")
            if (not np.isnan(avg_body[i]) and body[i] >= 2 * avg_body[i] and rng[i] >= 1.5 * a[i]
                    and body[i] >= 0.7 * rng[i]):
                hits.append("Bullish Elephant Bar" if green[i] else "Bearish Elephant Bar")
        # --- two candles
        if red[p] and green[i] and o[i] <= c[p] and c[i] >= o[p] and body[i] > body[p] and dt:
            hits.append("Bullish Engulfing")
        if green[p] and red[i] and o[i] >= c[p] and c[i] <= o[p] and body[i] > body[p] and ut:
            hits.append("Bearish Engulfing")
        inside = max(o[i], c[i]) <= max(o[p], c[p]) and min(o[i], c[i]) >= min(o[p], c[p])
        if red[p] and big(p) and inside and dt:
            hits.append("Bullish Harami Cross" if doji(i) else "Bullish Harami" if green[i] else None)
        if green[p] and big(p) and inside and ut:
            hits.append("Bearish Harami Cross" if doji(i) else "Bearish Harami" if red[i] else None)
        mid_p = (o[p] + c[p]) / 2
        if red[p] and big(p) and green[i] and o[i] < c[p] and mid_p < c[i] < o[p] and dt:
            hits.append("Piercing Line")
        if green[p] and big(p) and red[i] and o[i] > c[p] and o[p] < c[i] < mid_p and ut:
            hits.append("Dark Cloud Cover")
        tol = 0.05 * a[i]
        if abs(l[i] - l[p]) <= tol and red[p] and green[i] and dt:
            hits.append("Tweezer Bottom")
        if abs(h[i] - h[p]) <= tol and green[p] and red[i] and ut:
            hits.append("Tweezer Top")
        if red[p] and big(p) and green[i] and big(i) and o[i] >= o[p] + 0.2 * a[i] and l[i] > c[p]:
            hits.append("Bullish Kicker")
        if green[p] and big(p) and red[i] and big(i) and o[i] <= o[p] - 0.2 * a[i] and h[i] < c[p]:
            hits.append("Bearish Kicker")
        # --- three candles
        q = i - 2
        if q >= 0:
            if (red[q] and big(q) and small(p) and green[i] and big(i)
                    and c[i] > (o[q] + c[q]) / 2 and max(o[p], c[p]) < c[q] + 0.1 * a[i] and downtrend(q + 1)):
                hits.append("Morning Star")
            if (green[q] and big(q) and small(p) and red[i] and big(i)
                    and c[i] < (o[q] + c[q]) / 2 and min(o[p], c[p]) > c[q] - 0.1 * a[i] and uptrend(q + 1)):
                hits.append("Evening Star")
            three_g = all(green[k] and big(k) and upper[k] <= 0.3 * rng[k] for k in (q, p, i)) and c[q] < c[p] < c[i] \
                and o[q] < o[p] < c[q] and o[p] < o[i] < c[p]
            three_r = all(red[k] and big(k) and lower[k] <= 0.3 * rng[k] for k in (q, p, i)) and c[q] > c[p] > c[i] \
                and c[q] < o[p] < o[q] and c[p] < o[i] < o[p]
            if three_g:
                hits.append("Three White Soldiers")
            if three_r:
                hits.append("Three Black Crows")
            inside_qp = max(o[p], c[p]) <= max(o[q], c[q]) and min(o[p], c[p]) >= min(o[q], c[q])
            if red[q] and big(q) and green[p] and inside_qp and green[i] and c[i] > o[q] and downtrend(q + 1):
                hits.append("Three Inside Up")
            if green[q] and big(q) and red[p] and inside_qp and red[i] and c[i] < o[q] and uptrend(q + 1):
                hits.append("Three Inside Down")
            if red[q] and green[p] and o[p] <= c[q] and c[p] >= o[q] and body[p] > body[q] and c[i] > c[p] \
                    and downtrend(q + 1):
                hits.append("Three Outside Up")
            if green[q] and red[p] and o[p] >= c[q] and c[p] <= o[q] and body[p] > body[q] and c[i] < c[p] \
                    and uptrend(q + 1):
                hits.append("Three Outside Down")
        for name in hits:
            if not name:
                continue
            bias, rel, meaning = CANDLE_INFO[name]
            found.append({"name": name, "bias": bias, "reliability": rel, "meaning": meaning,
                          "when": df.index[i], "bars_ago": n - 1 - i, "price": float(c[i]),
                          "kind": "Candlestick", "status": "Formed"})
    return found


# ============================================================================
# 2. Swing points (ZigZag)
# ============================================================================
def zigzag(df: pd.DataFrame, k: float = 2.0) -> list[tuple[int, float, str]]:
    """Swing highs/lows: a new swing is confirmed once price reverses by k × ATR.
    Returns [(bar_position, price, 'H'|'L'), ...]; the last point is the current,
    still-unconfirmed extreme."""
    h, l = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    a = _atr(df).bfill().to_numpy()
    if len(df) < 5 or np.all(np.isnan(a)):
        return []
    piv = []
    trend = 0
    ext_i, ext_p = 0, h[0]
    lo_i, lo_p = 0, l[0]
    hi_i, hi_p = 0, h[0]
    for i in range(1, len(df)):
        th = k * a[i]
        if trend == 0:
            if h[i] > hi_p:
                hi_i, hi_p = i, h[i]
            if l[i] < lo_p:
                lo_i, lo_p = i, l[i]
            if hi_p - l[i] >= th and hi_i < i:
                piv.append((hi_i, hi_p, "H")); trend = -1; ext_i, ext_p = i, l[i]
            elif h[i] - lo_p >= th and lo_i < i:
                piv.append((lo_i, lo_p, "L")); trend = 1; ext_i, ext_p = i, h[i]
        elif trend == 1:
            if h[i] >= ext_p:
                ext_i, ext_p = i, h[i]
            elif ext_p - l[i] >= th:
                piv.append((ext_i, ext_p, "H")); trend = -1; ext_i, ext_p = i, l[i]
        else:
            if l[i] <= ext_p:
                ext_i, ext_p = i, l[i]
            elif h[i] - ext_p >= th:
                piv.append((ext_i, ext_p, "L")); trend = 1; ext_i, ext_p = i, h[i]
    if trend != 0:
        piv.append((ext_i, ext_p, "H" if trend == 1 else "L"))
    return piv


# ============================================================================
# 3. Classic chart patterns
# ============================================================================
CHART_INFO = {
    "Head & Shoulders": ("Bearish", "High", "Three peaks, middle highest. Breaking the neckline signals a trend reversal down."),
    "Inverse Head & Shoulders": ("Bullish", "High", "Three troughs, middle lowest. Breaking the neckline signals a reversal up."),
    "Double Top": ("Bearish", "Medium", "Two peaks at the same price ('M'). Bearish once the middle low breaks."),
    "Double Bottom": ("Bullish", "Medium", "Two lows at the same price ('W'). Bullish once the middle high breaks."),
    "Triple Top": ("Bearish", "High", "Three peaks at one level: strong resistance; bearish below the lows."),
    "Triple Bottom": ("Bullish", "High", "Three lows at one level: strong support; bullish above the highs."),
    "Ascending Triangle": ("Bullish", "Medium", "Flat top, rising lows: buyers pressing against resistance."),
    "Descending Triangle": ("Bearish", "Medium", "Flat bottom, falling highs: sellers pressing against support."),
    "Symmetrical Triangle": ("Neutral", "Medium", "Lower highs and higher lows: coiling; trade the breakout direction."),
    "Rising Wedge": ("Bearish", "Medium", "Both lines rising but converging: rally losing steam."),
    "Falling Wedge": ("Bullish", "Medium", "Both lines falling but converging: selling losing steam."),
    "Rising Channel": ("Bullish", "Low", "Parallel up-sloping lines: steady uptrend. Buy near the lower line."),
    "Falling Channel": ("Bearish", "Low", "Parallel down-sloping lines: steady downtrend. Sell near the upper line."),
    "Rectangle": ("Neutral", "Low", "Price boxed between flat support and resistance; trade the breakout."),
    "Bull Flag": ("Bullish", "High", "Sharp rise (pole) then a small pullback: continuation up is likely."),
    "Bear Flag": ("Bearish", "High", "Sharp fall (pole) then a small bounce: continuation down is likely."),
}


def _line(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    m = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0
    return lambda x: y1 + m * (x - x1), m


def _status(close_now, trigger, bullish):
    if bullish:
        return "Confirmed ✅ (broke out)" if close_now > trigger else "Forming ⏳ (wait for breakout)"
    return "Confirmed ✅ (broke down)" if close_now < trigger else "Forming ⏳ (wait for breakdown)"


def chart_patterns(df: pd.DataFrame, k: float = 2.0, max_age: int = 40) -> list[dict]:
    """Detect classic chart patterns from the latest swing points."""
    piv = zigzag(df, k)
    if len(df) < 30:
        return []
    c = df["Close"].to_numpy(float)
    n = len(df)
    now = c[-1]
    a = float(_atr(df).iloc[-1])
    tol = max(0.6 * a, 0.015 * now)
    idx = df.index
    out = []

    def add(name, trigger, stop, target, points, lines=None, bias=None):
        b, rel, meaning = CHART_INFO[name]
        bias = bias or b
        bullish = bias == "Bullish"
        if bias == "Neutral":
            bullish = now >= trigger
        st = _status(now, trigger, bullish) if bias != "Neutral" else (
            "Broke up ✅" if now > trigger else "Broke down ✅" if now < stop else "Inside pattern ⏳")
        if (bias == "Bullish" and now >= target) or (bias == "Bearish" and now <= target):
            st = "Target already reached 🏁 (move is done)"
        out.append({"name": name, "bias": bias, "reliability": rel, "meaning": meaning, "kind": "Chart",
                    "status": st, "trigger": float(trigger), "stop": float(stop), "target": float(target),
                    "when": idx[points[-1][0]], "bars_ago": n - 1 - points[-1][0], "price": now,
                    "points": [(idx[i], float(p)) for i, p in points],
                    "lines": [[(idx[i], float(p)) for i, p in ln] for ln in (lines or [])]})

    # last swing points (the very last one is the live extreme)
    P = piv[-7:]
    highs = [(i, p) for i, p, t in P if t == "H"]
    lows = [(i, p) for i, p, t in P if t == "L"]

    # ---- Head & Shoulders / Inverse (5 alternating points ending near now)
    for j in range(len(piv) - 5, max(len(piv) - 8, -1), -1):
        if j < 0:
            break
        seq = piv[j:j + 5]
        if len(seq) < 5 or n - 1 - seq[-1][0] > max_age:
            continue
        types = "".join(t for _, _, t in seq)
        (i1, p1, _), (i2, p2, _), (i3, p3, _), (i4, p4, _), (i5, p5, _) = seq
        if types == "HLHLH" and p3 > p1 + tol * 0.5 and p3 > p5 + tol * 0.5 and abs(p1 - p5) <= 1.5 * tol:
            neck, _ = _line((i2, p2), (i4, p4))
            trig = neck(n - 1)
            add("Head & Shoulders", trig, max(p5, now) + 0.5 * a, trig - (p3 - neck(i3)),
                [(i1, p1), (i2, p2), (i3, p3), (i4, p4), (i5, p5)], [[(i2, p2), (n - 1, trig)]])
            break
        if types == "LHLHL" and p3 < p1 - tol * 0.5 and p3 < p5 - tol * 0.5 and abs(p1 - p5) <= 1.5 * tol:
            neck, _ = _line((i2, p2), (i4, p4))
            trig = neck(n - 1)
            add("Inverse Head & Shoulders", trig, min(p5, now) - 0.5 * a, trig + (neck(i3) - p3),
                [(i1, p1), (i2, p2), (i3, p3), (i4, p4), (i5, p5)], [[(i2, p2), (n - 1, trig)]])
            break

    # ---- Triple / Double top & bottom
    last_types = "".join(t for _, _, t in piv[-5:])
    done = {x["name"] for x in out}
    if not done & {"Head & Shoulders", "Inverse Head & Shoulders"}:
        seq = piv[-5:]
        if len(seq) == 5 and last_types == "HLHLH":
            hs = [seq[0][1], seq[2][1], seq[4][1]]
            if max(hs) - min(hs) <= tol and n - 1 - seq[-1][0] <= max_age:
                trig = min(seq[1][1], seq[3][1])
                add("Triple Top", trig, max(hs) + 0.5 * a, trig - (np.mean(hs) - trig),
                    [(i, p) for i, p, _ in seq], [[(seq[0][0], np.mean(hs)), (n - 1, np.mean(hs))], [(seq[1][0], trig), (n - 1, trig)]])
        if len(seq) == 5 and last_types == "LHLHL":
            ls = [seq[0][1], seq[2][1], seq[4][1]]
            if max(ls) - min(ls) <= tol and n - 1 - seq[-1][0] <= max_age:
                trig = max(seq[1][1], seq[3][1])
                add("Triple Bottom", trig, min(ls) - 0.5 * a, trig + (trig - np.mean(ls)),
                    [(i, p) for i, p, _ in seq], [[(seq[0][0], np.mean(ls)), (n - 1, np.mean(ls))], [(seq[1][0], trig), (n - 1, trig)]])
        if not {x["name"] for x in out} & {"Triple Top", "Triple Bottom"}:
            for back in (0, 1):
                s3 = piv[len(piv) - 3 - back:len(piv) - back]
                if len(s3) < 3 or len(piv) - 3 - back < 0:
                    continue
                t3 = "".join(t for _, _, t in s3)
                prev_p = piv[len(piv) - 4 - back][1] if len(piv) - 4 - back >= 0 else None
                # rising lows into equal highs = ascending triangle, not a double top (and vice versa)
                depth = abs(s3[0][1] - s3[1][1])
                if t3 == "HLH" and prev_p is not None and tol < s3[1][1] - prev_p < 0.9 * depth:
                    continue
                if t3 == "LHL" and prev_p is not None and tol < prev_p - s3[1][1] < 0.9 * depth:
                    continue
                age = n - 1 - s3[-1][0]
                if t3 == "HLH" and abs(s3[0][1] - s3[2][1]) <= tol and age <= max_age \
                        and s3[0][1] - s3[1][1] >= 2 * a:
                    trig = s3[1][1]
                    top = max(s3[0][1], s3[2][1])
                    add("Double Top", trig, top + 0.5 * a, trig - (top - trig), [(i, p) for i, p, _ in s3],
                        [[(s3[1][0], trig), (n - 1, trig)]])
                    break
                if t3 == "LHL" and abs(s3[0][1] - s3[2][1]) <= tol and age <= max_age \
                        and s3[1][1] - s3[0][1] >= 2 * a:
                    trig = s3[1][1]
                    bot = min(s3[0][1], s3[2][1])
                    add("Double Bottom", trig, bot - 0.5 * a, trig + (trig - bot), [(i, p) for i, p, _ in s3],
                        [[(s3[1][0], trig), (n - 1, trig)]])
                    break

    # ---- Triangles, wedges, channels, rectangles (needs 2+ highs and 2+ lows)
    reversal_found = any(x["name"] in ("Head & Shoulders", "Inverse Head & Shoulders", "Double Top", "Double Bottom",
                                       "Triple Top", "Triple Bottom") for x in out)
    if not reversal_found and len(highs) >= 2 and len(lows) >= 2 and n - 1 - min(highs[-2][0], lows[-2][0]) <= 120:
        up_f, mu = _line(highs[-2], highs[-1])
        lo_f, ml = _line(lows[-2], lows[-1])
        flat = 0.15 * a  # slope per bar considered "flat"
        top_now, bot_now = up_f(n - 1), lo_f(n - 1)
        width_then = up_f(min(highs[-2][0], lows[-2][0])) - lo_f(min(highs[-2][0], lows[-2][0]))
        height = max(width_then, top_now - bot_now)
        name = None
        if abs(mu) <= flat and ml > flat:
            name = "Ascending Triangle"
        elif mu < -flat and abs(ml) <= flat:
            name = "Descending Triangle"
        elif mu < -flat and ml > flat:
            name = "Symmetrical Triangle"
        elif mu > flat and ml > flat:
            name = "Rising Wedge" if ml > mu * 1.15 else "Rising Channel"
        elif mu < -flat and ml < -flat:
            name = "Falling Wedge" if mu < ml * 1.15 else "Falling Channel"
        elif abs(mu) <= flat and abs(ml) <= flat:
            name = "Rectangle"
        if name and top_now > bot_now:
            start = min(highs[-2][0], lows[-2][0])
            lines = [[(highs[-2][0], highs[-2][1]), (n - 1, top_now)], [(lows[-2][0], lows[-2][1]), (n - 1, bot_now)]]
            pts = sorted(highs[-2:] + lows[-2:])
            b = CHART_INFO[name][0]
            if b == "Bullish" or (b == "Neutral" and now >= (top_now + bot_now) / 2):
                add(name, top_now, bot_now - 0.3 * a, top_now + height, pts, lines,
                    bias=None if b != "Neutral" else "Neutral")
            else:
                add(name, bot_now, top_now + 0.3 * a, bot_now - height, pts, lines,
                    bias=None if b != "Neutral" else "Neutral")
            if b == "Neutral":  # neutral patterns: trigger shown as both edges
                out[-1].update(trigger=float(top_now), stop=float(bot_now),
                               target=float(top_now + height if now >= bot_now else bot_now - height))
            _ = start

    # ---- Flags (pole of >= 6 ATR within 12 bars, then a tight pullback of 4-20 bars)
    hi, lo = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    for flag_len in range(4, 21):
        s = n - flag_len  # flag starts at s
        if s - 12 < 0:
            break
        pole_lo_i = s - 12 + int(np.argmin(lo[s - 12:s]))
        pole_hi_i = s - 12 + int(np.argmax(hi[s - 12:s]))
        f_hi, f_lo = hi[s:].max(), lo[s:].min()
        up_pole = hi[pole_hi_i] - lo[pole_lo_i]
        if pole_lo_i < pole_hi_i and up_pole >= 6 * a and pole_hi_i >= s - 3 \
                and (hi[pole_hi_i] - f_lo) <= 0.5 * up_pole and (f_hi - f_lo) <= 0.5 * up_pole:
            add("Bull Flag", f_hi, f_lo - 0.3 * a, f_hi + up_pole,
                [(pole_lo_i, lo[pole_lo_i]), (pole_hi_i, hi[pole_hi_i]), (n - 1, c[-1])],
                [[(s, f_hi), (n - 1, f_hi)], [(s, f_lo), (n - 1, f_lo)]])
            break
        dn_pole = hi[pole_hi_i] - lo[pole_lo_i]
        if pole_hi_i < pole_lo_i and dn_pole >= 6 * a and pole_lo_i >= s - 3 \
                and (f_hi - lo[pole_lo_i]) <= 0.5 * dn_pole and (f_hi - f_lo) <= 0.5 * dn_pole:
            add("Bear Flag", f_lo, f_hi + 0.3 * a, f_lo - dn_pole,
                [(pole_hi_i, hi[pole_hi_i]), (pole_lo_i, lo[pole_lo_i]), (n - 1, c[-1])],
                [[(s, f_hi), (n - 1, f_hi)], [(s, f_lo), (n - 1, f_lo)]])
            break
    return out


# ============================================================================
# 4. Harmonic patterns (XABCD)
# ============================================================================
# ratio ranges: AB/XA, BC/AB, CD/BC, AD/XA  (None = not checked)
HARMONICS = {
    "Gartley":   dict(ab=(0.58, 0.66), bc=(0.382, 0.886), cd=(1.13, 1.618), ad=(0.75, 0.82)),
    "Bat":       dict(ab=(0.382, 0.50), bc=(0.382, 0.886), cd=(1.618, 2.618), ad=(0.85, 0.92)),
    "Alt Bat":   dict(ab=(0.33, 0.382), bc=(0.382, 0.886), cd=(2.0, 3.618), ad=(1.09, 1.17)),
    "Butterfly": dict(ab=(0.75, 0.82), bc=(0.382, 0.886), cd=(1.618, 2.618), ad=(1.24, 1.66)),
    "Crab":      dict(ab=(0.382, 0.618), bc=(0.382, 0.886), cd=(2.24, 3.618), ad=(1.58, 1.66)),
    "Deep Crab": dict(ab=(0.85, 0.92), bc=(0.382, 0.886), cd=(2.0, 3.618), ad=(1.58, 1.66)),
    "Cypher":    dict(ab=(0.382, 0.618), bc=(1.13, 1.414), cd=None, ad=(0.75, 0.82)),  # Cypher: D = 0.786 of XC
    "Shark":     dict(ab=(0.382, 0.886), bc=(1.13, 1.618), cd=(1.618, 2.24), ad=(0.88, 1.13)),
}
HARMONIC_MEANING = ("Price-ratio (Fibonacci) reversal pattern. Point D is the Potential Reversal Zone: "
                    "look to {act} near D with a tight stop beyond it.")


def _in(v, rng, slack=0.03):
    return rng is None or (rng[0] * (1 - slack) <= v <= rng[1] * (1 + slack))


def harmonic_patterns(df: pd.DataFrame, k: float = 1.5, max_age: int = 15) -> list[dict]:
    """Completed harmonics (D = latest swing) and potential ones (D still to come: shows the zone)."""
    piv = zigzag(df, k)
    out = []
    if len(piv) < 4:
        return out
    n = len(df)
    now = float(df["Close"].iloc[-1])
    a = float(_atr(df).iloc[-1])
    idx = df.index

    def ratios(X, A, B, C, D=None):
        xa, ab, bc = abs(A - X), abs(B - A), abs(C - B)
        r = {"ab": ab / xa if xa else 0, "bc": bc / ab if ab else 0}
        if D is not None:
            r["cd"] = abs(D - C) / bc if bc else 0
            r["ad"] = abs(A - D) / xa if xa else 0
            r["xc"] = abs(C - D) / abs(C - X) if C != X else 0
        return r

    # --- completed: last 5 swings
    if len(piv) >= 5:
        (xi, X, xt), (ai, A, _), (bi, B, _), (ci, C, _), (di, D, _) = piv[-5:]
        if n - 1 - di <= max_age:
            bullish = xt == "L"  # X low -> A high -> ... -> D low => bullish
            r = ratios(X, A, B, C, D)
            for name, spec in HARMONICS.items():
                ok = _in(r["ab"], spec["ab"]) and _in(r["bc"], spec["bc"]) and _in(r["ad"], spec["ad"])
                if name == "Cypher":
                    ok = ok and _in(r["xc"], (0.75, 0.82))
                elif spec["cd"]:
                    ok = ok and _in(r["cd"], spec["cd"])
                if ok:
                    ad = abs(A - D)
                    sign = 1 if bullish else -1
                    out.append({
                        "name": f"{'Bullish' if bullish else 'Bearish'} {name}", "bias": "Bullish" if bullish else "Bearish",
                        "reliability": "Medium", "kind": "Harmonic",
                        "meaning": HARMONIC_MEANING.format(act="buy" if bullish else "sell"),
                        "status": ("Target already reached 🏁" if (now - (D + sign * 0.618 * ad)) * sign >= 0 else
                                   "Completed at D ✅" if (now - D) * sign >= 0 else "At D – waiting for reversal ⏳"),
                        "trigger": float(D), "stop": float(D - sign * max(a, 0.13 * abs(A - X))),
                        "target": float(D + sign * 0.618 * ad), "target1": float(D + sign * 0.382 * ad),
                        "when": idx[di], "bars_ago": n - 1 - di, "price": now,
                        "ratios": {k_: round(v, 3) for k_, v in r.items()},
                        "points": [(idx[i], float(p)) for i, p in ((xi, X), (ai, A), (bi, B), (ci, C), (di, D))],
                        "labels": ["X", "A", "B", "C", "D"], "lines": [],
                    })
                    break
    # --- potential: X,A,B,C known, D still to come.
    #     Case 1: D-leg already running (last pivot = live move toward D) -> X..C = piv[-5:-1]
    #     Case 2: C itself is the live extreme -> X..C = piv[-4:]
    if not any(p["kind"] == "Harmonic" for p in out):
        cands = ([piv[-5:-1]] if len(piv) >= 5 else []) + [piv[-4:]]
        for quad in cands:
            (xi, X, xt), (ai, A, _), (bi, B, _), (ci, C, _) = quad
            bullish = xt == "L"
            r = ratios(X, A, B, C)
            hit = None
            for name, spec in HARMONICS.items():
                if not (_in(r["ab"], spec["ab"]) and _in(r["bc"], spec["bc"])):
                    continue
                xa = abs(A - X)
                d1, d2 = A - (1 if bullish else -1) * spec["ad"][0] * xa, A - (1 if bullish else -1) * spec["ad"][1] * xa
                lo_d, hi_d = min(d1, d2), max(d1, d2)
                if bullish and not (hi_d < C and now > lo_d - a):
                    continue
                if not bullish and not (lo_d > C and now < hi_d + a):
                    continue
                hit = (name, lo_d, hi_d)
                break
            if not hit:
                continue
            name, lo_d, hi_d = hit
            mid = (lo_d + hi_d) / 2
            out.append({
                "name": f"Potential {'Bullish' if bullish else 'Bearish'} {name}", "bias": "Bullish" if bullish else "Bearish",
                "reliability": "Low", "kind": "Harmonic",
                "meaning": f"X-A-B-C fit a {name}. Watch the D zone ₹{lo_d:,.1f}–₹{hi_d:,.1f} for a reversal "
                           f"({'buy' if bullish else 'sell'} setup). Not a signal until price reaches D and turns.",
                "status": "Forming ⏳ (D not reached)", "trigger": float(mid),
                "stop": float(lo_d - a) if bullish else float(hi_d + a),
                "target": float(mid + (1 if bullish else -1) * 0.618 * abs(A - mid)),
                "when": idx[ci], "bars_ago": n - 1 - ci, "price": now, "d_zone": (float(lo_d), float(hi_d)),
                "ratios": {k_: round(v, 3) for k_, v in r.items()},
                "points": [(idx[i], float(p)) for i, p in ((xi, X), (ai, A), (bi, B), (ci, C))],
                "labels": ["X", "A", "B", "C"], "lines": [],
            })
            break
    # --- AB=CD (4 points A,B,C,D)
    if len(piv) >= 4:
        (ai, A, at), (bi, B, _), (ci, C, _), (di, D, _) = piv[-4:]
        bc_ab = abs(C - B) / abs(B - A) if B != A else 0
        cd_ab = abs(D - C) / abs(B - A) if B != A else 0
        if n - 1 - di <= max_age and 0.58 <= bc_ab <= 0.89 and 0.9 <= cd_ab <= 1.15:
            bullish = at == "H"  # A high -> B low -> C high -> D low  => bullish at D
            sign = 1 if bullish else -1
            out.append({
                "name": f"{'Bullish' if bullish else 'Bearish'} AB=CD", "bias": "Bullish" if bullish else "Bearish",
                "reliability": "Medium", "kind": "Harmonic",
                "meaning": "Two equal price legs (AB ≈ CD). D often marks the end of the move.",
                "status": "Completed at D ✅", "trigger": float(D), "stop": float(D - sign * a),
                "target": float(D + sign * 0.618 * abs(A - D)), "when": idx[di], "bars_ago": n - 1 - di,
                "price": now, "ratios": {"bc/ab": round(bc_ab, 3), "cd/ab": round(cd_ab, 3)},
                "points": [(idx[i], float(p)) for i, p in ((ai, A), (bi, B), (ci, C), (di, D))],
                "labels": ["A", "B", "C", "D"], "lines": [],
            })
    return out


# ============================================================================
# 5. Summary
# ============================================================================
REL_W = {"High": 3, "Medium": 2, "Low": 1}


def all_patterns(df: pd.DataFrame, candle_lookback: int = 5) -> dict:
    cs = candlestick_patterns(df, candle_lookback)
    ch = chart_patterns(df)
    hm = harmonic_patterns(df)
    bull = sum(REL_W[p["reliability"]] for p in cs + ch + hm if p["bias"] == "Bullish")
    bear = sum(REL_W[p["reliability"]] for p in cs + ch + hm if p["bias"] == "Bearish")
    tot = bull + bear
    lean = "Bullish" if bull > bear else "Bearish" if bear > bull else "Neutral"
    return {"candles": cs, "chart": ch, "harmonic": hm, "bull_pts": bull, "bear_pts": bear,
            "lean": lean, "lean_pct": (max(bull, bear) / tot * 100) if tot else 0}
