"""Background market engine.

Runs in one thread for the whole app (shared by every viewer):
  * every ~20 s  : 1-minute candles for all watched stocks  -> live prices
  * every ~60 s  : breakout scan on live daily candles       -> breakout table + calls
  * every ~60 s  : scalp scan on 1-minute candles            -> scalp radar + calls
  * every loop   : checks every open call against the 1-minute highs/lows
                   -> Target 1/2/3 hit, stop hit, still open (the live report)
Calls are saved to calls_log.json so the report survives app restarts.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback

import numpy as np
import pandas as pd

import analysis as A
import data as D
import scalp as SC

IST = "Asia/Kolkata"
SCALP_EXIT = (15, 15)  # scalp calls are closed at 15:15


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=IST)


def _to_ist(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(IST) if df.index.tz is None else df.index.tz_convert(IST)
    return df


def default_fetch_1m(symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    tick = [D.to_yahoo(s) for s in symbols]
    raw = yf.download(tick, period=period, interval="1m", group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    out = {}
    for s, t in zip(symbols, tick):
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = _to_ist(df)
            if len(df):
                out[s] = df
        except KeyError:
            continue
    return out


def default_fetch_daily(symbols: list[str]) -> dict[str, pd.DataFrame]:
    return D.history_many(symbols, "1y")


class GitHubStore:
    """Keeps the call log in the user's own GitHub repo, on a separate branch ('report-data') so saving
    never triggers a redeploy of the app (Streamlit only watches the app's own branch).
    One file per day: calls/YYYY-MM-DD.json."""
    API = "https://api.github.com"

    def __init__(self, token: str, repo: str, branch: str = "report-data"):
        import requests
        self.token, self.repo, self.branch = token, repo.strip().strip("/"), branch
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28"})
        self.sha: dict[str, str] = {}
        self.ok, self.error, self.last_save = False, "", None
        try:
            self._ensure_branch()
            self.ok = True
        except Exception as e:
            self.error = str(e)[:200]

    def _url(self, path):
        return f"{self.API}/repos/{self.repo}/{path}"

    def _ensure_branch(self):
        r = self.s.get(self._url(f"git/ref/heads/{self.branch}"), timeout=10)
        if r.status_code == 200:
            return
        if r.status_code not in (404, 409):
            r.raise_for_status()
        repo = self.s.get(self._url(""), timeout=10)
        repo.raise_for_status()
        base = repo.json()["default_branch"]
        ref = self.s.get(self._url(f"git/ref/heads/{base}"), timeout=10)
        ref.raise_for_status()
        mk = self.s.post(self._url("git/refs"), json={"ref": f"refs/heads/{self.branch}",
                                                      "sha": ref.json()["object"]["sha"]}, timeout=10)
        if mk.status_code not in (200, 201, 422):  # 422 = already exists
            mk.raise_for_status()

    def list(self, folder: str) -> list[str]:
        r = self.s.get(self._url(f"contents/{folder}"), params={"ref": self.branch}, timeout=10)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return [x["name"] for x in r.json() if x.get("type") == "file"]

    def read(self, path: str) -> str | None:
        import base64
        r = self.s.get(self._url(f"contents/{path}"), params={"ref": self.branch}, timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        j = r.json()
        self.sha[path] = j["sha"]
        if j.get("content"):
            return base64.b64decode(j["content"]).decode()
        raw = self.s.get(j["download_url"], timeout=15)  # files over 1 MB come without inline content
        return raw.text

    def write(self, path: str, text: str):
        import base64
        body = {"message": f"report {path}", "branch": self.branch,
                "content": base64.b64encode(text.encode()).decode()}
        for attempt in range(2):
            if path in self.sha:
                body["sha"] = self.sha[path]
            r = self.s.put(self._url(f"contents/{path}"), json=body, timeout=20)
            if r.status_code in (200, 201):
                self.sha[path] = r.json()["content"]["sha"]
                self.last_save = _now()
                return
            if r.status_code in (409, 422) and attempt == 0:  # file changed or sha unknown: refresh sha
                g = self.s.get(self._url(f"contents/{path}"), params={"ref": self.branch}, timeout=10)
                if g.status_code == 200:
                    self.sha[path] = g.json()["sha"]
                else:
                    self.sha.pop(path, None)
                continue
            r.raise_for_status()


def calls_to_json(calls: list[dict]) -> list[dict]:
    def conv(v):
        if isinstance(v, pd.Timestamp):
            return None if pd.isna(v) else v.isoformat()
        if isinstance(v, (np.floating, float)):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    return [{k: conv(v) for k, v in c.items()} for c in calls]


def calls_from_json(raw: list[dict]) -> list[dict]:
    for c in raw:
        for k in ("time", "t1_time", "t2_time", "t3_time", "stop_time", "close_time"):
            if c.get(k):
                c[k] = pd.Timestamp(c[k])
    return raw


class Engine:
    def __init__(self, universe, scalp_universe, log_path, fetch_1m=default_fetch_1m,
                 fetch_daily=default_fetch_daily, start=True, min_conf=60, scalp_conf=60, store=None):
        self.universe = list(dict.fromkeys(universe))
        self.scalp_universe = [s for s in dict.fromkeys(scalp_universe)]
        self.index_symbol = "^NSEI"
        self.all_symbols = list(dict.fromkeys(self.universe + self.scalp_universe + [self.index_symbol]))
        self.cfg = dict(SC.DEFAULT_FILTERS)
        self.fetch_1m, self.fetch_daily = fetch_1m, fetch_daily
        self.min_conf, self.scalp_conf = min_conf, scalp_conf
        self.log_path = log_path
        self.lock = threading.RLock()
        self.prices: dict[str, dict] = {}
        self.bars: dict[str, pd.DataFrame] = {}        # today's 1-minute candles
        self.scalp_hist: dict[str, pd.DataFrame] = {}  # previous days' 1-minute candles (indicator warm-up)
        self.daily: dict[str, pd.DataFrame] = {}
        self.breakouts = pd.DataFrame()
        self.scalp_now: list[dict] = []
        self.calls: list[dict] = []
        self.t = {"prices": 0.0, "daily": 0.0, "scalp_hist": 0.0, "scan": 0.0, "scalp": 0.0}
        self.updated = {"prices": None, "scan": None, "scalp": None, "eval": None}
        self.errors: list[str] = []
        self.store = store
        self._day_hash: dict[str, int] = {}
        self._last_remote = 0.0
        self._load()
        if start:
            threading.Thread(target=self._run, daemon=True, name="market-engine").start()

    # ------------------------------------------------------------------ persistence
    def _load(self):
        calls = []
        try:
            with open(self.log_path) as fh:
                calls = calls_from_json(json.load(fh))
        except Exception:
            pass
        if self.store is not None and self.store.ok:
            try:  # permanent copy in GitHub (last 30 days)
                names = sorted(n for n in self.store.list("calls") if n.endswith(".json"))[-30:]
                remote = []
                for n in names:
                    txt = self.store.read(f"calls/{n}")
                    if txt:
                        remote += calls_from_json(json.loads(txt))
                have = {c["key"] for c in calls}
                calls += [c for c in remote if c["key"] not in have]
                self._remote_days = [n[:-5] for n in names]
            except Exception as e:
                self.errors = (self.errors + [f"GitHub load: {e}"])[-5:]
        cutoff = _now() - pd.Timedelta(days=30)
        calls = [c for c in calls if c["time"] >= cutoff]
        calls.sort(key=lambda c: c["time"])
        self.calls = calls
        for day in getattr(self, "_remote_days", []):  # what GitHub already holds -> no needless re-save
            rows = [c for c in calls_to_json(calls) if c["time"][:10] == day]
            self._day_hash[day] = hash(json.dumps(rows, sort_keys=True))

    def merge_calls(self, raw: list[dict]) -> int:
        """Restore calls from a backup file; returns how many were new."""
        new = calls_from_json(raw)
        with self.lock:
            have = {c["key"] for c in self.calls}
            add = [c for c in new if c.get("key") and c["key"] not in have]
            self.calls += add
            self.calls.sort(key=lambda c: c["time"])
        self._save(force_remote=True)
        return len(add)

    def _save(self, force_remote: bool = False):
        try:
            with self.lock:
                out = calls_to_json(self.calls)
            tmp = self.log_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(out, fh)
            os.replace(tmp, self.log_path)
        except Exception:
            pass
        # permanent copy: every 5 minutes, only the days that changed
        if self.store is None or not self.store.ok:
            return
        if not force_remote and time.time() - self._last_remote < 300:
            return
        self._last_remote = time.time()
        try:
            by_day: dict[str, list] = {}
            for c in out:
                by_day.setdefault(c["time"][:10], []).append(c)
            for day, rows in by_day.items():
                h = hash(json.dumps(rows, sort_keys=True))
                if self._day_hash.get(day) == h:
                    continue
                self.store.write(f"calls/{day}.json", json.dumps(rows))
                self._day_hash[day] = h
        except Exception as e:
            self.store.error = str(e)[:200]
            self.errors = (self.errors + [f"GitHub save: {e}"])[-5:]

    # ------------------------------------------------------------------ main loop
    def _run(self):
        while True:
            is_open = D.market_status()[0]
            try:
                self.step(is_open)
            except Exception:
                self.errors = (self.errors + [traceback.format_exc(limit=2)])[-5:]
            # 1-minute data for ~130 stocks is heavy for Yahoo: refresh once a minute while the market
            # is open, and back off for 3 minutes when Yahoo sends nothing (rate limit)
            stale = time.time() - self.t["prices"] > 120
            time.sleep((180 if stale else 60) if is_open else 300)

    def step(self, is_open: bool | None = None):
        is_open = D.market_status()[0] if is_open is None else is_open
        now = time.time()
        if not self.daily or now - self.t["daily"] > 3600:
            self._load_daily()
        if not self.scalp_hist or now - self.t["scalp_hist"] > 1800:
            self._load_scalp_hist()
        if is_open or not self.prices:
            self._update_prices()
        if is_open or self.breakouts.empty:
            if now - self.t["scan"] >= 55 or self.breakouts.empty:
                self._scan_breakouts()
            if now - self.t["scalp"] >= 55 or not self.scalp_now:
                self._scan_scalp()
        self._evaluate()
        self._save()

    # ------------------------------------------------------------------ data
    def _load_daily(self):
        d = self.fetch_daily(self.universe)
        with self.lock:
            self.daily = d
        self.t["daily"] = time.time()

    def _load_scalp_hist(self):
        h = self.fetch_1m(self.scalp_universe, "5d")
        today = _now().normalize()
        with self.lock:
            self.scalp_hist = {s: df[df.index < today] for s, df in h.items()}
        self.t["scalp_hist"] = time.time()

    def _update_prices(self):
        bars = self.fetch_1m(self.all_symbols, "1d")
        prices = {}
        for s, df in bars.items():
            prev = None
            dd = self.daily.get(s)
            if dd is not None and len(dd):
                before = dd[dd.index.normalize() < df.index[-1].normalize()]
                prev = float(before["Close"].iloc[-1]) if len(before) else None
            last = float(df["Close"].iloc[-1])
            prices[s] = {"last": last, "open": float(df["Open"].iloc[0]), "high": float(df["High"].max()),
                         "low": float(df["Low"].min()), "volume": float(df["Volume"].sum()),
                         "prev_close": prev, "time": df.index[-1],
                         "day_pct": (last / prev - 1) * 100 if prev else np.nan}
        with self.lock:
            self.bars.update(bars)
            self.prices.update(prices)
        self.t["prices"] = time.time()
        self.updated["prices"] = _now()

    def live_daily(self, s: str) -> pd.DataFrame:
        df = self.daily.get(s)
        p = self.prices.get(s)
        if df is None or df.empty or not p:
            return df if df is not None else pd.DataFrame()
        day = p["time"].normalize()
        df = df.copy()
        row = [p["open"], p["high"], p["low"], p["last"], p["volume"]]
        if df.index[-1].normalize() == day:
            df.loc[df.index[-1], ["Open", "High", "Low", "Close", "Volume"]] = row
        elif df.index[-1].normalize() < day:
            df.loc[day] = row
        return df

    # ------------------------------------------------------------------ market mood
    def market_series(self) -> pd.Series | None:
        """NIFTY 50 1-minute closes; falls back to the average of the watched stocks."""
        b = self.bars.get(self.index_symbol)
        if b is not None and len(b) > 10:
            return b["Close"]
        closes = pd.DataFrame({s: df["Close"] for s, df in self.bars.items() if s in self.universe})
        if closes.empty:
            return None
        norm = closes.ffill().bfill()
        return (norm / norm.iloc[0]).mean(axis=1) * 100

    def market_state(self) -> dict:
        m = self.market_series()
        out = {"mkt30": np.nan, "nifty": None, "nifty_pct": np.nan, "breadth": np.nan, "above_vwap": np.nan,
               "mood": "Unknown"}
        if m is not None and len(m) > 2:
            m = m.sort_index()
            past = m[m.index <= m.index[-1] - pd.Timedelta(minutes=30)]
            out["mkt30"] = (m.iloc[-1] / past.iloc[-1] - 1) * 100 if len(past) else np.nan
        p = self.prices.get(self.index_symbol)
        if p:
            out["nifty"] = p["last"]
            out["nifty_pct"] = p["day_pct"]
        ups = [v["day_pct"] > 0 for s, v in self.prices.items() if s in self.universe and pd.notna(v.get("day_pct"))]
        out["breadth"] = float(np.mean(ups) * 100) if ups else np.nan
        above = []
        for s, b in self.bars.items():
            if s in self.universe and len(b) and b["Volume"].sum() > 0:
                tp = (b["High"] + b["Low"] + b["Close"]) / 3
                vw = (tp * b["Volume"]).sum() / b["Volume"].sum()
                above.append(b["Close"].iloc[-1] > vw)
        out["above_vwap"] = float(np.mean(above) * 100) if above else np.nan
        thr = self.cfg.get("mkt_threshold", 0.05)
        m30, br = out["mkt30"], out["above_vwap"] if pd.notna(out["above_vwap"]) else out["breadth"]
        if pd.notna(m30) and m30 > thr and (pd.isna(br) or br >= 50):
            out["mood"] = "Buyers in control"
        elif pd.notna(m30) and m30 < -thr and (pd.isna(br) or br < 50):
            out["mood"] = "Sellers in control"
        elif pd.notna(m30):
            out["mood"] = "Mixed / sideways"
        return out

    # ------------------------------------------------------------------ calls
    def _add_call(self, source, symbol, side, entry, stop, targets, conf, note="", when=None, ctx=None):
        when = when or _now()
        key = f"{source}|{symbol}|{side}|{when.date()}" if source == "Breakout" else \
            f"{source}|{symbol}|{side}|{when.isoformat()}"
        with self.lock:
            if any(c["key"] == key for c in self.calls):
                return
            self.calls.append({
                "key": key, "id": len(self.calls) + 1, "source": source, "symbol": symbol, "side": side,
                "time": when, "entry": float(entry), "stop": float(stop),
                "t1": float(targets[0]), "t2": float(targets[1]), "t3": float(targets[2]),
                "confidence": float(conf), "note": note, "status": "open", "stage": 0,
                "t1_time": None, "t2_time": None, "t3_time": None, "stop_time": None, "close_time": None,
                "result_r": 0.0, "last": float(entry), "best": float(entry),
                "taken": True, "blocked_by": "", **(ctx or {}),
            })

    # ------------------------------------------------------------------ scans
    def _scan_breakouts(self):
        rows = []
        mstate = self.market_state()
        for s in self.universe:
            try:
                df = self.live_daily(s)
                if df is None or len(df) < 60:
                    continue
                sd = A.add_indicators(df)
                summ = A.summarize_breakouts(A.detect_breakouts(df, 3, 1.5, d=sd))
                if not summ:
                    continue
                cs = A.confidence_series(sd).iloc[-1]
                conf = float(cs["bull"] if summ["side"] == "BUY" else cs["bear"])
                px = float(df["Close"].iloc[-1])
                r = abs(px - summ["stop"]) or px * 0.01
                sign = 1 if summ["side"] == "BUY" else -1
                tg = [px + sign * k * r for k in (1, 2, 3)]
                summ.update(symbol=s, price=px, confidence=conf, t1=tg[0], t2=tg[1], t3=tg[2],
                            score=float(A.score_series(sd).iloc[-1]))
                reasons = []
                m30 = mstate["mkt30"]
                thr = self.cfg.get("mkt_threshold", 0.05)
                if self.cfg.get("a_plus"):
                    if conf < 75:
                        reasons.append(f"A+: confidence {conf:.0f}% < 75%")
                    if not summ.get("major"):
                        reasons.append("A+: not a major breakout")
                    if pd.isna(m30) or m30 * sign <= thr:
                        reasons.append("A+: market not clearly moving our way")
                if self.cfg.get("market_filter") and pd.notna(m30):
                    if sign > 0 and m30 < -thr:
                        reasons.append(f"Market falling (NIFTY {m30:+.2f}% in 30 min)")
                    if sign < 0 and m30 > thr:
                        reasons.append(f"Market rising (NIFTY {m30:+.2f}% in 30 min)")
                p = self.prices.get(s) or {}
                dm = p.get("day_pct", np.nan)
                if pd.notna(dm) and dm * sign > 2 * self.cfg.get("max_stretch", 1.5):
                    reasons.append(f"Chasing: already {dm:+.1f}% today")
                atr_d = float(sd["ATR"].iloc[-1]) if pd.notna(sd["ATR"].iloc[-1]) else px * 0.02
                if abs(px - summ["level"]) > 1.5 * atr_d:
                    reasons.append("Too far beyond the broken level (late entry)")
                summ.update(blocked_by="; ".join(reasons), taken=not reasons)
                rows.append(summ)
                if summ["days_ago"] == 0 and conf >= self.min_conf:
                    self._add_call("Breakout", s, summ["side"], px, summ["stop"], tg, conf, summ["types"],
                                   ctx={"taken": not reasons, "blocked_by": "; ".join(reasons), "mkt30": m30,
                                        "day_move": dm, "breadth": mstate["breadth"]})
            except Exception:
                continue
        with self.lock:
            self.breakouts = pd.DataFrame(rows)
        self.t["scan"] = time.time()
        self.updated["scan"] = _now()

    def _scan_scalp(self):
        radar = []
        mkt = self.market_series()
        today = _now().normalize()
        for s in self.scalp_universe:
            try:
                today_bars = self.bars.get(s)
                if today_bars is None or today_bars.empty:
                    continue
                hist = self.scalp_hist.get(s, pd.DataFrame())
                base = pd.concat([hist, today_bars[today_bars.index >= today]]) if len(hist) else today_bars
                base = base[~base.index.duplicated(keep="last")]
                if len(base) < 60:
                    continue
                r = SC.scalp_analysis(base, self.scalp_conf, filters=self.cfg, mkt=mkt)
                ev = r["all_events"]
                if len(ev):
                    for _, e in ev[ev["time"] >= today].iterrows():
                        sign = 1 if e["side"] == "BUY" else -1
                        risk = abs(e["entry"] - e["stop"])
                        self._add_call("Scalp", s, e["side"], e["entry"], e["stop"],
                                       [e["entry"] + sign * k * abs(e["t1"] - e["entry"]) for k in (1, 1.5, 2)],
                                       e["confidence"], "1-min scalp", when=e["time"],
                                       ctx={"taken": bool(e["taken"]), "blocked_by": e["blocked_by"],
                                            "mkt30": e["mkt30"], "buy_pressure": e["buy_pressure"],
                                            "vwap_side": e["vwap_side"], "day_move": e["day_move"]})
                radar.append({"symbol": s, "call": r["call"], "side": r["side"], "signal_side": r["signal_side"],
                              "confidence": r["confidence"], "blocked": r["wait_reason"] if r["blocked_now"] else "",
                              "price": r["price"], "stop": r["stop"], "t1": r["targets"][0],
                              "t2": r["targets"][1], "t3": r["targets"][2], "buy_pressure": r["buy_pressure"],
                              "vwap_side": r["vwap_side"], "day_move": r["day_move"],
                              "win_rate": r["stats_all"]["win_rate"], "signals_today": r["stats_today"]["signals"]})
            except Exception:
                continue
        with self.lock:
            self.scalp_now = radar
        self.t["scalp"] = time.time()
        self.updated["scalp"] = _now()

    # ------------------------------------------------------------------ evaluation
    def _path_after(self, c) -> pd.DataFrame:
        """Price path after the call: daily candles for earlier days + 1-minute candles today."""
        s, t0 = c["symbol"], c["time"]
        parts = []
        dd = self.daily.get(s)
        today = _now().normalize()
        if dd is not None and t0.normalize() < today:
            mid = dd[(dd.index.normalize() > t0.normalize()) & (dd.index.normalize() < today)]
            parts.append(mid[["High", "Low", "Close"]])
        b = self.bars.get(s)
        if b is not None and len(b):
            parts.append(b[b.index > t0][["High", "Low", "Close"]])
        return pd.concat(parts) if parts else pd.DataFrame(columns=["High", "Low", "Close"])

    def _evaluate(self):
        """Track every call the way the app tells you to trade it:
        book 1/3 at T1 (stop moves to entry), 1/3 at T2 (stop moves to T1), last 1/3 at T3 or the stop.
        Scalps close what is left at 15:15. P&L uses the average of all booked parts."""
        now = _now()
        today = now.date()
        with self.lock:
            # open calls, plus today's calls tracked by the old full-exit rule (re-checked once)
            calls = [c for c in self.calls if not c.get("closed")
                     or ("avg_px" not in c and c["time"].date() == today)]
        for c in calls:
            path = self._path_after(c)
            sign = 1 if c["side"] == "BUY" else -1
            stop, entry = c["stop"], c["entry"]
            risk = abs(entry - stop) or entry * 0.005
            stage, closed, close_t, exit_px = 0, False, None, None
            t_hits = {1: None, 2: None, 3: None}
            stop_t = None
            best = entry
            booked = []          # (fraction, price) parts already sold
            left = 1.0
            for ts, row in path.iterrows():
                if c["source"] == "Scalp" and ts.date() == c["time"].date() and \
                        (ts.hour, ts.minute) >= SCALP_EXIT:
                    booked.append((left, float(row["Close"]))); left = 0.0
                    closed, close_t, exit_px = True, ts, float(row["Close"])
                    break
                hi, lo = row["High"], row["Low"]
                best = max(best, hi) if sign > 0 else min(best, lo)
                cur_stop = stop if stage == 0 else (entry if stage == 1 else c["t1"])
                if (sign > 0 and lo <= cur_stop) or (sign < 0 and hi >= cur_stop):
                    booked.append((left, float(cur_stop))); left = 0.0
                    stop_t, closed, close_t, exit_px = ts, True, ts, float(cur_stop)
                    break
                for k in (1, 2, 3):
                    tgt = c[f"t{k}"]
                    if t_hits[k] is None and ((sign > 0 and hi >= tgt) or (sign < 0 and lo <= tgt)):
                        t_hits[k] = ts
                        stage = k
                        part = left if k == 3 else min(left, 1 / 3)
                        booked.append((part, float(tgt))); left -= part
                if stage == 3:
                    closed, close_t, exit_px = True, ts, float(c["t3"])
                    break
            # scalp calls from earlier days with no 15:15 bar still close
            if not closed and c["source"] == "Scalp" and c["time"].date() < today:
                closed, close_t = True, c["time"]
            last = float(path["Close"].iloc[-1]) if len(path) else c["entry"]
            if left > 1e-9:  # what is still held: valued at the live price (or closed at it)
                booked.append((left, last))
            avg_px = sum(f * p for f, p in booked) / max(sum(f for f, _ in booked), 1e-9)
            result = (avg_px - entry) * sign / risk
            if closed and stage >= 1:
                status = f"T{stage} hit"
            elif closed and stop_t is not None:
                status = "Stop hit"
            elif closed:
                status = "Closed (no target)"
            elif stage >= 1:
                status = f"T{stage} hit (running)"
            else:
                status = "open"
            with self.lock:
                c.update(stage=stage, t1_time=t_hits[1], t2_time=t_hits[2], t3_time=t_hits[3],
                         stop_time=stop_t, close_time=close_t, result_r=float(result), last=last,
                         avg_px=float(avg_px), exit=float(avg_px) if closed else None,
                         final_exit=exit_px, best=float(best), status=status, closed=bool(closed))
        self.updated["eval"] = _now()

    # ------------------------------------------------------------------ read helpers (UI)
    def snapshot_calls(self) -> pd.DataFrame:
        with self.lock:
            df = pd.DataFrame([dict(c) for c in self.calls])
        return df

    def price(self, s):
        return self.prices.get(s)
