"""Market data: live NSE quotes (direct from nseindia.com) with Yahoo Finance fallback,
plus historical/intraday OHLCV via yfinance."""
from __future__ import annotations

import datetime as dt
from urllib.parse import quote as urlquote

import pandas as pd
import requests
import yfinance as yf

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

BENCHMARKS = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY ENERGY": "^CNXENERGY",
    "SENSEX": "^BSESN",
}

DEFAULT_WATCHLIST = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "BHARTIARTL", "ITC", "SBIN",
    "LT", "HINDUNILVR", "KOTAKBANK", "AXISBANK", "BAJFINANCE", "MARUTI", "SUNPHARMA",
    "M&M", "TITAN", "NTPC", "ULTRACEMCO", "ASIANPAINT", "HCLTECH", "WIPRO", "POWERGRID",
]


def to_yahoo(symbol: str) -> str:
    s = symbol.strip().upper()
    if s.startswith("^") or s.endswith((".NS", ".BO")):
        return s
    return s + ".NS"


def to_nse(symbol: str) -> str:
    return symbol.strip().upper().removesuffix(".NS")


def market_status(now: dt.datetime | None = None) -> tuple[bool, str]:
    """NSE equity session: Mon–Fri 09:15–15:30 IST (exchange holidays not checked)."""
    now = now or dt.datetime.now(IST)
    if now.weekday() >= 5:
        return False, "Closed (weekend)"
    open_t, close_t = now.replace(hour=9, minute=15, second=0), now.replace(hour=15, minute=30, second=0)
    if now < open_t:
        return False, "Pre-open / closed"
    if now > close_t:
        return False, "Closed for the day"
    return True, "Market OPEN"


# ----------------------------------------------------------------------------
# NSE direct (official website JSON, needs browser-like session cookies)
# ----------------------------------------------------------------------------
class NSEClient:
    BASE = "https://www.nseindia.com"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/129.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }

    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update(self.HEADERS)
        self._warm = False
        self.blocked_until = 0.0  # after a failure NSE is skipped for a while (cloud servers are often blocked)

    def _get(self, path: str) -> dict:
        import time as _t
        if _t.time() < self.blocked_until:
            raise RuntimeError("NSE temporarily skipped")
        try:
            if not self._warm:
                self.s.get(self.BASE, timeout=4)
                self._warm = True
            r = self.s.get(self.BASE + path, timeout=4)
            if r.status_code in (401, 403):  # cookies expired -> refresh once
                self.s.get(self.BASE, timeout=4)
                r = self.s.get(self.BASE + path, timeout=4)
            r.raise_for_status()
            return r.json()
        except Exception:
            self.blocked_until = _t.time() + 600  # try NSE again in 10 minutes
            self._warm = False
            raise

    def quote(self, symbol: str) -> dict:
        j = self._get(f"/api/quote-equity?symbol={urlquote(to_nse(symbol))}")
        p = j["priceInfo"]
        return {
            "source": "NSE (live)",
            "name": j.get("info", {}).get("companyName", symbol),
            "last": float(p["lastPrice"]), "change": float(p["change"]), "pchange": float(p["pChange"]),
            "prev_close": float(p["previousClose"]), "open": float(p["open"]),
            "day_high": float(p["intraDayHighLow"]["max"]), "day_low": float(p["intraDayHighLow"]["min"]),
            "high_52w": float(p["weekHighLow"]["max"]), "low_52w": float(p["weekHighLow"]["min"]),
            "vwap": float(p.get("vwap") or 0) or None,
            "time": j.get("metadata", {}).get("lastUpdateTime", ""),
        }


def yahoo_quote(symbol: str) -> dict:
    t = yf.Ticker(to_yahoo(symbol))
    fi = t.fast_info
    last, prev = float(fi["last_price"]), float(fi["previous_close"])
    return {
        "source": "Yahoo Finance",
        "name": symbol, "last": last, "change": last - prev, "pchange": (last / prev - 1) * 100,
        "prev_close": prev, "open": float(fi["open"]), "day_high": float(fi["day_high"]),
        "day_low": float(fi["day_low"]), "high_52w": float(fi["year_high"]),
        "low_52w": float(fi["year_low"]), "vwap": None,
        "time": dt.datetime.now(IST).strftime("%d-%b-%Y %H:%M:%S"),
    }


_nse = NSEClient()


def live_quote(symbol: str) -> dict:
    """Try NSE first (true live), fall back to Yahoo (may be delayed)."""
    if not symbol.startswith("^") and not symbol.upper().endswith(".BO"):
        try:
            return _nse.quote(symbol)
        except Exception:
            pass
    return yahoo_quote(symbol)


# ----------------------------------------------------------------------------
# History
# ----------------------------------------------------------------------------
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    return df


class DataUnavailable(Exception):
    """Yahoo returned nothing (usually a short rate limit). Raised so callers don't cache the empty answer."""


def _fetch(symbol: str, period: str, interval: str) -> pd.DataFrame:
    import time as _t
    t = to_yahoo(symbol)
    for attempt in range(2):
        try:
            df = _clean(yf.Ticker(t).history(period=period, interval=interval, auto_adjust=True))
            if len(df):
                return df
        except Exception:
            pass
        try:
            raw = yf.download(t, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = _clean(raw)
            if len(df):
                return df
        except Exception:
            pass
        _t.sleep(1.5)
    raise DataUnavailable(f"No data from Yahoo for {symbol} ({period}, {interval})")


def history(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    return _fetch(symbol, period, interval)


def intraday(symbol: str, interval: str = "5m", days: int = 5) -> pd.DataFrame:
    # Yahoo limits: 1m -> 7 days, 5m/15m -> 60 days
    return _fetch(symbol, f"{days}d", interval)


def history_many(symbols: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    tick = [to_yahoo(s) for s in symbols]
    raw = yf.download(tick, period=period, interval="1d", group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    out = {}
    for s, t in zip(symbols, tick):
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = _clean(df)
            if len(df):
                out[s] = df
        except KeyError:
            continue
    return out


def fundamentals(symbol: str) -> dict:
    try:
        info = yf.Ticker(to_yahoo(symbol)).info or {}
    except Exception:
        return {}
    keys = {
        "longName": "Company", "sector": "Sector", "industry": "Industry",
        "marketCap": "Market cap", "trailingPE": "P/E (TTM)", "forwardPE": "Forward P/E",
        "priceToBook": "P/B", "trailingEps": "EPS (TTM)", "dividendYield": "Dividend yield",
        "returnOnEquity": "ROE", "debtToEquity": "Debt / Equity", "profitMargins": "Net margin",
        "revenueGrowth": "Revenue growth (YoY)", "earningsGrowth": "Earnings growth (YoY)",
        "beta": "Beta (Yahoo)", "fiftyTwoWeekHigh": "52W high", "fiftyTwoWeekLow": "52W low",
        "heldPercentInsiders": "Promoter/insider holding", "heldPercentInstitutions": "Institutional holding",
        "longBusinessSummary": "About",
    }
    return {label: info.get(k) for k, label in keys.items() if info.get(k) is not None}
