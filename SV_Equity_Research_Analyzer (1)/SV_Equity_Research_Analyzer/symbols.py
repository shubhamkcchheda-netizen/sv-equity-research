"""All NSE-listed equities for the search box (symbol + company name).

Source order:
1. NSE's official equity list (EQUITY_L.csv), downloaded and cached for 24 h
2. A local copy saved from the last successful download (nse_equity_list.csv)
3. A built-in list of major stocks
Anything still missing can be found with Yahoo Finance search (search_yahoo).
"""
from __future__ import annotations

import io
import os

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_COPY = os.path.join(HERE, "nse_equity_list.csv")
NSE_URLS = [
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/129.0 Safari/537.36",
    "Accept": "text/csv,*/*", "Referer": "https://www.nseindia.com/",
}

BUILTIN = {
    "RELIANCE": "Reliance Industries Ltd", "TCS": "Tata Consultancy Services Ltd",
    "HDFCBANK": "HDFC Bank Ltd", "ICICIBANK": "ICICI Bank Ltd", "INFY": "Infosys Ltd",
    "BHARTIARTL": "Bharti Airtel Ltd", "SBIN": "State Bank of India", "ITC": "ITC Ltd",
    "LT": "Larsen & Toubro Ltd", "HINDUNILVR": "Hindustan Unilever Ltd",
    "KOTAKBANK": "Kotak Mahindra Bank Ltd", "AXISBANK": "Axis Bank Ltd",
    "BAJFINANCE": "Bajaj Finance Ltd", "BAJAJFINSV": "Bajaj Finserv Ltd",
    "BAJAJ-AUTO": "Bajaj Auto Ltd", "MARUTI": "Maruti Suzuki India Ltd",
    "M&M": "Mahindra & Mahindra Ltd", "SUNPHARMA": "Sun Pharmaceutical Industries Ltd",
    "TITAN": "Titan Company Ltd", "NTPC": "NTPC Ltd", "POWERGRID": "Power Grid Corporation of India Ltd",
    "ULTRACEMCO": "UltraTech Cement Ltd", "ASIANPAINT": "Asian Paints Ltd", "HCLTECH": "HCL Technologies Ltd",
    "WIPRO": "Wipro Ltd", "TECHM": "Tech Mahindra Ltd", "NESTLEIND": "Nestle India Ltd",
    "ONGC": "Oil & Natural Gas Corporation Ltd", "COALINDIA": "Coal India Ltd",
    "TATASTEEL": "Tata Steel Ltd", "JSWSTEEL": "JSW Steel Ltd", "HINDALCO": "Hindalco Industries Ltd",
    "ADANIENT": "Adani Enterprises Ltd", "ADANIPORTS": "Adani Ports and Special Economic Zone Ltd",
    "ADANIGREEN": "Adani Green Energy Ltd", "ADANIPOWER": "Adani Power Ltd",
    "GRASIM": "Grasim Industries Ltd", "CIPLA": "Cipla Ltd", "DRREDDY": "Dr. Reddy's Laboratories Ltd",
    "DIVISLAB": "Divi's Laboratories Ltd", "APOLLOHOSP": "Apollo Hospitals Enterprise Ltd",
    "EICHERMOT": "Eicher Motors Ltd", "HEROMOTOCO": "Hero MotoCorp Ltd", "TVSMOTOR": "TVS Motor Company Ltd",
    "BRITANNIA": "Britannia Industries Ltd", "TATACONSUM": "Tata Consumer Products Ltd",
    "INDUSINDBK": "IndusInd Bank Ltd", "SBILIFE": "SBI Life Insurance Company Ltd",
    "HDFCLIFE": "HDFC Life Insurance Company Ltd", "SHRIRAMFIN": "Shriram Finance Ltd",
    "BEL": "Bharat Electronics Ltd", "HAL": "Hindustan Aeronautics Ltd", "TRENT": "Trent Ltd",
    "BPCL": "Bharat Petroleum Corporation Ltd", "IOC": "Indian Oil Corporation Ltd",
    "GAIL": "GAIL (India) Ltd", "TATAPOWER": "Tata Power Company Ltd", "DLF": "DLF Ltd",
    "PIDILITIND": "Pidilite Industries Ltd", "DMART": "Avenue Supermarts Ltd",
    "VEDL": "Vedanta Ltd", "BANKBARODA": "Bank of Baroda", "PNB": "Punjab National Bank",
    "CANBK": "Canara Bank", "IRCTC": "Indian Railway Catering & Tourism Corporation Ltd",
    "IRFC": "Indian Railway Finance Corporation Ltd", "ETERNAL": "Eternal Ltd", "PAYTM": "One 97 Communications Ltd", "NYKAA": "FSN E-Commerce Ventures Ltd",
    "POLYCAB": "Polycab India Ltd", "HAVELLS": "Havells India Ltd", "SIEMENS": "Siemens Ltd",
    "ABB": "ABB India Ltd", "LTIM": "LTIMindtree Ltd", "PERSISTENT": "Persistent Systems Ltd",
    "MPHASIS": "Mphasis Ltd", "COFORGE": "Coforge Ltd", "DABUR": "Dabur India Ltd",
    "GODREJCP": "Godrej Consumer Products Ltd", "MARICO": "Marico Ltd", "COLPAL": "Colgate-Palmolive (India) Ltd",
    "LUPIN": "Lupin Ltd", "AUROPHARMA": "Aurobindo Pharma Ltd", "TORNTPHARM": "Torrent Pharmaceuticals Ltd",
    "SUZLON": "Suzlon Energy Ltd", "YESBANK": "Yes Bank Ltd", "IDEA": "Vodafone Idea Ltd",
    "IDFCFIRSTB": "IDFC First Bank Ltd", "FEDERALBNK": "Federal Bank Ltd", "AUBANK": "AU Small Finance Bank Ltd",
    "BHEL": "Bharat Heavy Electricals Ltd", "SAIL": "Steel Authority of India Ltd", "NHPC": "NHPC Ltd",
    "RECLTD": "REC Ltd", "PFC": "Power Finance Corporation Ltd", "LICI": "Life Insurance Corporation of India",
    "JIOFIN": "Jio Financial Services Ltd", "MAZDOCK": "Mazagon Dock Shipbuilders Ltd",
    "CHOLAFIN": "Cholamandalam Investment and Finance Company Ltd", "MUTHOOTFIN": "Muthoot Finance Ltd",
    "ASHOKLEY": "Ashok Leyland Ltd", "AMBUJACEM": "Ambuja Cements Ltd", "SHREECEM": "Shree Cement Ltd",
    "INDIGO": "InterGlobe Aviation Ltd", "INDHOTEL": "The Indian Hotels Company Ltd",
    "TATAELXSI": "Tata Elxsi Ltd", "TATACHEM": "Tata Chemicals Ltd", "UPL": "UPL Ltd",
}


def _from_csv_text(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.rename(columns={"NAME OF COMPANY": "NAME"})
    if "SERIES" in df.columns:  # keep normal equity + BE/BZ/SM trade-for-trade/SME series
        df = df[df["SERIES"].str.strip().isin(["EQ", "BE", "BZ", "SM", "ST"])]
    return df[["SYMBOL", "NAME"]].assign(SYMBOL=lambda x: x.SYMBOL.str.strip(), NAME=lambda x: x.NAME.str.strip())


def load_all() -> tuple[pd.DataFrame, str]:
    """Returns (DataFrame[SYMBOL, NAME], source description)."""
    for url in NSE_URLS:
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.ok and "SYMBOL" in r.text[:200].upper():
                df = _from_csv_text(r.text)
                if len(df) > 500:
                    try:
                        df.to_csv(LOCAL_COPY, index=False)
                    except OSError:
                        pass
                    return df.sort_values("SYMBOL").reset_index(drop=True), f"NSE official list · {len(df):,} stocks"
        except Exception:
            continue
    if os.path.exists(LOCAL_COPY):
        try:
            df = pd.read_csv(LOCAL_COPY)
            return df, f"Saved NSE list · {len(df):,} stocks"
        except Exception:
            pass
    df = pd.DataFrame(sorted(BUILTIN.items()), columns=["SYMBOL", "NAME"])
    return df, f"Built-in list · {len(df)} stocks (NSE list unreachable — use the search below for others)"


def search_yahoo(query: str, limit: int = 10) -> list[tuple[str, str]]:
    """Live name/symbol search via Yahoo Finance, NSE results only."""
    try:
        import yfinance as yf
        quotes = yf.Search(query, max_results=25).quotes
    except Exception:
        return []
    out = []
    for q in quotes:
        sym = q.get("symbol", "")
        if sym.endswith(".NS"):
            out.append((sym[:-3], q.get("longname") or q.get("shortname") or sym))
    return out[:limit]
