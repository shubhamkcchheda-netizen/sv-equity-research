# SV Equity Research Analyzer

Live NSE stock tracking with buy/sell signals, stop-loss & targets, and comparison against NIFTY 50, sector indices and peer stocks.

## Get the app on your phone and PC

### Option A — Free online link (works anywhere, phone + PC)
1. Make a free account at **github.com**, click **New repository**, name it `sv-equity-research`, keep it Public, click **Create**.
2. On the new repo page click **uploading an existing file**, drag in **all files from this folder** (including the `.streamlit` folder), click **Commit changes**.
3. Go to **share.streamlit.io**, sign in with GitHub, click **Create app → Deploy a public app from GitHub**.
4. Pick repo `sv-equity-research`, branch `main`, main file `app.py`, click **Deploy**. In 2–3 minutes you get a link like `https://sv-equity-research.streamlit.app`.
5. **On your phone:** open the link in Chrome → ⋮ menu → **Add to Home screen** (iPhone: Safari → Share → **Add to Home Screen**). It now opens like an app.

Note: on the free cloud, NSE often blocks direct requests from cloud servers, so the app uses Yahoo Finance prices there (usually seconds behind). Running on your own PC (Option B) can use NSE directly.

### Option B — Run on your PC and open on your phone (same Wi-Fi)
1. Run it on the PC as below. The window shows a **Network URL** like `http://192.168.1.5:8501`.
2. Open that address in your phone's browser while on the same Wi-Fi, then **Add to Home screen**.

## Run it on a PC

1. Install Python 3.10+ from python.org (tick "Add to PATH" on Windows).
2. Unzip this folder, then:
   - **Windows:** double-click `run_windows.bat`
   - **Mac / Linux:** `./run_mac_linux.sh`
   - or manually: `pip install -r requirements.txt` then `streamlit run app.py`
3. The app opens at http://localhost:8501

## What's inside

| Tab | What it does |
|---|---|
| **🔍 Search** | Type any part of a name or symbol ("tata", "hdfc", "infy") and pick from every NSE-listed stock. The list is downloaded from NSE daily; a backup search finds anything missing. |
| **Header (live)** | Live price straight from nseindia.com (falls back to Yahoo Finance), day/52-week range, NIFTY level. Auto-refreshes every 1–60 s. |
| **Real-time updates** | During market hours today's candle is rebuilt from every live tick, so the main signal, confidence, key levels, targets/stop, chart, patterns, intraday and scalping all update every 1–2 s (each section shows its last update time). Breakouts re-scan every minute. Charts keep your zoom while updating. After 3:30 pm it slows to once a minute. |
| **🎯 Signal & Chart** | STRONG BUY → STRONG SELL verdict from a 9-factor score (−100…+100), buy zone, ATR-based stop-loss, 1.5R / 3R targets, position size from your capital & risk %, plus a "why" table. Candlestick chart with SMA 20/50/200, Bollinger, Supertrend, support/resistance, buy/sell markers, volume, RSI, MACD. |
| **⏳ How long to hold** | On the Signal tab: how many trading days this stock has usually taken to reach Target 1, 2 and 3 after the same kind of signal (with the expected date and how often each target was reached), plus a time stop: sell if Target 1 hasn't come within the usual time. |
| **💼 My Positions** | Add stocks you bought (price, quantity, date). Each one shows a live action — HOLD, BOOK PARTIAL PROFIT, BOOK MORE PROFIT, BOOK FULL PROFIT, SELL, SELL NOW (stop hit) or SELL (time stop) — with the reason, live P&L, a trailing stop that moves up as targets are hit, targets ticked off, and days held vs the usual holding time. Saved on the device running the app. |
| **🚀 Breakouts** | Automatically scans stocks (major ~110, your watchlist, or all NSE) every 5 minutes and lists fresh breakouts with a 🟢 BUY or 🔴 SELL tag: resistance/support breaks (levels touched 2+ times), 52-week high/low, 20-day range, Bollinger squeeze and big gaps. Breakouts must come with above-average volume (except 52-week levels). MAJOR = heavy volume plus several confirmations. Shows level broken, volume, stop-loss and target, and opens any stock in full analysis. |
| **🕯️ Patterns** | Reads 32 candlestick patterns (engulfing, harami, harami cross, hammer, shooting star, doji types, morning/evening star, three soldiers/crows, piercing, dark cloud, tweezers, marubozu, kicker, three inside/outside, Elephant Bar…), chart patterns (head & shoulders, inverse H&S, double/triple tops & bottoms, triangles, wedges, channels, rectangles, bull/bear flags) and harmonic patterns (Gartley, Bat, Alt Bat, Butterfly, Crab, Deep Crab, Cypher, Shark, AB=CD, plus *potential* ones with the D zone). Draws them on the chart with trigger, stop and target, and scans the market for confirmed patterns. Works on daily, 15-min and 5-min candles. |
| **⚡ Scalping** | Live scalp BUY/SELL on 1/3/5-min candles (VWAP, EMA 9/21, RSI 7, volume burst, 15-bar breakout, candle close), with stop and 3 targets, and an honest record of how the last 5 days of calls did (did T1 come before the stop?). |
| **⏱️ Intraday Live** | Same engine on 1m/5m/15m candles with VWAP, auto-refreshing during market hours. |
| **⚖️ vs NIFTY & Peers** | Rebased performance chart, beta, correlation, alpha, up/down capture, Sharpe, max drawdown, returns (1W…3Y, YTD), relative-strength line, rolling beta, correlation heatmap. |
| **🔎 Scanner** | Ranks a whole watchlist by signal confidence; re-scans automatically every minute while the market is open. |
| **🧪 Backtest** | Tests the signal rules on past data vs buy-and-hold, with costs. |
| **🏢 Fundamentals** | P/E, P/B, ROE, market cap, margins, holdings. |

## Signal factors (weights)

Price vs 200-SMA (15) · 50/200 cross (10) · EMA 9/21 (10) · MACD (10) · RSI (15) · Bollinger %B (10) · Supertrend (15) · ADX/DI (10) · Volume spike (5).
Score ≥ 40 Strong Buy, ≥ 15 Buy, ≤ −15 Sell, ≤ −40 Strong Sell. Edit `WEIGHTS` and `component_scores()` in `analysis.py` to tune.

## Data notes

- Live quotes use NSE's public website API. NSE sometimes blocks automated requests; the app then silently uses Yahoo Finance (shown in the header as the source), which can be delayed a little.
- History and intraday candles come from Yahoo Finance (`SYMBOL.NS`). 1-minute candles go back 7 days.
- Exchange holidays aren't checked in the market-open indicator.
- For true tick-by-tick streaming, swap `live_quote()` in `data.py` for a broker API (Zerodha Kite Connect, Upstox, Angel One SmartAPI, Dhan) using your API key.

> ⚠️ Educational tool, not investment advice. Rule-based signals are often wrong — always use a stop-loss and size positions sensibly.
