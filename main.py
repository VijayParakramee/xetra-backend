"""
XETRA Intelligence Dashboard — FastAPI Backend
Fetches real German stock data from Yahoo Finance (yfinance)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

app = FastAPI(title="XETRA Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DAX 40 Stock Universe ─────────────────────────────────────────────────────
DAX_STOCKS = [
    {"symbol": "SAP.DE",   "name": "SAP SE",             "sector": "Technology"},
    {"symbol": "SIE.DE",   "name": "Siemens AG",          "sector": "Industrials"},
    {"symbol": "ALV.DE",   "name": "Allianz SE",           "sector": "Financials"},
    {"symbol": "BMW.DE",   "name": "BMW AG",               "sector": "Automotive"},
    {"symbol": "VOW3.DE",  "name": "Volkswagen AG",        "sector": "Automotive"},
    {"symbol": "BAYN.DE",  "name": "Bayer AG",             "sector": "Healthcare"},
    {"symbol": "BAS.DE",   "name": "BASF SE",              "sector": "Materials"},
    {"symbol": "MUV2.DE",  "name": "Munich Re",            "sector": "Financials"},
    {"symbol": "DBK.DE",   "name": "Deutsche Bank",        "sector": "Financials"},
    {"symbol": "DTE.DE",   "name": "Deutsche Telekom",     "sector": "Telecom"},
    {"symbol": "MBG.DE",   "name": "Mercedes-Benz",        "sector": "Automotive"},
    {"symbol": "EOAN.DE",  "name": "E.ON SE",              "sector": "Utilities"},
    {"symbol": "ADS.DE",   "name": "Adidas AG",            "sector": "Consumer"},
    {"symbol": "RWE.DE",   "name": "RWE AG",               "sector": "Utilities"},
    {"symbol": "LIN.DE",   "name": "Linde PLC",            "sector": "Materials"},
    {"symbol": "BEI.DE",   "name": "Beiersdorf AG",        "sector": "Consumer"},
    {"symbol": "HEI.DE",   "name": "HeidelbergMaterials",  "sector": "Materials"},
    {"symbol": "MTX.DE",   "name": "MTU Aero Engines",     "sector": "Industrials"},
    {"symbol": "VNA.DE",   "name": "Vonovia SE",           "sector": "Real Estate"},
    {"symbol": "ZAL.DE",   "name": "Zalando SE",           "sector": "Consumer"},
]

# ── Technical Indicator Calculations ─────────────────────────────────────────
def calc_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(closes: pd.Series):
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return macd, signal, hist

def calc_bollinger(closes: pd.Series, period: int = 20, std: float = 2.0):
    sma   = closes.rolling(period).mean()
    sigma = closes.rolling(period).std()
    return sma + std * sigma, sma, sma - std * sigma

def detect_candle_pattern(df: pd.DataFrame) -> dict | None:
    if len(df) < 3:
        return None
    c0, c1, c2 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    body   = abs(c2["Close"] - c2["Open"])
    rng    = c2["High"] - c2["Low"]
    upper  = c2["High"] - max(c2["Open"], c2["Close"])
    lower  = min(c2["Open"], c2["Close"]) - c2["Low"]

    if rng > 0 and body < rng * 0.1:
        return {"name": "Doji", "signal": "neutral", "desc": "Market indecision — possible reversal"}
    if lower > body * 2 and upper < body * 0.5 and c2["Close"] > c2["Open"]:
        return {"name": "Hammer", "signal": "bullish", "desc": "Bullish reversal at support"}
    if upper > body * 2 and lower < body * 0.5 and c2["Close"] < c2["Open"]:
        return {"name": "Shooting Star", "signal": "bearish", "desc": "Bearish reversal at resistance"}
    if (c1["Close"] < c1["Open"] and c2["Close"] > c2["Open"]
            and c2["Close"] > c1["Open"] and c2["Open"] < c1["Close"]):
        return {"name": "Bullish Engulfing", "signal": "bullish", "desc": "Strong buy — bulls take control"}
    if (c1["Close"] > c1["Open"] and c2["Close"] < c2["Open"]
            and c2["Close"] < c1["Open"] and c2["Open"] > c1["Close"]):
        return {"name": "Bearish Engulfing", "signal": "bearish", "desc": "Strong sell — bears take control"}
    if (c0["Close"] < c0["Open"] and c1["Close"] < c1["Open"]
            and c2["Close"] > c2["Open"] and c2["Close"] > c1["Open"]):
        return {"name": "Morning Star", "signal": "bullish", "desc": "3-candle bullish reversal pattern"}
    return None

def score_stock(df: pd.DataFrame) -> dict:
    closes = df["Close"]
    rsi    = calc_rsi(closes)
    macd, sig, hist = calc_macd(closes)
    upper_bb, mid_bb, lower_bb = calc_bollinger(closes)
    sma20 = closes.rolling(20).mean()
    sma50 = closes.rolling(50).mean()
    ema9  = closes.ewm(span=9, adjust=False).mean()
    pattern = detect_candle_pattern(df)

    last_rsi    = float(rsi.iloc[-1])   if not pd.isna(rsi.iloc[-1])   else 50.0
    last_macd   = float(macd.iloc[-1])  if not pd.isna(macd.iloc[-1])  else 0.0
    last_sig    = float(sig.iloc[-1])   if not pd.isna(sig.iloc[-1])   else 0.0
    last_upper  = float(upper_bb.iloc[-1]) if not pd.isna(upper_bb.iloc[-1]) else None
    last_lower  = float(lower_bb.iloc[-1]) if not pd.isna(lower_bb.iloc[-1]) else None
    last_mid    = float(mid_bb.iloc[-1])   if not pd.isna(mid_bb.iloc[-1])   else None
    last_sma20  = float(sma20.iloc[-1])  if not pd.isna(sma20.iloc[-1])  else None
    last_sma50  = float(sma50.iloc[-1])  if not pd.isna(sma50.iloc[-1])  else None
    last_ema9   = float(ema9.iloc[-1])   if not pd.isna(ema9.iloc[-1])   else None
    price       = float(closes.iloc[-1])
    prev_price  = float(closes.iloc[-2]) if len(closes) > 1 else price
    price_5d    = float(closes.iloc[-6]) if len(closes) > 5 else price

    score   = 0
    signals = []

    # RSI
    if last_rsi < 30:
        score += 30
        signals.append({"label": "RSI Oversold", "value": f"RSI: {last_rsi:.1f}", "type": "buy"})
    elif last_rsi < 45:
        score += 15
        signals.append({"label": "RSI Low", "value": f"RSI: {last_rsi:.1f}", "type": "buy"})
    elif last_rsi > 70:
        score -= 25
        signals.append({"label": "RSI Overbought", "value": f"RSI: {last_rsi:.1f}", "type": "sell"})
    else:
        signals.append({"label": "RSI Neutral", "value": f"RSI: {last_rsi:.1f}", "type": "neutral"})

    # MACD
    if last_macd > last_sig:
        score += 20
        signals.append({"label": "MACD Bullish Cross", "value": f"MACD: {last_macd:.3f}", "type": "buy"})
    else:
        score -= 10
        signals.append({"label": "MACD Bearish", "value": f"MACD: {last_macd:.3f}", "type": "sell"})

    # Bollinger Bands
    if last_lower and price < last_lower:
        score += 20
        signals.append({"label": "Below Lower BB", "value": f"€{price:.2f} < €{last_lower:.2f}", "type": "buy"})
    elif last_upper and price > last_upper:
        score -= 15
        signals.append({"label": "Above Upper BB", "value": f"€{price:.2f} > €{last_upper:.2f}", "type": "sell"})
    else:
        signals.append({"label": "Inside Bollinger Band", "value": "Normal range", "type": "neutral"})

    # SMA trend
    if last_sma20 and last_sma50:
        if last_sma20 > last_sma50:
            score += 15
            signals.append({"label": "SMA20 > SMA50", "value": "Uptrend confirmed", "type": "buy"})
        else:
            score -= 10
            signals.append({"label": "SMA20 < SMA50", "value": "Downtrend", "type": "sell"})

    # EMA9 momentum
    if last_ema9 and price > last_ema9:
        score += 10
        signals.append({"label": "Price > EMA9", "value": "Short-term momentum +", "type": "buy"})

    # Candlestick
    if pattern:
        if pattern["signal"] == "bullish":
            score += 15
            signals.append({"label": pattern["name"], "value": pattern["desc"], "type": "buy"})
        elif pattern["signal"] == "bearish":
            score -= 10
            signals.append({"label": pattern["name"], "value": pattern["desc"], "type": "sell"})

    # Volume spike
    avg_vol = float(df["Volume"].iloc[-10:].mean())
    last_vol = float(df["Volume"].iloc[-1])
    if avg_vol > 0 and last_vol > avg_vol * 1.5:
        score += 10
        signals.append({"label": "Volume Spike", "value": f"+{((last_vol/avg_vol)-1)*100:.0f}% vs avg", "type": "buy"})

    rec = ("STRONG BUY" if score >= 50 else
           "BUY"        if score >= 25 else
           "HOLD"       if score >= 0  else
           "CAUTION"    if score >= -20 else "AVOID")

    change1d = ((price - prev_price) / prev_price * 100) if prev_price else 0
    change5d = ((price - price_5d)   / price_5d   * 100) if price_5d   else 0

    return {
        "score": score,
        "recommendation": rec,
        "signals": signals,
        "rsi": round(last_rsi, 2),
        "macd": round(last_macd, 4),
        "macd_signal": round(last_sig, 4),
        "price": round(price, 2),
        "change1d": round(change1d, 2),
        "change5d": round(change5d, 2),
        "pattern": pattern,
        "bb_upper": round(last_upper, 2) if last_upper else None,
        "bb_lower": round(last_lower, 2) if last_lower else None,
        "bb_mid":   round(last_mid, 2)   if last_mid   else None,
        "sma20": round(last_sma20, 2) if last_sma20 else None,
        "sma50": round(last_sma50, 2) if last_sma50 else None,
        "ema9":  round(last_ema9, 2)  if last_ema9  else None,
        "volume": int(last_vol),
        "avg_volume": int(avg_vol),
    }

# ── In-memory cache ───────────────────────────────────────────────────────────
_cache: dict = {}
_cache_ts: dict = {}
CACHE_TTL = 300  # 5 minutes

def is_stale(symbol: str) -> bool:
    if symbol not in _cache_ts:
        return True
    return (datetime.utcnow() - _cache_ts[symbol]).seconds > CACHE_TTL

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "XETRA Intelligence API running", "version": "2.0"}

@app.get("/stocks")
def get_all_stocks():
    """Return summary list with scores for all tracked stocks."""
    results = []
    for stock in DAX_STOCKS:
        sym = stock["symbol"]
        try:
            if is_stale(sym):
                ticker = yf.Ticker(sym)
                df = ticker.history(period="3mo", interval="1d")
                if df.empty:
                    continue
                _cache[sym] = df
                _cache_ts[sym] = datetime.utcnow()
            df = _cache[sym]
            analysis = score_stock(df)
            results.append({**stock, **analysis})
        except Exception as e:
            results.append({**stock, "error": str(e), "price": 0, "score": 0, "recommendation": "N/A"})
    return results

@app.get("/stock/{symbol}")
def get_stock_detail(symbol: str):
    """Return full OHLCV + indicators for a single stock."""
    sym = symbol.upper()
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(period="6mo", interval="1d")
        if df.empty:
            raise HTTPException(404, f"No data for {sym}")
        _cache[sym] = df
        _cache_ts[sym] = datetime.utcnow()

        # Build OHLCV array
        ohlcv = []
        for idx, row in df.iterrows():
            ohlcv.append({
                "date":   str(idx.date()),
                "open":   round(float(row["Open"]),  2),
                "high":   round(float(row["High"]),  2),
                "low":    round(float(row["Low"]),   2),
                "close":  round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })

        # Indicators for chart overlay
        closes  = df["Close"]
        rsi_s   = calc_rsi(closes)
        macd_s, sig_s, hist_s = calc_macd(closes)
        ub, mb, lb = calc_bollinger(closes)
        sma20   = closes.rolling(20).mean()
        sma50   = closes.rolling(50).mean()
        ema9    = closes.ewm(span=9, adjust=False).mean()

        def to_list(s): return [round(float(v), 4) if not pd.isna(v) else None for v in s]

        indicators = {
            "rsi":        to_list(rsi_s),
            "macd":       to_list(macd_s),
            "macd_signal":to_list(sig_s),
            "macd_hist":  to_list(hist_s),
            "bb_upper":   to_list(ub),
            "bb_mid":     to_list(mb),
            "bb_lower":   to_list(lb),
            "sma20":      to_list(sma20),
            "sma50":      to_list(sma50),
            "ema9":       to_list(ema9),
        }

        analysis = score_stock(df)

        # Fetch company info
        info = {}
        try:
            raw = ticker.info
            info = {
                "longName":        raw.get("longName", sym),
                "sector":          raw.get("sector", ""),
                "industry":        raw.get("industry", ""),
                "marketCap":       raw.get("marketCap"),
                "trailingPE":      raw.get("trailingPE"),
                "forwardPE":       raw.get("forwardPE"),
                "dividendYield":   raw.get("dividendYield"),
                "52wHigh":         raw.get("fiftyTwoWeekHigh"),
                "52wLow":          raw.get("fiftyTwoWeekLow"),
                "avgVolume":       raw.get("averageVolume"),
                "beta":            raw.get("beta"),
                "shortSummary":    raw.get("longBusinessSummary", "")[:300],
            }
        except:
            pass

        return {"symbol": sym, "ohlcv": ohlcv, "indicators": indicators, "analysis": analysis, "info": info}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
def health():
    return {"status": "ok", "cached_symbols": list(_cache.keys()), "time": datetime.utcnow().isoformat()}
