"""
Parakramee Intelligence — Backend v3.2
Bulletproof NaN handling for all float values
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import httpx, os, math

app = FastAPI(title="Parakramee Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_IDS  = os.getenv("TELEGRAM_CHAT_IDS", "")

# ── Safe number converter — kills ALL NaN/Inf problems ───────────────────────
def safe_float(v, default=None):
    """Convert any value to a JSON-safe float or return default."""
    try:
        if v is None: return default
        f = float(v)
        if math.isnan(f) or math.isinf(f): return default
        return f
    except: return default

def safe_round(v, d=2, default=None):
    f = safe_float(v, default)
    if f is None: return default
    return round(f, d)

def series_to_list(s, decimals=4):
    """Convert pandas Series to JSON-safe list."""
    result = []
    for v in s:
        f = safe_float(v)
        result.append(round(f, decimals) if f is not None else None)
    return result

# ── Telegram ──────────────────────────────────────────────────────────────────
async def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS: return
    ids = [c.strip() for c in TELEGRAM_CHAT_IDS.split(",") if c.strip()]
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        for chat_id in ids:
            try:
                await client.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
            except: pass

# ── DAX Stocks ────────────────────────────────────────────────────────────────
DAX_STOCKS = [
    {"symbol": "SAP.DE",  "name": "SAP SE",            "sector": "Technology"},
    {"symbol": "SIE.DE",  "name": "Siemens AG",         "sector": "Industrials"},
    {"symbol": "ALV.DE",  "name": "Allianz SE",         "sector": "Financials"},
    {"symbol": "BMW.DE",  "name": "BMW AG",             "sector": "Automotive"},
    {"symbol": "VOW3.DE", "name": "Volkswagen AG",      "sector": "Automotive"},
    {"symbol": "BAYN.DE", "name": "Bayer AG",           "sector": "Healthcare"},
    {"symbol": "BAS.DE",  "name": "BASF SE",            "sector": "Materials"},
    {"symbol": "MUV2.DE", "name": "Munich Re",          "sector": "Financials"},
    {"symbol": "DBK.DE",  "name": "Deutsche Bank",      "sector": "Financials"},
    {"symbol": "DTE.DE",  "name": "Deutsche Telekom",   "sector": "Telecom"},
    {"symbol": "MBG.DE",  "name": "Mercedes-Benz",      "sector": "Automotive"},
    {"symbol": "EOAN.DE", "name": "E.ON SE",            "sector": "Utilities"},
    {"symbol": "ADS.DE",  "name": "Adidas AG",          "sector": "Consumer"},
    {"symbol": "RWE.DE",  "name": "RWE AG",             "sector": "Utilities"},
    {"symbol": "LIN.DE",  "name": "Linde PLC",          "sector": "Materials"},
    {"symbol": "BEI.DE",  "name": "Beiersdorf AG",      "sector": "Consumer"},
    {"symbol": "MTX.DE",  "name": "MTU Aero Engines",   "sector": "Industrials"},
    {"symbol": "VNA.DE",  "name": "Vonovia SE",         "sector": "Real Estate"},
    {"symbol": "ZAL.DE",  "name": "Zalando SE",         "sector": "Consumer"},
    {"symbol": "HEI.DE",  "name": "HeidelbergMaterials","sector": "Materials"},
]

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_rsi(closes, period=14):
    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(closes):
    ema12  = closes.ewm(span=12, adjust=False).mean()
    ema26  = closes.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

def calc_bollinger(closes, period=20, std=2.0):
    sma   = closes.rolling(period).mean()
    sigma = closes.rolling(period).std()
    return sma + std * sigma, sma, sma - std * sigma

def calc_atr(df, period=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_candle(df):
    if len(df) < 3: return None
    c0, c1, c2 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    body  = abs(c2["Close"] - c2["Open"])
    rng   = c2["High"] - c2["Low"]
    upper = c2["High"] - max(c2["Open"], c2["Close"])
    lower = min(c2["Open"], c2["Close"]) - c2["Low"]
    if rng > 0 and body < rng * 0.1:
        return {"name": "Doji", "signal": "neutral", "desc": "Market indecision"}
    if lower > body * 2 and upper < body * 0.5 and c2["Close"] > c2["Open"]:
        return {"name": "Hammer", "signal": "bullish", "desc": "Bullish reversal at support"}
    if upper > body * 2 and lower < body * 0.5 and c2["Close"] < c2["Open"]:
        return {"name": "Shooting Star", "signal": "bearish", "desc": "Bearish reversal"}
    if (c1["Close"] < c1["Open"] and c2["Close"] > c2["Open"] and
            c2["Close"] > c1["Open"] and c2["Open"] < c1["Close"]):
        return {"name": "Bullish Engulfing", "signal": "bullish", "desc": "Bulls take control"}
    if (c1["Close"] > c1["Open"] and c2["Close"] < c2["Open"] and
            c2["Close"] < c1["Open"] and c2["Open"] > c1["Close"]):
        return {"name": "Bearish Engulfing", "signal": "bearish", "desc": "Bears take control"}
    if (c0["Close"] < c0["Open"] and c1["Close"] < c1["Open"] and
            c2["Close"] > c2["Open"] and c2["Close"] > c1["Open"]):
        return {"name": "Morning Star", "signal": "bullish", "desc": "3-candle bullish reversal"}
    if (c0["Close"] > c0["Open"] and c1["Close"] > c1["Open"] and
            c2["Close"] < c2["Open"] and c2["Close"] < c1["Open"]):
        return {"name": "Evening Star", "signal": "bearish", "desc": "3-candle bearish reversal"}
    return None

def calc_levels(df, price, bb_u, bb_m, bb_l, sma50, atr):
    """Calculate price targets and stop loss — all NaN-safe."""
    p = safe_float(price, 100)
    a = safe_float(atr, p * 0.02)
    stop  = round(p - 1.5 * a, 2)
    t1    = safe_round(bb_m, 2) if safe_float(bb_m, 0) > p else round(p * 1.04, 2)
    t2_bb = safe_round(bb_u, 2) if safe_float(bb_u, 0) > t1 else None
    t2_sm = safe_round(sma50, 2) if safe_float(sma50, 0) > t1 else None
    t2    = max(t2_bb, t2_sm) if t2_bb and t2_sm else (t2_bb or t2_sm or round(p * 1.08, 2))
    # 52-week high safe calculation
    try:
        h52 = safe_float(df["High"].rolling(min(252, len(df))).max().iloc[-1], p * 1.12)
    except:
        h52 = p * 1.12
    t3   = round(max(h52, p * 1.12), 2)
    risk = p - stop
    rr   = round((t2 - p) / risk, 2) if risk > 0 else 0
    bl   = safe_round(bb_l, 2)
    return {
        "stop_loss":     round(stop, 2),
        "target1":       round(t1, 2),
        "target2":       round(t2, 2),
        "target3":       round(t3, 2),
        "buy_zone_low":  round(bl * 0.99, 2) if bl else round(p * 0.97, 2),
        "buy_zone_high": round(bl * 1.01, 2) if bl else round(p * 1.01, 2),
        "risk_reward":   round(rr, 2),
        "risk_pct":      round((p - stop) / p * 100, 2),
        "reward_pct":    round((t2 - p) / p * 100, 2),
    }

def detect_sells(df, rsi_s, macd_s, sig_s, bb_u, bb_l, sma20, sma50):
    sells = []
    n = len(df) - 1
    if n < 2: return sells
    def sv(s, i):
        try: return safe_float(s.iloc[i])
        except: return None
    rsi  = sv(rsi_s, n) or 50
    macd = sv(macd_s, n) or 0;  pmacd = sv(macd_s, n-1) or 0
    sig  = sv(sig_s, n) or 0;   psig  = sv(sig_s, n-1) or 0
    price = safe_float(df["Close"].iloc[n], 0)
    s20 = sv(sma20, n); ps20 = sv(sma20, n-1)
    s50 = sv(sma50, n); ps50 = sv(sma50, n-1)
    bbu = sv(bb_u, n)
    if rsi > 70:
        sells.append({"type":"RSI Overbought","strength":"strong" if rsi>75 else "moderate","detail":f"RSI at {rsi:.1f}","urgency":"high"})
    if pmacd > psig and macd < sig:
        sells.append({"type":"MACD Bearish Cross","strength":"strong","detail":"MACD crossed below signal","urgency":"high"})
    if bbu and price >= bbu * 0.99:
        sells.append({"type":"Upper Bollinger Band","strength":"moderate","detail":f"Price at upper BB €{bbu:.2f}","urgency":"medium"})
    if all(v is not None for v in [s20, s50, ps20, ps50]):
        if ps20 >= ps50 and s20 < s50:
            sells.append({"type":"Death Cross","strength":"strong","detail":"SMA20 below SMA50","urgency":"high"})
    pat = detect_candle(df.iloc[-3:])
    if pat and pat["signal"] == "bearish":
        sells.append({"type":f"Candle: {pat['name']}","strength":"strong","detail":pat["desc"],"urgency":"high"})
    return sells

def score_stock(df):
    closes = df["Close"]
    rsi_s  = calc_rsi(closes)
    macd_s, sig_s, _ = calc_macd(closes)
    bb_u, bb_m, bb_l = calc_bollinger(closes)
    sma20 = closes.rolling(20).mean()
    sma50 = closes.rolling(50).mean()
    ema9  = closes.ewm(span=9, adjust=False).mean()
    atr   = calc_atr(df)
    pat   = detect_candle(df)

    def sv(s):
        try: return safe_float(s.iloc[-1])
        except: return None

    rsi  = sv(rsi_s) or 50
    macd = sv(macd_s) or 0
    sig  = sv(sig_s) or 0
    bbu  = sv(bb_u); bbm = sv(bb_m); bbl = sv(bb_l)
    s20  = sv(sma20); s50 = sv(sma50); e9 = sv(ema9)
    atr_v = sv(atr)
    price = safe_float(closes.iloc[-1], 0)
    prev  = safe_float(closes.iloc[-2], price) if len(closes) > 1 else price
    p5d   = safe_float(closes.iloc[-6], price) if len(closes) > 5 else price

    score = 0; signals = []

    if rsi < 30:   score += 30; signals.append({"label":"RSI Oversold","value":f"RSI:{rsi:.1f}","type":"buy"})
    elif rsi < 45: score += 15; signals.append({"label":"RSI Low","value":f"RSI:{rsi:.1f}","type":"buy"})
    elif rsi > 70: score -= 25; signals.append({"label":"RSI Overbought","value":f"RSI:{rsi:.1f}","type":"sell"})
    else:          signals.append({"label":"RSI Neutral","value":f"RSI:{rsi:.1f}","type":"neutral"})

    if macd > sig: score += 20; signals.append({"label":"MACD Bullish","value":f"{macd:.3f}","type":"buy"})
    else:          score -= 10; signals.append({"label":"MACD Bearish","value":f"{macd:.3f}","type":"sell"})

    if bbl and price < bbl:   score += 20; signals.append({"label":"Below Lower BB","value":f"€{price:.2f}","type":"buy"})
    elif bbu and price > bbu: score -= 15; signals.append({"label":"Above Upper BB","value":f"€{price:.2f}","type":"sell"})
    else: signals.append({"label":"Inside BB","value":"Normal range","type":"neutral"})

    if s20 and s50:
        if s20 > s50: score += 15; signals.append({"label":"Uptrend","value":"SMA20>SMA50","type":"buy"})
        else:         score -= 10; signals.append({"label":"Downtrend","value":"SMA20<SMA50","type":"sell"})

    if e9 and price > e9: score += 10; signals.append({"label":"Price>EMA9","value":"Momentum+","type":"buy"})

    if pat:
        if pat["signal"] == "bullish": score += 15; signals.append({"label":pat["name"],"value":pat["desc"],"type":"buy"})
        elif pat["signal"] == "bearish": score -= 10; signals.append({"label":pat["name"],"value":pat["desc"],"type":"sell"})

    avg_vol  = safe_float(df["Volume"].iloc[-10:].mean(), 0)
    last_vol = safe_float(df["Volume"].iloc[-1], 0)
    if avg_vol > 0 and last_vol > avg_vol * 1.5:
        score += 10; signals.append({"label":"Volume Spike","value":f"+{((last_vol/avg_vol)-1)*100:.0f}%","type":"buy"})

    rec = ("STRONG BUY" if score>=50 else "BUY" if score>=25 else
           "HOLD" if score>=0 else "CAUTION" if score>=-20 else "AVOID")
    c1d = ((price-prev)/prev*100) if prev else 0
    c5d = ((price-p5d)/p5d*100) if p5d else 0

    sells = detect_sells(df, rsi_s, macd_s, sig_s, bb_u, bb_l, sma20, sma50)
    ss = len([s for s in sells if s["urgency"]=="high"])*2 + len([s for s in sells if s["urgency"]=="medium"])
    sell_r = ("STRONG SELL" if ss>=4 else "SELL" if ss>=2 else "WATCH" if ss>=1 else "HOLD")

    levels = calc_levels(df, price, bbu, bbm, bbl, s50, atr_v)

    return {
        "score": score, "recommendation": rec,
        "signals": signals, "sell_signals": sells, "sell_rating": sell_r,
        "rsi": safe_round(rsi, 2, 50),
        "macd": safe_round(macd, 4, 0),
        "macd_signal": safe_round(sig, 4, 0),
        "price": safe_round(price, 2, 0),
        "change1d": safe_round(c1d, 2, 0),
        "change5d": safe_round(c5d, 2, 0),
        "pattern": pat,
        "atr": safe_round(atr_v, 2),
        "bb_upper": safe_round(bbu, 2),
        "bb_lower": safe_round(bbl, 2),
        "bb_mid": safe_round(bbm, 2),
        "sma20": safe_round(s20, 2),
        "sma50": safe_round(s50, 2),
        "ema9": safe_round(e9, 2),
        "volume": int(last_vol),
        "avg_volume": int(avg_vol),
        **levels
    }

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: dict = {}
_cache_ts: dict = {}
_prev: dict = {}
CACHE_TTL = 300

def is_stale(sym):
    return sym not in _cache_ts or (datetime.utcnow() - _cache_ts[sym]).seconds > CACHE_TTL

def fetch(sym):
    df = yf.Ticker(sym).history(period="6mo", interval="1d")
    if df.empty: raise ValueError(f"No data for {sym}")
    _cache[sym] = df; _cache_ts[sym] = datetime.utcnow()
    return df

async def alert(sym, name, a):
    prev = _prev.get(sym, {})
    if a["recommendation"] in ("BUY","STRONG BUY") and prev.get("rec") not in ("BUY","STRONG BUY"):
        await send_telegram(f"🟢 <b>{a['recommendation']} — {name}</b>\n💰 €{a['price']} | Score: {a['score']}/100\n🎯 T1: €{a['target1']} T2: €{a['target2']}\n🛑 SL: €{a['stop_loss']}\n<i>Parakramee Intelligence</i>")
    if a["sell_rating"] in ("SELL","STRONG SELL") and prev.get("sell") not in ("SELL","STRONG SELL"):
        await send_telegram(f"🔴 <b>{a['sell_rating']} — {name}</b>\n💰 €{a['price']}\n<i>Parakramee Intelligence</i>")
    _prev[sym] = {"rec": a["recommendation"], "sell": a["sell_rating"]}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root(): return {"status": "ok", "message": "Parakramee Intelligence API running"}

@app.get("/ping")
def ping(): return {"pong": True, "time": datetime.utcnow().isoformat()}

@app.get("/health")
def health(): return {"status": "ok", "cached": list(_cache.keys()), "time": datetime.utcnow().isoformat()}

@app.get("/stocks")
async def get_stocks(background_tasks: BackgroundTasks):
    results = []
    for s in DAX_STOCKS:
        sym = s["symbol"]
        try:
            df = fetch(sym) if is_stale(sym) else _cache[sym]
            a  = score_stock(df)
            background_tasks.add_task(alert, sym, s["name"], a)
            results.append({**s, **a})
        except Exception as e:
            print(f"Error {sym}: {e}")
            results.append({**s, "price": 0, "score": 0, "recommendation": "N/A",
                           "sell_signals": [], "sell_rating": "N/A", "error": str(e)})
    return results

@app.get("/stock/{symbol}")
async def get_stock(symbol: str, background_tasks: BackgroundTasks):
    sym = symbol.upper()
    try:
        df     = fetch(sym)
        closes = df["Close"]
        rsi_s  = calc_rsi(closes)
        macd_s, sig_s, hist_s = calc_macd(closes)
        bb_u, bb_m, bb_l = calc_bollinger(closes)
        sma20  = closes.rolling(20).mean()
        sma50  = closes.rolling(50).mean()
        ema9   = closes.ewm(span=9, adjust=False).mean()

        ohlcv = []
        for i, r in df.iterrows():
            ohlcv.append({
                "date":   str(i.date()),
                "open":   safe_round(r["Open"], 2, 0),
                "high":   safe_round(r["High"], 2, 0),
                "low":    safe_round(r["Low"],  2, 0),
                "close":  safe_round(r["Close"],2, 0),
                "volume": int(safe_float(r["Volume"], 0)),
            })

        indicators = {
            "rsi":         series_to_list(rsi_s),
            "macd":        series_to_list(macd_s),
            "macd_signal": series_to_list(sig_s),
            "macd_hist":   series_to_list(hist_s),
            "bb_upper":    series_to_list(bb_u),
            "bb_mid":      series_to_list(bb_m),
            "bb_lower":    series_to_list(bb_l),
            "sma20":       series_to_list(sma20),
            "sma50":       series_to_list(sma50),
            "ema9":        series_to_list(ema9),
        }

        a = score_stock(df)
        info = {}
        try:
            raw = yf.Ticker(sym).info
            info = {
                "longName":      raw.get("longName", sym),
                "sector":        raw.get("sector", ""),
                "industry":      raw.get("industry", ""),
                "marketCap":     safe_float(raw.get("marketCap")),
                "trailingPE":    safe_float(raw.get("trailingPE")),
                "forwardPE":     safe_float(raw.get("forwardPE")),
                "dividendYield": safe_float(raw.get("dividendYield")),
                "52wHigh":       safe_float(raw.get("fiftyTwoWeekHigh")),
                "52wLow":        safe_float(raw.get("fiftyTwoWeekLow")),
                "beta":          safe_float(raw.get("beta")),
            }
        except: pass

        sm = next((s for s in DAX_STOCKS if s["symbol"]==sym), {"symbol":sym,"name":sym,"sector":""})
        background_tasks.add_task(alert, sym, sm["name"], a)
        return {"symbol": sym, "ohlcv": ohlcv, "indicators": indicators, "analysis": a, "info": info}

    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/telegram/test")
async def test_telegram():
    await send_telegram("✅ <b>Parakramee Intelligence</b>\nTelegram alerts working!\n<i>Trade with Courage. Win with Intelligence.</i>")
    return {"status": "sent"}
