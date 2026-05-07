"""
Parakramee Intelligence Dashboard — Backend v3.1
Real German stock data + Sell Signals + Price Targets + Stop Loss + Telegram Alerts
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import httpx
import os

app = FastAPI(title="Parakramee Intelligence API v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_IDS  = os.getenv("TELEGRAM_CHAT_IDS", "")

async def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        return
    ids = [c.strip() for c in TELEGRAM_CHAT_IDS.split(",") if c.strip()]
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        for chat_id in ids:
            try:
                await client.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
            except Exception as e:
                print(f"Telegram error: {e}")

DAX_STOCKS = [
    {"symbol": "SAP.DE",   "name": "SAP SE",             "sector": "Technology"},
    {"symbol": "SIE.DE",   "name": "Siemens AG",          "sector": "Industrials"},
    {"symbol": "ALV.DE",   "name": "Allianz SE",          "sector": "Financials"},
    {"symbol": "BMW.DE",   "name": "BMW AG",              "sector": "Automotive"},
    {"symbol": "VOW3.DE",  "name": "Volkswagen AG",       "sector": "Automotive"},
    {"symbol": "BAYN.DE",  "name": "Bayer AG",            "sector": "Healthcare"},
    {"symbol": "BAS.DE",   "name": "BASF SE",             "sector": "Materials"},
    {"symbol": "MUV2.DE",  "name": "Munich Re",           "sector": "Financials"},
    {"symbol": "DBK.DE",   "name": "Deutsche Bank",       "sector": "Financials"},
    {"symbol": "DTE.DE",   "name": "Deutsche Telekom",    "sector": "Telecom"},
    {"symbol": "MBG.DE",   "name": "Mercedes-Benz",       "sector": "Automotive"},
    {"symbol": "EOAN.DE",  "name": "E.ON SE",             "sector": "Utilities"},
    {"symbol": "ADS.DE",   "name": "Adidas AG",           "sector": "Consumer"},
    {"symbol": "RWE.DE",   "name": "RWE AG",              "sector": "Utilities"},
    {"symbol": "LIN.DE",   "name": "Linde PLC",           "sector": "Materials"},
    {"symbol": "BEI.DE",   "name": "Beiersdorf AG",       "sector": "Consumer"},
    {"symbol": "MTX.DE",   "name": "MTU Aero Engines",    "sector": "Industrials"},
    {"symbol": "VNA.DE",   "name": "Vonovia SE",          "sector": "Real Estate"},
    {"symbol": "ZAL.DE",   "name": "Zalando SE",          "sector": "Consumer"},
    {"symbol": "HEI.DE",   "name": "HeidelbergMaterials", "sector": "Materials"},
]
def clean(v):
    if v is None: return None
    try: return None if (isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf'))) else v
    except: return None
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
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_candle_pattern(df):
    if len(df) < 3:
        return None
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
    if (c1["Close"] < c1["Open"] and c2["Close"] > c2["Open"] and c2["Close"] > c1["Open"] and c2["Open"] < c1["Close"]):
        return {"name": "Bullish Engulfing", "signal": "bullish", "desc": "Bulls take control"}
    if (c1["Close"] > c1["Open"] and c2["Close"] < c2["Open"] and c2["Close"] < c1["Open"] and c2["Open"] > c1["Close"]):
        return {"name": "Bearish Engulfing", "signal": "bearish", "desc": "Bears take control"}
    if (c0["Close"] < c0["Open"] and c1["Close"] < c1["Open"] and c2["Close"] > c2["Open"] and c2["Close"] > c1["Open"]):
        return {"name": "Morning Star", "signal": "bullish", "desc": "3-candle bullish reversal"}
    if (c0["Close"] > c0["Open"] and c1["Close"] > c1["Open"] and c2["Close"] < c2["Open"] and c2["Close"] < c1["Open"]):
        return {"name": "Evening Star", "signal": "bearish", "desc": "3-candle bearish reversal"}
    return None

def calc_price_levels(df, price, bb_upper, bb_mid, bb_lower, sma50, atr_val):
    stop_loss = round(price - (1.5 * atr_val), 2) if atr_val else round(price * 0.93, 2)
    t1_pct = round(price * 1.04, 2)
    t1_bb  = round(bb_mid, 2) if bb_mid and bb_mid > price else None
    target1 = t1_bb if t1_bb and t1_bb < t1_pct else t1_pct
    t2_bb  = round(bb_upper, 2) if bb_upper and bb_upper > target1 else None
    t2_sma = round(sma50, 2) if sma50 and sma50 > target1 else None
    target2 = max(t2_bb, t2_sma) if t2_bb and t2_sma else (t2_bb or t2_sma or round(price * 1.08, 2))
    high_52w = float(df["High"].rolling(252).max().iloc[-1]) if len(df) >= 50 else price * 1.12
    target3  = round(max(high_52w, price * 1.12), 2)
    risk = price - stop_loss
    reward = target2 - price
    rr = round(reward / risk, 2) if risk > 0 else 0
    return {
        "stop_loss": stop_loss, "target1": target1, "target2": target2, "target3": target3,
        "buy_zone_low": round(bb_lower * 0.99, 2) if bb_lower else round(price * 0.97, 2),
        "buy_zone_high": round(bb_lower * 1.01, 2) if bb_lower else round(price * 1.01, 2),
        "risk_reward": 0 if (rr is None or (isinstance(rr, float) and (rr != rr))) else rr,
        "risk_pct": round((price - stop_loss) / price * 100, 2),
        "reward_pct": round((target2 - price) / price * 100, 2),
    }

def detect_sell_signals(df, rsi_s, macd_s, signal_s, bb_upper, bb_lower, sma20, sma50):
    sell_signals = []
    last = len(df) - 1
    if last < 2: return sell_signals
    def sf(s, i):
        try: return float(s.iloc[i]) if not pd.isna(s.iloc[i]) else None
        except: return None
    last_rsi  = sf(rsi_s, last) or 50
    last_macd = sf(macd_s, last) or 0; prev_macd = sf(macd_s, last-1) or 0
    last_sig  = sf(signal_s, last) or 0; prev_sig = sf(signal_s, last-1) or 0
    price     = float(df["Close"].iloc[last])
    last_sma20 = sf(sma20, last); prev_sma20 = sf(sma20, last-1)
    last_sma50 = sf(sma50, last); prev_sma50 = sf(sma50, last-1)
    last_bb_u  = sf(bb_upper, last)
    if last_rsi > 70:
        sell_signals.append({"type": "RSI Overbought", "strength": "strong" if last_rsi > 75 else "moderate",
            "detail": f"RSI at {last_rsi:.1f}", "urgency": "high"})
    if prev_macd > prev_sig and last_macd < last_sig:
        sell_signals.append({"type": "MACD Bearish Cross", "strength": "strong",
            "detail": "MACD crossed below signal line", "urgency": "high"})
    if last_bb_u and price >= last_bb_u * 0.99:
        sell_signals.append({"type": "Upper Bollinger Band", "strength": "moderate",
            "detail": f"Price at upper BB €{last_bb_u:.2f}", "urgency": "medium"})
    if all(v is not None for v in [last_sma20, last_sma50, prev_sma20, prev_sma50]):
        if prev_sma20 >= prev_sma50 and last_sma20 < last_sma50:
            sell_signals.append({"type": "Death Cross", "strength": "strong",
                "detail": "SMA20 crossed below SMA50", "urgency": "high"})
    pattern = detect_candle_pattern(df.iloc[-3:])
    if pattern and pattern["signal"] == "bearish":
        sell_signals.append({"type": f"Candle: {pattern['name']}", "strength": "strong",
            "detail": pattern["desc"], "urgency": "high"})
    return sell_signals

def score_stock(df):
    closes = df["Close"]
    rsi_s  = calc_rsi(closes)
    macd_s, sig_s, hist_s = calc_macd(closes)
    bb_u, bb_m, bb_l = calc_bollinger(closes)
    sma20 = closes.rolling(20).mean()
    sma50 = closes.rolling(50).mean()
    ema9  = closes.ewm(span=9, adjust=False).mean()
    atr   = calc_atr(df)
    pattern = detect_candle_pattern(df)
    def safe(s):
        try: v = s.iloc[-1]; return float(v) if not pd.isna(v) else None
        except: return None
    last_rsi=safe(rsi_s) or 50; last_macd=safe(macd_s) or 0; last_sig=safe(sig_s) or 0
    last_bb_u=safe(bb_u); last_bb_m=safe(bb_m); last_bb_l=safe(bb_l)
    last_sma20=safe(sma20); last_sma50=safe(sma50); last_ema9=safe(ema9); last_atr=safe(atr)
    price=float(closes.iloc[-1]); prev_price=float(closes.iloc[-2]) if len(closes)>1 else price
    price_5d=float(closes.iloc[-6]) if len(closes)>5 else price
    score=0; signals=[]
    if last_rsi<30: score+=30; signals.append({"label":"RSI Oversold","value":f"RSI: {last_rsi:.1f}","type":"buy"})
    elif last_rsi<45: score+=15; signals.append({"label":"RSI Low","value":f"RSI: {last_rsi:.1f}","type":"buy"})
    elif last_rsi>70: score-=25; signals.append({"label":"RSI Overbought","value":f"RSI: {last_rsi:.1f}","type":"sell"})
    else: signals.append({"label":"RSI Neutral","value":f"RSI: {last_rsi:.1f}","type":"neutral"})
    if last_macd>last_sig: score+=20; signals.append({"label":"MACD Bullish","value":f"{last_macd:.3f}","type":"buy"})
    else: score-=10; signals.append({"label":"MACD Bearish","value":f"{last_macd:.3f}","type":"sell"})
    if last_bb_l and price<last_bb_l: score+=20; signals.append({"label":"Below Lower BB","value":f"€{price:.2f}","type":"buy"})
    elif last_bb_u and price>last_bb_u: score-=15; signals.append({"label":"Above Upper BB","value":f"€{price:.2f}","type":"sell"})
    else: signals.append({"label":"Inside BB","value":"Normal range","type":"neutral"})
    if last_sma20 and last_sma50:
        if last_sma20>last_sma50: score+=15; signals.append({"label":"Uptrend","value":"SMA20>SMA50","type":"buy"})
        else: score-=10; signals.append({"label":"Downtrend","value":"SMA20<SMA50","type":"sell"})
    if last_ema9 and price>last_ema9: score+=10; signals.append({"label":"Price>EMA9","value":"Momentum+","type":"buy"})
    if pattern:
        if pattern["signal"]=="bullish": score+=15; signals.append({"label":pattern["name"],"value":pattern["desc"],"type":"buy"})
        elif pattern["signal"]=="bearish": score-=10; signals.append({"label":pattern["name"],"value":pattern["desc"],"type":"sell"})
    avg_vol=float(df["Volume"].iloc[-10:].mean()); last_vol=float(df["Volume"].iloc[-1])
    if avg_vol>0 and last_vol>avg_vol*1.5: score+=10; signals.append({"label":"Volume Spike","value":f"+{((last_vol/avg_vol)-1)*100:.0f}%","type":"buy"})
    rec=("STRONG BUY" if score>=50 else "BUY" if score>=25 else "HOLD" if score>=0 else "CAUTION" if score>=-20 else "AVOID")
    change1d=((price-prev_price)/prev_price*100) if prev_price else 0
    change5d=((price-price_5d)/price_5d*100) if price_5d else 0
    sell_sigs=detect_sell_signals(df,rsi_s,macd_s,sig_s,bb_u,bb_l,sma20,sma50)
    sell_score=len([s for s in sell_sigs if s["urgency"]=="high"])*2+len([s for s in sell_sigs if s["urgency"]=="medium"])
    sell_rating=("STRONG SELL" if sell_score>=4 else "SELL" if sell_score>=2 else "WATCH" if sell_score>=1 else "HOLD")
    levels=calc_price_levels(df,price,last_bb_u,last_bb_m,last_bb_l,last_sma50,last_atr)
    return {"score":score,"recommendation":rec,"signals":signals,"sell_signals":sell_sigs,"sell_rating":sell_rating,
            "rsi":round(last_rsi,2),"macd":round(last_macd,4),"macd_signal":round(last_sig,4),
            "price":round(price,2),"change1d":round(change1d,2),"change5d":round(change5d,2),
            "pattern":pattern,"atr":round(last_atr,2) if last_atr else None,
            "bb_upper":round(last_bb_u,2) if last_bb_u else None,
            "bb_lower":round(last_bb_l,2) if last_bb_l else None,
            "bb_mid":round(last_bb_m,2) if last_bb_m else None,
            "sma20":round(last_sma20,2) if last_sma20 else None,
            "sma50":round(last_sma50,2) if last_sma50 else None,
            "ema9":round(last_ema9,2) if last_ema9 else None,
            "volume":int(last_vol),"avg_volume":int(avg_vol),**levels}

_cache:dict={}; _cache_ts:dict={}; _prev_signals:dict={}; CACHE_TTL=300

def is_stale(sym):
    return sym not in _cache_ts or (datetime.utcnow()-_cache_ts[sym]).seconds>CACHE_TTL

def fetch_stock(sym):
    df=yf.Ticker(sym).history(period="6mo",interval="1d")
    if df.empty: raise ValueError(f"No data for {sym}")
    _cache[sym]=df; _cache_ts[sym]=datetime.utcnow()
    return df

async def check_and_alert(sym,name,analysis):
    prev=_prev_signals.get(sym,{}); rec=analysis["recommendation"]; sell=analysis["sell_rating"]; price=analysis["price"]
    if rec in ("BUY","STRONG BUY") and prev.get("rec") not in ("BUY","STRONG BUY"):
        await send_telegram(f"{'🟢🚀' if rec=='STRONG BUY' else '🟢'} <b>{rec} — {name} ({sym})</b>\n\n💰 €{price} | Score: {analysis['score']}/100\n🎯 T1: €{analysis['target1']} | T2: €{analysis['target2']}\n🛑 SL: €{analysis['stop_loss']}\n<i>Parakramee Intelligence</i>")
    if sell in ("SELL","STRONG SELL") and prev.get("sell") not in ("SELL","STRONG SELL"):
        await send_telegram(f"🔴 <b>{sell} — {name} ({sym})</b>\n\n💰 €{price}\n{chr(10).join(['• '+s['type'] for s in analysis['sell_signals'][:3]])}\n<i>Parakramee Intelligence</i>")
    _prev_signals[sym]={"rec":rec,"sell":sell}

@app.get("/")
def root(): return {"status":"ok","message":"Parakramee Intelligence API is running!"}

@app.get("/ping")
def ping(): return {"pong":True,"time":datetime.utcnow().isoformat()}

@app.get("/health")
def health(): return {"status":"ok","cached":list(_cache.keys()),"time":datetime.utcnow().isoformat()}

@app.get("/stocks")
async def get_all_stocks(background_tasks:BackgroundTasks):
    results=[]
    for stock in DAX_STOCKS:
        sym=stock["symbol"]
        try:
            df=fetch_stock(sym) if is_stale(sym) else _cache[sym]
            analysis=score_stock(df)
            background_tasks.add_task(check_and_alert,sym,stock["name"],analysis)
            results.append({**stock,**analysis})
        except Exception as e:
            print(f"Error {sym}: {e}")
            results.append({**stock,"error":str(e),"price":0,"score":0,"recommendation":"N/A","sell_signals":[],"sell_rating":"N/A"})
    return results

@app.get("/stock/{symbol}")
async def get_stock_detail(symbol:str,background_tasks:BackgroundTasks):
    sym=symbol.upper()
    try:
        df=fetch_stock(sym)
        closes=df["Close"]
        rsi_s=calc_rsi(closes); macd_s,sig_s,hist_s=calc_macd(closes)
        bb_u,bb_m,bb_l=calc_bollinger(closes)
        sma20=closes.rolling(20).mean(); sma50=closes.rolling(50).mean(); ema9=closes.ewm(span=9,adjust=False).mean()
        def to_list(s): return [round(float(v),4) if (v is not None and not pd.isna(v)) else None for v in s]
        ohlcv=[{"date":str(i.date()),"open":round(float(r["Open"]),2),"high":round(float(r["High"]),2),
                "low":round(float(r["Low"]),2),"close":round(float(r["Close"]),2),"volume":int(r["Volume"])} for i,r in df.iterrows()]
        indicators={"rsi":to_list(rsi_s),"macd":to_list(macd_s),"macd_signal":to_list(sig_s),"macd_hist":to_list(hist_s),
                    "bb_upper":to_list(bb_u),"bb_mid":to_list(bb_m),"bb_lower":to_list(bb_l),
                    "sma20":to_list(sma20),"sma50":to_list(sma50),"ema9":to_list(ema9)}
        analysis=score_stock(df)
        info={}
        try:
            raw=yf.Ticker(sym).info
            info={"longName":raw.get("longName",sym),"sector":raw.get("sector",""),"industry":raw.get("industry",""),
                  "marketCap":raw.get("marketCap"),"trailingPE":raw.get("trailingPE"),"forwardPE":raw.get("forwardPE"),
                  "dividendYield":raw.get("dividendYield"),"52wHigh":raw.get("fiftyTwoWeekHigh"),
                  "52wLow":raw.get("fiftyTwoWeekLow"),"beta":raw.get("beta")}
        except: pass
        stock_meta=next((s for s in DAX_STOCKS if s["symbol"]==sym),{"symbol":sym,"name":sym,"sector":""})
        background_tasks.add_task(check_and_alert,sym,stock_meta["name"],analysis)
        return {"symbol":sym,"ohlcv":ohlcv,"indicators":indicators,"analysis":analysis,"info":info}
    except Exception as e:
        raise HTTPException(500,str(e))

@app.post("/telegram/test")
async def test_telegram():
    await send_telegram("✅ <b>Parakramee Intelligence</b>\n\nTelegram alerts working!\n<i>Trade with Courage. Win with Intelligence.</i>")
    return {"status":"sent"}
