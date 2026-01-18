import json, time
import yfinance as yf
from datetime import datetime
from config import MARKET_FILE
from cache import is_expired, set_expire

def fetch(tickers):
    data = {}
    for t in tickers:
        try:
            yf_t = yf.Ticker(f"{t}.JK")
            hist = yf_t.history(period="5d")
            if len(hist) < 2: 
                continue
            data[t] = {
                "price": float(hist["Close"].iloc[-1]),
                "volume_2d": int(hist["Volume"].tail(2).mean())
            }
            time.sleep(0.3)
        except Exception:
            continue
    return data

def get(tickers):
    if not MARKET_FILE.exists() or is_expired("market"):
        data = fetch(tickers)
        MARKET_FILE.write_text(json.dumps(data, indent=2))
        set_expire("market", datetime.now().replace(hour=8, minute=0))
        return data
    return json.loads(MARKET_FILE.read_text())
