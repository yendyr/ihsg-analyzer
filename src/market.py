import json
import yfinance as yf
from datetime import datetime
from config import MARKET_FILE
from cache import is_expired, set_expire

def fetch_market(tickers):
    result = {}
    for t in tickers:
        yf_t = yf.Ticker(f"{t}.JK")
        hist = yf_t.history(period="3d")
        if len(hist) >= 2:
            vol_avg = int(hist["Volume"][-2:].mean())
            price = float(hist["Close"][-1])
            result[t] = {"price": price, "volume_2d_avg": vol_avg}
    return result

def get_market_data(tickers):
    if not MARKET_FILE.exists() or is_expired("market_expire"):
        data = fetch_market(tickers)
        MARKET_FILE.write_text(json.dumps(data, indent=2))
        set_expire("market_expire", datetime.now().replace(hour=8, minute=0))
        return data
    return json.loads(MARKET_FILE.read_text())
