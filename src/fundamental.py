import json, time
import yfinance as yf
from datetime import datetime
from config import FUND_FILE
from cache import is_expired, set_expire

FIELDS = [
    "priceToBook","returnOnEquity","profitMargins",
    "debtToEquity","earningsGrowth","revenueGrowth",
    "sector","trailingPE","forwardPE"
]

def fetch(tickers):
    data = {}
    for t in tickers:
        try:
            info = yf.Ticker(f"{t}.JK").info
            data[t] = {f: info.get(f) for f in FIELDS}
            data[t]["targetPrice"] = info.get("targetMeanPrice")
            time.sleep(0.3)
        except Exception:
            continue
    return data

def get(tickers):
    if not FUND_FILE.exists() or is_expired("fund"):
        data = fetch(tickers)
        FUND_FILE.write_text(json.dumps(data, indent=2))
        set_expire("fund", datetime.now().replace(hour=8, minute=0))
        return data
    return json.loads(FUND_FILE.read_text())
