import json
import yfinance as yf
from datetime import datetime
from config import FIN_FILE, PBV_MIN, PBV_MAX, VOLUME_MIN
from cache import is_expired, set_expire

def fetch_fundamental(tickers):
    data = {}
    for t in tickers:
        info = yf.Ticker(f"{t}.JK").info
        pbv = info.get("priceToBook")
        roe = info.get("returnOnEquity")
        if pbv:
            data[t] = {"pbv": pbv, "roe": roe}
    return data

def get_fundamental(tickers):
    if not FIN_FILE.exists() or is_expired("fund_expire"):
        data = fetch_fundamental(tickers)
        FIN_FILE.write_text(json.dumps(data, indent=2))
        set_expire("fund_expire", datetime.now().replace(hour=8, minute=0))
        return data
    return json.loads(FIN_FILE.read_text())

def screening(issi, market, fund):
    result = []
    for t in issi:
        if t not in market or t not in fund:
            continue
        pbv = fund[t]["pbv"]
        vol = market[t]["volume_2d_avg"]
        if PBV_MIN < pbv < PBV_MAX and vol > VOLUME_MIN:
            result.append(t)
    return result
