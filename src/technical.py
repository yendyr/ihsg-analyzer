# src/technical.py
import os
import json
from datetime import datetime, time
import yfinance as yf
import ta
import pandas as pd

CACHE_DIR = "data/cache/technical"


def _cache_path(ticker: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{ticker}.json")


def _is_expired(path: str) -> bool:
    """
    Cache expires every day at 08:00 local time.
    """
    if not os.path.exists(path):
        return True

    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    today_8am = datetime.combine(datetime.now().date(), time(8, 0))

    # If file was created before today's 08:00, it's expired
    return mtime < today_8am


def _load_cache(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(path: str, data: dict):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def analyze(ticker: str):
    """
    Technical analysis using RSI, trend, support & resistance.
    Cached daily, expires at 08:00.
    """

    cache_file = _cache_path(ticker)

    # ---- load cache if valid ----
    if not _is_expired(cache_file):
        cached = _load_cache(cache_file)
        if cached:
            return cached

    # ---- fetch from Yahoo ----
    try:
        df = yf.download(
            f"{ticker}.JK",
            period="6mo",
            auto_adjust=False,
            progress=False,
            threads=False
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # ---- ensure 1D Series ----
    close = df["Close"]
    low = df["Low"]
    high = df["High"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]

    if close.isna().all():
        return None

    # ---- RSI ----
    try:
        rsi_series = ta.momentum.RSIIndicator(close).rsi()
        rsi = float(rsi_series.iloc[-1])
    except Exception:
        rsi = None

    # ---- Trend (MA50) ----
    ma50 = close.rolling(50).mean().iloc[-1]
    price = close.iloc[-1]
    trend = "Uptrend" if price > ma50 else "Downtrend"

    # ---- Support & Resistance (20-day swing) ----
    support = float(low.tail(20).min())
    resistance = float(high.tail(20).max())

    result = {
        "ticker": ticker,
        "price": round(float(price), 2),
        "rsi": round(rsi, 2) if rsi is not None else None,
        "trend": trend,
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # ---- save cache ----
    _save_cache(cache_file, result)

    return result
