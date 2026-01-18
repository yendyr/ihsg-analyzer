import yfinance as yf
import ta

def analyze(t):
    df = yf.download(f"{t}.JK", period="6mo", progress=False)
    if df.empty:
        return None
    rsi = ta.momentum.RSIIndicator(df["Close"]).rsi().iloc[-1]
    ma50 = df["Close"].rolling(50).mean().iloc[-1]
    price = df["Close"].iloc[-1]
    trend = "Uptrend" if price > ma50 else "Downtrend"
    return {
        "rsi": round(float(rsi),2),
        "trend": trend,
        "support": round(float(df["Low"].tail(20).min()),2),
        "resistance": round(float(df["High"].tail(20).max()),2)
    }
