import yfinance as yf
import ta

def analyze(ticker):
    df = yf.download(f"{ticker}.JK", period="6mo", progress=False)
    rsi = ta.momentum.RSIIndicator(df["Close"]).rsi().iloc[-1]
    trend = "Uptrend" if df["Close"].iloc[-1] > df["Close"].rolling(50).mean().iloc[-1] else "Downtrend"
    support = df["Low"].tail(20).min()
    resistance = df["High"].tail(20).max()
    return {
        "rsi": round(float(rsi), 2),
        "trend": trend,
        "support": round(float(support), 2),
        "resistance": round(float(resistance), 2)
    }
