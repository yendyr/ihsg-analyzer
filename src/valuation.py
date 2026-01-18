def calculate(price, roe):
    base = price * (1 + (roe or 0.1))
    return {
        "conservative": round(base * 0.9, 2),
        "moderate": round(base * 1.0, 2),
        "optimistic": round(base * 1.2, 2)
    }
