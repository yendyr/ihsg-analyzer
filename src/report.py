def narrative(t, fund, tech):
    return (
        f"{t} berada di sektor {fund.get('sector')}. "
        f"PBV {round(fund.get('priceToBook',0),2)} dan ROE "
        f"{round((fund.get('returnOnEquity') or 0)*100,2)}%. "
        f"Secara teknikal RSI {tech['rsi']} dan trend {tech['trend']}."
    )

def build(t, market, fund, tech, val):
    price = market["price"]
    if val is None:
        upside = None
    else:
        upside = (val["moderate"] - price) / price * 100

    return {
        "ticker": t,
        "price": price,
        "valuation": val,
        "upside_%": round(upside, 2) if isinstance(upside, (int, float)) else None,
        "consensus": fund.get("targetPrice"),
        "technical": tech,
        "narrative": narrative(t,fund,tech)
    }
