def build(t, market, fund, tech, val):
    price = market["price"]
    return {
        "ticker": t,
        "price": price,
        "pbv": fund["pbv"],
        "roe": fund["roe"],
        "valuation": val,
        "upside_moderate_%": round((val["moderate"] - price) / price * 100, 2),
        "technical": tech
    }
