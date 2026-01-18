def sector_valuation(price, fund):
    sector = fund.get("sector","Other")
    roe = fund.get("returnOnEquity") or 0.1
    pe = fund.get("forwardPE") or fund.get("trailingPE") or 10

    if "Bank" in sector:
        fair = price * (roe / 0.12)
    elif "Energy" in sector or "Commodity" in sector:
        fair = price * (pe / 8)
    elif "Consumer" in sector:
        fair = price * (pe / 15)
    else:
        fair = price * (pe / 10)

    return {
        "conservative": round(fair * 0.9,2),
        "moderate": round(fair,2),
        "optimistic": round(fair * 1.2,2)
    }
