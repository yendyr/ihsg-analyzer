def _to_float(val):
    """
    Safely convert valuation inputs to float.
    """
    if val is None:
        return None

    if isinstance(val, (int, float)):
        return float(val)

    if isinstance(val, str):
        val = val.strip().replace(",", "")
        if val == "" or val.lower() in ["n/a", "na", "-", "null"]:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    return None

def sector_valuation(price, fund):
    price = _to_float(price)
    if price is None or price <= 0:
        return None
    
    raw_sector = fund.get("sector")
    sector = raw_sector if isinstance(raw_sector, str) else "Other"

    roe = _to_float(fund.get("returnOnEquity"))
    pe = _to_float(fund.get("forwardPE")) or _to_float(fund.get("trailingPE"))
    pbv = _to_float(fund.get("priceToBook"))

    # ---- sanitize ----
    roe = roe if roe is not None else 0
    pe = pe if pe and pe > 0 else None
    pbv = pbv if pbv and pbv > 0 else None

    fair = None

    if "Bank" in sector:
        # ROE-based only if ROE healthy
        if roe > 0.08:
            fair = price * (roe / 0.12)
        elif pbv:
            fair = price * (1.5 / pbv)  # PBV mean-reversion
        else:
            return None
    elif "Energy" in sector or "Mining" in sector or "Commodity" in sector or "Materials" in sector:
        if pe:
            fair = price * (pe / 8)
        else:
            return None
    elif "Consumer" in sector or "Retail" in sector or "Communication" in sector:
        if pe:
            fair = price * (pe / 15)
        elif pbv:
            fair = price * (2 / pbv)
        else:
            return None
    elif "Real Estate" in sector or "Property" in sector:
        if pbv:
            fair = price * (1.2 / pbv)
        else:
            return None
    elif "Infrastructure" in sector or "Industrial" in sector or "Utilities" in sector or "Transportation" in sector:
        if pe:
            fair = price * (pe / 12)
        elif pbv:
            fair = price * (1.5 / pbv)
        else:
            return None
    elif "Technology" in sector or "Healthcare" in sector:
        if pe:
            fair = price * (pe / 20)
        else:
            return None
    else:
        if pe:
            fair = price * (pe / 10)
        elif pbv:
            fair = price * (1.5 / pbv)
        else:
            return None

    # ---- final guardrail ----
    if fair is None or fair <= 0 or fair > price * 5:
        return None
    
    return {
        "conservative": round(fair * 0.85, 2),
        "moderate": round(fair,2),
        "optimistic": round(fair * 1.15,2)
    }
