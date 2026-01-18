import json
from datetime import datetime
from issi import get
from market import get as get_market
from fundamental import get as get_fund
from technical import analyze
from valuation import sector_valuation
from report import build
from config import REPORT_FILE
from cache import is_expired, set_expire

def main():
    if REPORT_FILE.exists() and not is_expired("report"):
        print(REPORT_FILE.read_text())
        return

    issi = get()
    market = get_market(issi)
    fund = get_fund(issi)

    reports = []
    for t in issi:
        if t not in market or t not in fund:
            continue
        if not fund[t].get("priceToBook"):
            continue
        tech = analyze(t)
        if not tech:
            continue
        val = sector_valuation(market[t]["price"], fund[t])
        reports.append(build(t, market[t], fund[t], tech, val))

    reports.sort(
        key=lambda x: x["upside_%"] if isinstance(x["upside_%"], (int, float)) else -999,
        reverse=True
    )

    REPORT_FILE.write_text(json.dumps(reports, indent=2))
    set_expire("report", datetime.now().replace(hour=8,minute=30))
    print(json.dumps(reports, indent=2))

if __name__ == "__main__":
    main()
