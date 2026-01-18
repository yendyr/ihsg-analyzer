import json
from datetime import datetime
from cache import is_expired, set_expire
from config import REPORT_FILE
from issi import get_issi_list
from market import get_market_data
from fundamental import get_fundamental, screening
from technical import analyze
from valuation import calculate
from report import build

def main():
    if REPORT_FILE.exists() and not is_expired("report_expire"):
        print(REPORT_FILE.read_text())
        return

    issi = get_issi_list()
    market = get_market_data(issi)
    fund = get_fundamental(issi)
    passed = screening(issi, market, fund)

    reports = []
    for t in passed:
        tech = analyze(t)
        val = calculate(market[t]["price"], fund[t]["roe"])
        reports.append(build(t, market[t], fund[t], tech, val))

    REPORT_FILE.write_text(json.dumps(reports, indent=2))
    set_expire("report_expire", datetime.now().replace(hour=8, minute=30))
    print(json.dumps(reports, indent=2))

if __name__ == "__main__":
    main()
