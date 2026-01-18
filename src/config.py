from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

ISSI_FILE = DATA_DIR / "issi_list.json"
MARKET_FILE = DATA_DIR / "market.json"
FUND_FILE = DATA_DIR / "fundamental.json"
REPORT_FILE = DATA_DIR / "report.json"
META_FILE = DATA_DIR / "meta.json"

PBV_MIN = 0
PBV_MAX = 2
VOLUME_MIN = 200_000
