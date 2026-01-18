from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

ISSI_FILE = DATA_DIR / "issi_list.json"
MARKET_FILE = DATA_DIR / "market_data.json"
FIN_FILE = DATA_DIR / "financial_data.json"
REPORT_FILE = DATA_DIR / "report.json"
META_FILE = DATA_DIR / "meta.json"

PBV_MIN = 0
PBV_MAX = 2
VOLUME_MIN = 200_000
