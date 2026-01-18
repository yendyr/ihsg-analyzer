import json
import requests
from datetime import datetime, timedelta
from config import ISSI_FILE
from cache import is_expired, set_expire

IDX_ISSI_URL = "https://www.idx.co.id/umbraco/Surface/ListedCompany/GetShariaStock"

def fetch_issi():
    r = requests.get(IDX_ISSI_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    return sorted({row["KodeEmiten"] for row in data})

def get_issi_list():
    if not ISSI_FILE.exists() or is_expired("issi_expire"):
        data = fetch_issi()
        ISSI_FILE.write_text(json.dumps(data, indent=2))
        set_expire("issi_expire", datetime.now() + timedelta(days=30))
        return data
    return json.loads(ISSI_FILE.read_text())
