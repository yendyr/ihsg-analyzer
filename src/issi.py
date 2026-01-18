import json, requests
from datetime import datetime, timedelta
from config import ISSI_FILE
from cache import is_expired, set_expire

URL = "https://www.idx.co.id/umbraco/Surface/ListedCompany/GetShariaStock"

def fetch():
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    return sorted({i["KodeEmiten"] for i in r.json()})

def get():
    if not ISSI_FILE.exists() or is_expired("issi"):
        data = fetch()
        ISSI_FILE.write_text(json.dumps(data, indent=2))
        set_expire("issi", datetime.now() + timedelta(days=30))
        return data
    return json.loads(ISSI_FILE.read_text())
