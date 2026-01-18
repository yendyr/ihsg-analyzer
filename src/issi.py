# src/issi.py
import json
import os
from datetime import datetime, timedelta

ISSI_FILE = os.path.join("data", "issi_list.json")
MAX_AGE_DAYS = 30


def get():
    """
    Load ISSI / DES universe from local static JSON only.
    No online fetch. Production-safe.
    """

    if not os.path.exists(ISSI_FILE):
        raise RuntimeError(
            "ISSI master file not found.\n"
            "Expected file: data/issi_list.json\n"
            "Please generate it from official IDX / DES PDF."
        )

    with open(ISSI_FILE, "r") as f:
        data = json.load(f)

    # ---- basic validation ----
    if "symbols" not in data or not isinstance(data["symbols"], list):
        raise RuntimeError("Invalid issi_list.json format: missing 'symbols' list")

    symbols = [s.strip().upper() for s in data["symbols"] if s.strip()]

    if len(symbols) < 100:
        raise RuntimeError(
            f"ISSI list too small ({len(symbols)} symbols). "
            "Likely invalid or incomplete file."
        )

    # ---- expiration warning (non-blocking) ----
    last_updated = data.get("last_updated")
    if last_updated:
        try:
            last_dt = datetime.strptime(last_updated[:10], "%Y-%m-%d")
            age_days = (datetime.now() - last_dt).days
            if age_days > MAX_AGE_DAYS:
                print(
                    f"⚠️  WARNING: ISSI list is {age_days} days old. "
                    "Consider updating from latest DES."
                )
        except Exception:
            print("⚠️  WARNING: Unable to parse ISSI last_updated date")

    # ---- final clean universe ----
    symbols = sorted(set(symbols))

    return symbols
