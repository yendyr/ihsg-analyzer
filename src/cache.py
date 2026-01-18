import json
from datetime import datetime
from config import META_FILE

def load_meta():
    if META_FILE.exists():
        return json.loads(META_FILE.read_text())
    return {}

def save_meta(meta):
    META_FILE.write_text(json.dumps(meta, indent=2))

def is_expired(key):
    meta = load_meta()
    exp = meta.get(key)
    if not exp:
        return True
    return datetime.now() >= datetime.fromisoformat(exp)

def set_expire(key, dt):
    meta = load_meta()
    meta[key] = dt.isoformat()
    save_meta(meta)
