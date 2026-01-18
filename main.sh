#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_DIR"

echo "▶ Checking dependencies..."
$PYTHON_BIN - <<EOF
import sys, subprocess
req = [r.strip() for r in open("requirements.txt") if r.strip()]
for r in req:
    try:
        __import__(r.split("==")[0])
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", r])
EOF

echo "▶ Running ISSI MVP Final"
$PYTHON_BIN src/runner.py
