#!/bin/bash

# Cek dan install library yang dibutuhkan
python3 - <<'PYCHECK'
import importlib.util, subprocess, sys
for pkg in ["tqdm", "tabulate", "yfinance", "pandas", "numpy"]:
    if importlib.util.find_spec(pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--user", pkg])
PYCHECK

echo "📦 Mengambil data saham syariah sehat (PBV < 0.9, NetProfit positif) dengan cache 10 menit (lokal)"
python3 main.py
