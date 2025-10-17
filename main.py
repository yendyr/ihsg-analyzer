import yfinance as yf
import numpy as np
import pandas as pd
import json, os, time, sys
from tqdm import tqdm
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Konfigurasi cache ===
CACHE_FILE = os.path.join(os.getcwd(), "stock_cache.json")
CACHE_TTL = 600

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"⚠️ Gagal simpan cache: {e}")

cache = load_cache()

def format_rupiah(n):
    if not n or n == "-" or (isinstance(n, float) and np.isnan(n)):
        return "-"
    try:
        n = float(n)
        if n >= 1e12:
            return f"Rp {n/1e12:.2f} T"
        elif n >= 1e9:
            return f"Rp {n/1e9:.2f} M"
        elif n >= 1e6:
            return f"Rp {n/1e6:.2f} Jt"
        else:
            return f"Rp {n:,.0f}".replace(",", ".")
    except:
        return "-"

def fmt_num(x):
    if x is None or x == "-" or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if isinstance(x, (float, int)):
        return f"{x:,.2f}".replace(",", ".")
    return str(x)

def hist_from_cache(hist_dict):
    try:
        df = pd.DataFrame(hist_dict)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()

def fetch_stock(ticker):
    now = time.time()
    if ticker in cache:
        entry = cache[ticker]
        if now - entry.get("timestamp", 0) < CACHE_TTL and entry.get("hist") is not None:
            hist = hist_from_cache(entry["hist"])
            info = entry.get("info", {})
            return (ticker, info, hist, None)
    try:
        data = yf.Ticker(ticker)
        info = data.info
        hist = data.history(period="3mo")
        new_cache_entry = None
        if hist is not None and not hist.empty:
            hist_serializable = hist.tail(60).reset_index()
            hist_serializable['Date'] = hist_serializable['Date'].astype(str)
            hist_dict = hist_serializable.to_dict(orient='list')
            new_cache_entry = {"timestamp": now, "info": info, "hist": hist_dict}
        return (ticker, info, hist, new_cache_entry)
    except Exception:
        return (ticker, None, None, None)

# === INPUT MANUAL / FILE ===
try:
    code = input("Masukkan kode saham (atau ENTER untuk baca hanya dari list syariah): ").strip().upper()
except EOFError:
    code = ""

if code:
    tickers = [code + ".JK"]
    print(f"📊 Mode tunggal: {code}")
    mode_prompt = True
else:
    with open("shariah_tickers.txt") as f:
        tickers = [line.strip() + ".JK" for line in f.readlines()]
    print(f"📃 Mode daftar syariah: {len(tickers)} ticker dibaca dari file")
    mode_prompt = False


# === Mulai proses ===
output = []
new_cache_entries = {}
max_threads = 3

with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = {executor.submit(fetch_stock, t): t for t in tickers}
    for future in tqdm(as_completed(futures), total=len(futures), desc="⏳ Memproses ticker"):
        tkr = futures[future]
        try:
            ticker, info, hist, new_entry = future.result()
            if new_entry:
                new_cache_entries[ticker] = new_entry
            if hist is None or hist.empty:
                continue

            net_income = info.get("netIncomeToCommon")
            current_price = info.get("currentPrice")
            pbv = info.get("priceToBook")
            roe = info.get("returnOnEquity")
            book_value = info.get("bookValue")
            target_price = info.get("targetMeanPrice")
            per = info.get("trailingPE")
            roa = info.get("returnOnAssets")

            # ==== FILTER HANYA JIKA MODE FILE ====
            if not mode_prompt:
                if net_income is None or net_income <= 0:
                    continue
                if pbv is None or pbv <= 0 or pbv > 0.9:
                    continue
                if target_price and target_price < current_price:
                    continue
                if (not target_price) and book_value and (book_value * 0.9) < current_price:
                    continue
            # =====================================

            reason_skip = []
            if per is None:
                reason_skip.append("PER: n/a")
            elif per <= 0 or per > 25:
                reason_skip.append(f"PER {per:.2f}")

            if roa is None:
                reason_skip.append("ROA: n/a")
            elif roa < 0.05:
                reason_skip.append(f"ROA {roa*100:.1f}%")

            if roe is None:
                reason_skip.append("ROE: n/a")
            elif roe < 0.08:
                reason_skip.append(f"ROE {roe*100:.1f}%")

            recent = hist[-60:]
            low = recent["Low"].min()
            high = recent["High"].max()
            support = round(low * 1.05, 2)
            resistance = round(high * 0.95, 2)

            fair_value_analyst = target_price
            fair_value_proxy = book_value * 0.9 if book_value else None
            tofair_analyst = round((fair_value_analyst - current_price)/current_price*100, 2) if fair_value_analyst else None
            tofair_proxy = round((fair_value_proxy - current_price)/current_price*100, 2) if fair_value_proxy else None

            fundamental_status = "Sehat" if not reason_skip else "Kurang"

            output.append({
                "Ticker": ticker.replace(".JK",""),
                "Harga": fmt_num(current_price),
                "FairValueAnalyst": fmt_num(fair_value_analyst),
                "%ToFairAnalyst": fmt_num(tofair_analyst),
                "FairValueProxy": fmt_num(fair_value_proxy),
                "%ToFairProxy": fmt_num(tofair_proxy),
                "PBV": pbv,
                "NetProfit": format_rupiah(net_income),
                "Support": fmt_num(support),
                "Resistance": fmt_num(resistance),
                "Fundamental": fundamental_status
            })
        except Exception as e:
            output.append({
                "Ticker": tkr.replace(".JK",""),
                "Harga": "-",
                "FairValueAnalyst": "-",
                "%ToFairAnalyst": "-",
                "FairValueProxy": "-",
                "%ToFairProxy": "-",
                "PBV": 999,
                "NetProfit": "-",
                "Support": "-",
                "Resistance": "-",
                "Fundamental": f"Error: {e}"
            })

# === Simpan cache ===
if new_cache_entries:
    current_cache = load_cache()
    current_cache.update(new_cache_entries)
    save_cache(current_cache)

# === Output hasil ===
if output:
    output_sorted = sorted(
        output,
        key=lambda x: (0 if x["Fundamental"] == "Sehat" else 1, x["PBV"])
    )

    headers = ["Ticker", "Harga", "FairValueAnalyst", "%ToFairAnalyst",
               "FairValueProxy", "%ToFairProxy", "PBV", "NetProfit",
               "Support", "Resistance", "Fundamental"]

    rows = []
    for r in output_sorted:
        fair_analyst = r["FairValueAnalyst"]
        tofair_analyst_val = r["%ToFairAnalyst"]
        tofair_proxy_val = r["%ToFairProxy"]
        tofair_analyst = f"{tofair_analyst_val}%" if tofair_analyst_val not in ["-", None] else "-"
        tofair_proxy = f"{tofair_proxy_val}%" if tofair_proxy_val not in ["-", None] else "-"
        pbv_fmt = f"{r['PBV']:.2f}" if isinstance(r["PBV"], (float, int)) else "-"

        if fair_analyst not in ["-", None]: 
            color_start = "\033[92m" 
            color_end = "\033[0m"
        else: 
            color_start = ""
            color_end = ""

        rows.append([
            f"{color_start}{r['Ticker']}{color_end}",
            f"{color_start}{r['Harga']}{color_end}",
            f"{color_start}{fair_analyst}{color_end}",
            f"{color_start}{tofair_analyst}{color_end}",
            f"{color_start}{r['FairValueProxy']}{color_end}",
            f"{color_start}{tofair_proxy}{color_end}",
            f"{color_start}{pbv_fmt}{color_end}",
            f"{color_start}{r['NetProfit']}{color_end}",
            f"{color_start}{r['Support']}{color_end}",
            f"{color_start}{r['Resistance']}{color_end}",
            f"{color_start}{r['Fundamental']}{color_end}"
        ])

    print(tabulate(rows, headers=headers, tablefmt="fancy_grid", stralign="right", numalign="right"))
else:
    print("⚠️ Tidak ada saham yang valid ditemukan.")
