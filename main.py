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
            net_income = entry.get("net_income")
            net_income_prev = entry.get("net_income_prev")
            trend = entry.get("trend")
            liquidity = entry.get("liquidity")
            volatility = entry.get("volatility")
            volume_trend = entry.get("volume_trend", "")
            return (ticker, info, hist, net_income, net_income_prev, trend, liquidity, volatility, volume_trend, None)
    try:
        data = yf.Ticker(ticker)
        info = data.info
        hist = data.history(period="2mo")
        hist = hist.tail(30)  # 🔹 hanya 30 hari terakhir
        quarterly = data.quarterly_financials

        net_income = net_income_prev = trend = None

        if not quarterly.empty and "Net Income" in quarterly.index:
            netincomes = quarterly.loc["Net Income"].to_list()
            if quarterly.columns[0] < quarterly.columns[-1]:
                netincomes = netincomes[::-1]
            if len(netincomes) >= 1:
                net_income = netincomes[0]
            if len(netincomes) >= 2:
                net_income_prev = netincomes[1]
            if net_income is not None and net_income_prev is not None:
                if net_income > net_income_prev:
                    trend = "▲"
                elif net_income < net_income_prev:
                    trend = "▼"
                else:
                    trend = "→"

        # === Likuiditas & Volatilitas 30 hari ===
        liquidity = volatility = volume_trend = None
        if hist is not None and not hist.empty:
            avg_volume = hist["Volume"].mean()
            max_volume = hist["Volume"].max()
            liquidity = (avg_volume / max_volume * 100) if max_volume > 0 else 0
            liquidity = min(round(liquidity, 2), 100)

            # Tren volume 3 hari terakhir
            if len(hist) >= 4:
                vols = hist["Volume"].tail(4).values
                if vols[1] < vols[2] < vols[3]:
                    volume_trend = " ▲"
                elif vols[1] > vols[2] > vols[3]:
                    volume_trend = " ▼"
                else:
                    volume_trend = ""
            else:
                volume_trend = ""

            # Volatilitas 30 hari (harian)
            daily_returns = hist["Close"].pct_change().dropna()
            if not daily_returns.empty:
                volatility = round(daily_returns.std(), 4)

        if volatility is not None:
            volatility = round(volatility * 100, 2)

        new_cache_entry = None
        if hist is not None and not hist.empty:
            hist_serializable = hist.reset_index()
            hist_serializable['Date'] = hist_serializable['Date'].astype(str)
            hist_dict = hist_serializable.to_dict(orient='list')
            new_cache_entry = {
                "timestamp": now,
                "info": info,
                "hist": hist_dict,
                "net_income": net_income,
                "net_income_prev": net_income_prev,
                "trend": trend,
                "liquidity": liquidity,
                "volatility": volatility,
                "volume_trend": volume_trend
            }

        return (ticker, info, hist, net_income, net_income_prev, trend, liquidity, volatility, volume_trend, new_cache_entry)
    except Exception:
        return (ticker, None, None, None, None, None, None, None, None, None)

# === INPUT ===
try:
    code = input("Masukkan kode saham (atau ENTER untuk baca list syariah): ").strip().upper()
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

# === Proses ===
output = []
new_cache_entries = {}
max_threads = 2

with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = {executor.submit(fetch_stock, t): t for t in tickers}
    for future in tqdm(as_completed(futures), total=len(futures), desc="⏳ Memproses ticker"):
        tkr = futures[future]
        try:
            ticker, info, hist, net_income, net_income_prev, trend, liquidity, volatility, volume_trend, new_entry = future.result()
            if new_entry:
                new_cache_entries[ticker] = new_entry
            if hist is None or hist.empty:
                continue

            current_price = info.get("currentPrice")
            pbv = info.get("priceToBook")
            roe = info.get("returnOnEquity")
            book_value = info.get("bookValue")
            target_price = info.get("targetMeanPrice")
            per = info.get("trailingPE")
            roa = info.get("returnOnAssets")

            # === Filter ===
            if not mode_prompt:
                if net_income is None or net_income <= 0:
                    continue
                if pbv is None or pbv <= 0 or pbv > 0.9:
                    continue
                if target_price and target_price < current_price:
                    continue
                if (not target_price) and book_value and (book_value * 0.9) < current_price:
                    continue

            reason_skip = []

            # === PER Berdasarkan Sektor ===
            SECTOR_PER_AVG = {
                "Financial Services": 12, "Consumer Defensive": 18, "Energy": 8, "Technology": 25,
                "Industrials": 15, "Basic Materials": 14, "Healthcare": 20, "Real Estate": 10, "Utilities": 10
            }
            sector = info.get("sector")
            avg_per = SECTOR_PER_AVG.get(sector, 15)
            max_per_allowed = avg_per * 1.3
            min_per_allowed = avg_per * 0.4
            if per is None:
                reason_skip.append("PER: n/a")
            elif per <= 0:
                reason_skip.append(f"PER {per:.2f} (negatif)")
            elif per < min_per_allowed:
                reason_skip.append(f"PER {per:.2f} < sektor-min {min_per_allowed:.1f}")
            elif per > max_per_allowed:
                reason_skip.append(f"PER {per:.2f} > sektor-max {max_per_allowed:.1f}")

            # === ROA Berdasarkan Sektor ===
            SECTOR_ROA_AVG = {
                "Financial Services": 0.02, "Consumer Defensive": 0.10, "Energy": 0.07, "Technology": 0.08,
                "Industrials": 0.06, "Basic Materials": 0.06, "Healthcare": 0.07, "Real Estate": 0.04, "Utilities": 0.05
            }
            avg_roa = SECTOR_ROA_AVG.get(sector, 0.06)
            min_roa_allowed = avg_roa * 0.5
            max_roa_allowed = avg_roa * 2.0
            if roa is None:
                reason_skip.append("ROA: n/a")
            elif roa < min_roa_allowed:
                reason_skip.append(f"ROA {roa*100:.1f}% < sektor-min {min_roa_allowed*100:.1f}%")
            elif roa > max_roa_allowed:
                reason_skip.append(f"ROA {roa*100:.1f}% > sektor-max {max_roa_allowed*100:.1f}%")

            # === ROE Berdasarkan Sektor ===
            SECTOR_ROE_AVG = {
                "Financial Services": 0.15, "Consumer Defensive": 0.20, "Energy": 0.12, "Technology": 0.15,
                "Industrials": 0.12, "Basic Materials": 0.10, "Healthcare": 0.13, "Real Estate": 0.08, "Utilities": 0.09
            }
            avg_roe = SECTOR_ROE_AVG.get(sector, 0.12)
            min_roe_allowed = avg_roe * 0.5
            max_roe_allowed = avg_roe * 2.0
            if roe is None:
                reason_skip.append("ROE: n/a")
            elif roe < min_roe_allowed:
                reason_skip.append(f"ROE {roe*100:.1f}% < sektor-min {min_roe_allowed*100:.1f}%")
            elif roe > max_roe_allowed:
                reason_skip.append(f"ROE {roe*100:.1f}% > sektor-max {max_roe_allowed*100:.1f}%")

            # === Support & Resistance (ATR 14 dari 30 hari) ===
            recent = hist.tail(30)
            low = recent["Low"].min()
            high = recent["High"].max()
            tr = pd.concat([
                recent["High"] - recent["Low"],
                (recent["High"] - recent["Close"].shift()).abs(),
                (recent["Low"] - recent["Close"].shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            if np.isnan(atr) or atr == 0:
                atr = recent["Close"].iloc[-1] * 0.02
            support = round(low + 0.5 * atr, 2)
            resistance = round(high - 0.5 * atr, 2)

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
                "NetProfit": f"{format_rupiah(net_income)} {trend or ''}",
                "Liquidity": f"{liquidity:,.0f}{volume_trend or ''}" if liquidity is not None else "-",
                "Volatility(%) (30d)": f"{volatility:.2f}" if volatility is not None else "-",
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
                "Liquidity": "-",
                "Volatility(%) (30d)": "-",
                "Support": "-",
                "Resistance": "-",
                "Fundamental": f"Error: {e}"
            })

# === Simpan cache ===
if new_cache_entries:
    current_cache = load_cache()
    current_cache.update(new_cache_entries)
    save_cache(current_cache)

# === Output ===
if output:
    output_sorted = sorted(output, key=lambda x: (0 if x["Fundamental"] == "Sehat" else 1, x["PBV"]))
    headers = ["Code","Harga","FairValueAnalyst","%ToFairAnalyst","FairValueProxy","%ToFairProxy","PBV",
               "NetProfit","Liquidity","Volatility(%)","Supp.","Resist.","State"]
    rows = []

    for r in output_sorted:
        tofair_analyst_val = None
        tofair_proxy_val = None

        try:
            tofair_analyst_val = float(str(r['%ToFairAnalyst']).replace('%','').replace(',','.'))
        except:
            pass
        try:
            tofair_proxy_val = float(str(r['%ToFairProxy']).replace('%','').replace(',','.'))
        except:
            pass

        pbv_val = r['PBV'] if isinstance(r['PBV'], (float, int)) else None
        liquidity_str = str(r['Liquidity'])
        liquidity_val = None
        try:
            # ambil angka di awal string sebelum spasi atau simbol panah
            liquidity_val = float(liquidity_str.split()[0].replace(',', '.'))
        except:
            pass
        netprofit_str = str(r['NetProfit'])

        # 🟩 highlight logic
        undervalued = (
            (tofair_analyst_val is not None and tofair_analyst_val > 25)
            or (tofair_proxy_val is not None and tofair_proxy_val > 50)
        )
        liquidity_high = liquidity_val is not None and liquidity_val > 40
        liquidity_up = "▲" in liquidity_str
        profit_up = "▲" in netprofit_str
        pbv_low = pbv_val is not None and pbv_val < 0.8

        highlight_green = False
        if undervalued and (liquidity_high or liquidity_up) and (profit_up or pbv_low):
            highlight_green = True

        # === Format tampilan ===
        tofair_analyst = f"{r['%ToFairAnalyst']}%" if r['%ToFairAnalyst'] not in ["-", None] else "-"
        tofair_proxy = f"{r['%ToFairProxy']}%" if r['%ToFairProxy'] not in ["-", None] else "-"
        pbv_fmt = f"{r['PBV']:.2f}" if isinstance(r['PBV'], (float, int)) else "-"

        row = [
            r['Ticker'], r['Harga'], r['FairValueAnalyst'], tofair_analyst, r['FairValueProxy'],
            tofair_proxy, pbv_fmt, r['NetProfit'], r['Liquidity'], r['Volatility(%) (30d)'],
            r['Support'], r['Resistance'], r['Fundamental']
        ]

        # 🟩 Jika memenuhi semua kondisi → warna hijau terang
        if highlight_green:
            row = [f"\033[92m{v}\033[0m" for v in row]  # ANSI bright green

        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="fancy_grid", stralign="right", numalign="right"))
else:
    print("⚠️ Tidak ada saham yang valid ditemukan.")
