"""
download_prices.py  -  run ONCE on your laptop (needs internet).

    pip install yfinance
    python download_prices_v2.py

Saves 10 YEARS of real daily data (date, close, volume) for ~28 stocks into
data/<TICKER>.csv   (same folder your app already uses).

Why: your first backtest had only ~276 tests, so luck alone could move the
accuracy by +/-6%. More years x more stocks = thousands of tests, so a real
edge (if one exists) can actually be seen.

Your app (main.py) only reads AAPL/TSLA/GOOGL/INFY and uses the last 250
days, so the extra files do not change how the app behaves.
"""
import csv
import os
import sys
import time

try:
    import yfinance as yf
except ImportError:
    print("yfinance is not installed. Run:  pip install yfinance")
    sys.exit(1)

TICKERS = ["AAPL", "TSLA", "GOOGL", "INFY",                      # your 4
           "MSFT", "AMZN", "META", "NVDA", "NFLX", "AMD", "INTC", "ORCL",
           "ADBE", "CRM", "CSCO", "IBM", "JPM", "BAC", "GS", "WMT", "KO",
           "PEP", "DIS", "NKE", "XOM", "CVX", "PFE", "JNJ"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)

ok = 0
for t in TICKERS:
    try:
        df = yf.Ticker(t).history(period="10y", interval="1d", auto_adjust=True)
        df = df.dropna(subset=["Close"])
        if len(df) < 500:
            print(f"{t}: only {len(df)} rows - skipped")
            continue
        path = os.path.join(OUT_DIR, f"{t}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "close", "volume"])
            for date, close, vol in zip(df.index, df["Close"], df["Volume"]):
                w.writerow([str(date.date()), round(float(close), 4), int(vol)])
        print(f"{t:<6} {len(df)} days  (last close {df['Close'].iloc[-1]:.2f})")
        ok += 1
    except Exception as e:                       # keep going if one ticker fails
        print(f"{t}: failed ({type(e).__name__}: {str(e)[:60]})")
    time.sleep(0.4)

print(f"\nDone: {ok}/{len(TICKERS)} stocks saved in {OUT_DIR}")
print("Now run:  python backtest_v2.py")