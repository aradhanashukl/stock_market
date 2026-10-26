"""
backtest.py  -  "Is the forecast actually any good?"

    python backtest.py

HOW IT WORKS (walk-forward test, no peeking at the future):
  For many past days t:
    1. Give the model ONLY the prices before day t.
    2. Ask: "will the price 5 days later be higher or lower than today?"
    3. Look at what really happened and score the guess.
  Then compare against two dumb baselines:
    - always-UP   : always guess "up"
    - momentum    : guess the same direction as the last 5 days
  If the model can't beat these, it has no real edge. That is normal -
  beating them consistently is very hard - and the honest number is
  what you should quote.

It tests the linear-regression forecaster (fast). Uses data/*.csv from
download_prices.py.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml.predictor import LinearPredictor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TICKERS  = ["AAPL", "TSLA", "GOOGL", "INFY"]
HORIZON  = 5      # predict 5 steps ahead
MIN_HIST = 60     # need at least this much history before the first test
STEP     = 5      # test every 5th day so tests do not overlap


def load(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return [float(r["close"]) for r in csv.DictReader(f)]


def backtest(closes):
    model = LinearPredictor(window=20, forecast=HORIZON)
    n = hits = up_hits = mom_hits = 0
    mae_model = mae_naive = 0.0
    for t in range(MIN_HIST, len(closes) - HORIZON, STEP):
        past   = closes[:t]
        today  = past[-1]
        actual = closes[t - 1 + HORIZON]

        model.train(past)
        pred = model.predict(past)["predictions"][-1]

        went_up   = actual > today
        n        += 1
        hits     += (pred > today) == went_up
        up_hits  += went_up                                   # "always up"
        mom_hits += ((today > past[-1 - HORIZON]) == went_up)  # "momentum"
        mae_model += abs(pred - actual)
        mae_naive += abs(today - actual)                      # "tomorrow = today"
    return n, hits, up_hits, mom_hits, mae_model, mae_naive


def main():
    print(f"{'Ticker':<7}{'Tests':>6}{'Model':>9}{'Always-UP':>11}{'Momentum':>10}"
          f"{'Err(model)':>12}{'Err(naive)':>12}")
    tot = [0, 0, 0, 0]
    found = False
    for t in TICKERS:
        closes = load(t)
        if not closes:
            print(f"{t:<7} no data/{t}.csv  (run download_prices.py first)")
            continue
        found = True
        n, h, u, m, em, en = backtest(closes)
        if n == 0:
            print(f"{t:<7} not enough history ({len(closes)} rows)")
            continue
        tot = [tot[0] + n, tot[1] + h, tot[2] + u, tot[3] + m]
        print(f"{t:<7}{n:>6}{h/n:>9.1%}{u/n:>11.1%}{m/n:>10.1%}"
              f"{em/n:>12.2f}{en/n:>12.2f}")
    if found and tot[0]:
        print(f"{'ALL':<7}{tot[0]:>6}{tot[1]/tot[0]:>9.1%}{tot[2]/tot[0]:>11.1%}"
              f"{tot[3]/tot[0]:>10.1%}")
        print("\nRead it like this: 50% = a coin flip. Only trust an edge if the "
              "Model column is clearly above BOTH baselines on many tests.")

if __name__ == "__main__":
    main()