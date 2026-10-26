"""
backtest_lstm.py  -  "Does the LSTM beat the simple model? Be honest."

    python backtest_lstm.py          # full test   (slow: several minutes)
    python backtest_lstm.py quick    # every 20th day (about half the time)

Walk-forward test, no peeking at the future. On each test day t:
  1. Show the models ONLY prices before day t (the last 250, like the live app).
  2. Ask: "will the price 5 days later be higher or lower than today?"
  3. Check what really happened.

The linear model and the LSTM are tested on the SAME days, so the
comparison is fair. Baselines: always-UP and momentum.

CONFUSION MATRIX (rows = what really happened, columns = what the model said):

                     model said UP    model said DOWN
    really went UP   [ right  ]       [ missed a rise ]
    really went DOWN [ false alarm ]  [ right  ]

Uses data/*.csv from download_prices.py. Saves backtest_results.json.
"""
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml.predictor import LinearPredictor, LSTMPredictor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results.json")
TICKERS  = ["AAPL", "TSLA", "GOOGL", "INFY"]
HORIZON  = 5       # predict 5 days ahead
MIN_HIST = 60      # history needed before the first test
TRAIN_N  = 250     # same window size the live app uses
STEP     = 20 if "quick" in sys.argv else 10


def load(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return [float(r["close"]) for r in csv.DictReader(f)]


class Score:
    """Collects results for one model."""
    def __init__(self):
        self.tp = self.fn = self.fp = self.tn = 0     # UP is the "positive" class
        self.err = self.sq = self.pct = self.bias = 0.0
        self.up_ret = self.dn_ret = 0.0               # sum of real 5-day returns
        self.up_n = self.dn_n = 0
        self.failed = 0

    def add(self, said_up, went_up, pred_price, actual, today):
        if went_up and said_up:         self.tp += 1
        elif went_up and not said_up:   self.fn += 1
        elif not went_up and said_up:   self.fp += 1
        else:                           self.tn += 1
        e = pred_price - actual
        self.err  += abs(e)
        self.sq   += e * e
        self.pct  += abs(e) / actual * 100
        self.bias += e
        ret = (actual / today - 1) * 100              # what REALLY happened, in %
        if said_up: self.up_ret += ret; self.up_n += 1
        else:       self.dn_ret += ret; self.dn_n += 1

    @property
    def n(self):            return self.tp + self.fn + self.fp + self.tn
    @property
    def accuracy(self):     return (self.tp + self.tn) / self.n if self.n else 0.0
    @property
    def said_down(self):    return (self.fn + self.tn) / self.n if self.n else 0.0

    @staticmethod
    def _div(a, b):         return a / b if b else 0.0
    @property
    def prec_up(self):      return self._div(self.tp, self.tp + self.fp)
    @property
    def rec_up(self):       return self._div(self.tp, self.tp + self.fn)
    @property
    def prec_dn(self):      return self._div(self.tn, self.tn + self.fn)
    @property
    def rec_dn(self):       return self._div(self.tn, self.tn + self.fp)
    @staticmethod
    def f1(p, r):           return 2 * p * r / (p + r) if (p + r) else 0.0
    @property
    def balanced_acc(self): return (self.rec_up + self.rec_dn) / 2
    @property
    def mcc(self):
        d = math.sqrt((self.tp + self.fp) * (self.tp + self.fn) *
                      (self.tn + self.fp) * (self.tn + self.fn))
        return (self.tp * self.tn - self.fp * self.fn) / d if d else 0.0


def merge(scores):
    out = Score()
    for s in scores:
        for f in ("tp","fn","fp","tn","err","sq","pct","bias",
                  "up_ret","dn_ret","up_n","dn_n","failed"):
            setattr(out, f, getattr(out, f) + getattr(s, f))
    return out


def backtest_one(ticker, closes, lstm_cls=LSTMPredictor):
    lin, lstm = Score(), Score()
    base_up = base_mom = base_n = 0
    naive_err = 0.0
    all_ret = 0.0
    days = list(range(MIN_HIST, len(closes) - HORIZON, STEP))

    for k, t in enumerate(days, 1):
        past   = closes[:t]
        window = past[-TRAIN_N:]
        today  = past[-1]
        actual = closes[t - 1 + HORIZON]
        went_up = actual > today

        # linear model
        lm = LinearPredictor(window=20, forecast=HORIZON)
        lm.train(window)
        lp = lm.predict(window)["predictions"][-1]
        lin.add(lp > today, went_up, lp, actual, today)

        # LSTM (trained from scratch on only the past, same seeds as the app)
        nn_model = lstm_cls()
        tr = nn_model.train(window)
        pr = nn_model.predict(window) if tr.get("ok") else {"ok": False}
        if pr.get("ok"):
            p = pr["predictions"][-1]
            lstm.add(p > today, went_up, p, actual, today)
        else:
            lstm.failed += 1

        # baselines
        base_n   += 1
        base_up  += went_up
        base_mom += ((today > past[-1 - HORIZON]) == went_up)
        naive_err += abs(today - actual)
        all_ret   += (actual / today - 1) * 100

        print(f"\r  {ticker}: test {k}/{len(days)}", end="", flush=True)
    print(f"\r  {ticker}: done ({len(days)} tests)        ")
    return lin, lstm, base_n, base_up, base_mom, naive_err, all_ret


def print_matrix(name, s):
    print(f"\n  {name}  (n={s.n}, accuracy {s.accuracy:.1%})")
    print(f"                        said UP   said DOWN")
    print(f"    really went UP     {s.tp:>7}   {s.fn:>9}")
    print(f"    really went DOWN   {s.fp:>7}   {s.tn:>9}")
    if s.tp + s.fp:
        print(f"    when it said UP, it was right   {s.tp/(s.tp+s.fp):.1%} of the time")
    if s.tn + s.fn:
        print(f"    when it said DOWN, it was right {s.tn/(s.tn+s.fn):.1%} of the time")


def main():
    print(f"Walk-forward backtest | every {STEP}th day | {HORIZON}-day horizon | "
          f"trains on last {TRAIN_N} prices")
    print("The LSTM trains 5 models at every test day, so this takes a while...\n")
    t0 = time.time()

    per = {}
    for tk in TICKERS:
        closes = load(tk)
        if not closes:
            print(f"  {tk}: no data/{tk}.csv  (run download_prices.py first)")
            continue
        if len(closes) < MIN_HIST + HORIZON + STEP:
            print(f"  {tk}: not enough history ({len(closes)} rows)")
            continue
        per[tk] = backtest_one(tk, closes)

    if not per:
        return

    # ── Per-stock table ───────────────────────────────────────────────────────
    print(f"\n{'Ticker':<7}{'Tests':>6}{'Linear':>9}{'LSTM':>8}{'AlwaysUP':>10}{'Momentum':>10}"
          f"{'LSTM says DOWN':>16}")
    for tk, (lin, lstm, n, up, mom, _, _r) in per.items():
        print(f"{tk:<7}{n:>6}{lin.accuracy:>9.1%}{lstm.accuracy:>8.1%}{up/n:>10.1%}"
              f"{mom/n:>10.1%}{lstm.said_down:>16.0%}")

    LIN  = merge([v[0] for v in per.values()])
    LSTM = merge([v[1] for v in per.values()])
    N    = sum(v[2] for v in per.values())
    UP   = sum(v[3] for v in per.values())
    MOM  = sum(v[4] for v in per.values())
    NAIVE_ERR = sum(v[5] for v in per.values())
    print(f"{'ALL':<7}{N:>6}{LIN.accuracy:>9.1%}{LSTM.accuracy:>8.1%}{UP/N:>10.1%}"
          f"{MOM/N:>10.1%}{LSTM.said_down:>16.0%}")

    margin = 1.96 * math.sqrt(0.25 / N)
    print(f"\nWith {N} tests, random luck alone moves any accuracy by about +/-{margin:.1%}.")
    print(f"Price error per test (lower is better): linear {LIN.err/LIN.n:.2f} | "
          f"LSTM {LSTM.err/max(LSTM.n,1):.2f} | 'price stays the same' {NAIVE_ERR/N:.2f}")
    if LSTM.failed:
        print(f"WARNING: LSTM failed to train on {LSTM.failed} test days (skipped).")

    # ── Confusion matrices ────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("CONFUSION MATRICES  (all stocks combined)")
    print("=" * 52)
    print_matrix("LINEAR model", LIN)
    print_matrix("LSTM ensemble", LSTM)

    # ── Full metrics table ────────────────────────────────────────────────────
    ALL_RET = sum(v[6] for v in per.values()) / N
    def avg(a, n): return a / n if n else float("nan")
    rows = [
        ("Tests (5-day-ahead calls)",           f"{N}",                         f"{N}"),
        ("Directional accuracy",                f"{LIN.accuracy:.1%}",          f"{LSTM.accuracy:.1%}"),
        ("Balanced accuracy",                   f"{LIN.balanced_acc:.1%}",      f"{LSTM.balanced_acc:.1%}"),
        ("Precision, UP calls",                 f"{LIN.prec_up:.1%}",           f"{LSTM.prec_up:.1%}"),
        ("Recall, UP (rises caught)",           f"{LIN.rec_up:.1%}",            f"{LSTM.rec_up:.1%}"),
        ("F1, UP",                              f"{Score.f1(LIN.prec_up, LIN.rec_up):.3f}", f"{Score.f1(LSTM.prec_up, LSTM.rec_up):.3f}"),
        ("Precision, DOWN calls",               f"{LIN.prec_dn:.1%}",           f"{LSTM.prec_dn:.1%}"),
        ("Recall, DOWN (falls caught)",         f"{LIN.rec_dn:.1%}",            f"{LSTM.rec_dn:.1%}"),
        ("F1, DOWN",                            f"{Score.f1(LIN.prec_dn, LIN.rec_dn):.3f}", f"{Score.f1(LSTM.prec_dn, LSTM.rec_dn):.3f}"),
        ("Matthews corr. (0 = no skill)",       f"{LIN.mcc:+.3f}",              f"{LSTM.mcc:+.3f}"),
        ("Share of calls saying DOWN",          f"{LIN.said_down:.0%}",         f"{LSTM.said_down:.0%}"),
        ("Mean abs. price error",               f"{avg(LIN.err, LIN.n):.2f}",   f"{avg(LSTM.err, LSTM.n):.2f}"),
        ("RMSE",                                f"{math.sqrt(avg(LIN.sq, LIN.n)):.2f}", f"{math.sqrt(avg(LSTM.sq, LSTM.n)):.2f}"),
        ("Mean abs. % error",                   f"{avg(LIN.pct, LIN.n):.2f}%",  f"{avg(LSTM.pct, LSTM.n):.2f}%"),
        ("Mean bias (pred - actual)",           f"{avg(LIN.bias, LIN.n):+.2f}", f"{avg(LSTM.bias, LSTM.n):+.2f}"),
        ("Avg real 5-day return after UP call", f"{avg(LIN.up_ret, LIN.up_n):+.2f}%", f"{avg(LSTM.up_ret, LSTM.up_n):+.2f}%"),
        ("Avg real 5-day return after DOWN call", f"{avg(LIN.dn_ret, LIN.dn_n):+.2f}%", f"{avg(LSTM.dn_ret, LSTM.dn_n):+.2f}%"),
    ]
    print("\n" + "=" * 62)
    print(f"{'Metric':<40}{'Linear':>10}{'LSTM':>12}")
    print("=" * 62)
    for name, a, b in rows:
        print(f"{name:<40}{a:>10}{b:>12}")
    print("-" * 62)
    print(f"{'Always-UP baseline accuracy':<40}{UP/N:>10.1%}")
    print(f"{'Momentum baseline accuracy':<40}{MOM/N:>10.1%}")
    print(f"{'Price error if you guess "no change"':<40}{NAIVE_ERR/N:>10.2f}")
    print(f"{'Avg real 5-day return, ALL days':<40}{ALL_RET:>+9.2f}%")
    print("=" * 62)
    print("A useful model has a HIGHER return after UP calls than after DOWN calls.")

    # ── Verdict ───────────────────────────────────────────────────────────────
    best_base = max(UP / N, MOM / N)
    print("\n" + "=" * 52)
    print("HOW TO READ THIS")
    print("=" * 52)
    print("- 50% is a coin flip. An edge only counts if it clears BOTH baselines")
    print(f"  (best baseline here: {best_base:.1%}) by more than the +/-{margin:.1%} luck band.")
    print("- 'LSTM says DOWN' far from ~45% means the model is biased to one answer.")
    print("- Report whatever these numbers say - an honest result is a good result.")

    # ── Save for the dashboard / README ───────────────────────────────────────
    def pack(s):
        return {"n": s.n, "accuracy": round(s.accuracy, 4),
                "matrix": {"tp": s.tp, "fn": s.fn, "fp": s.fp, "tn": s.tn},
                "mean_abs_error": round(s.err / s.n, 4) if s.n else None,
                "rmse": round(math.sqrt(s.sq / s.n), 4) if s.n else None,
                "precision_up": round(s.prec_up, 4), "recall_up": round(s.rec_up, 4),
                "precision_down": round(s.prec_dn, 4), "recall_down": round(s.rec_dn, 4),
                "balanced_accuracy": round(s.balanced_acc, 4), "mcc": round(s.mcc, 4),
                "avg_return_after_up_pct": round(s.up_ret / s.up_n, 4) if s.up_n else None,
                "avg_return_after_down_pct": round(s.dn_ret / s.dn_n, 4) if s.dn_n else None}
    with open(OUT_FILE, "w") as f:
        json.dump({"step_days": STEP, "horizon_days": HORIZON, "tests": N,
                   "always_up": round(UP / N, 4), "momentum": round(MOM / N, 4),
                   "naive_mean_abs_error": round(NAIVE_ERR / N, 4),
                   "luck_margin": round(margin, 4),
                   "linear": pack(LIN), "lstm": pack(LSTM),
                   "per_ticker": {tk: {"tests": v[2], "linear": round(v[0].accuracy, 4),
                                       "lstm": round(v[1].accuracy, 4)}
                                  for tk, v in per.items()}}, f, indent=2)
    print(f"\nSaved {OUT_FILE}   (took {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()