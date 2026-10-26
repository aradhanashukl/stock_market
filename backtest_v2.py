"""
backtest_v2.py  -  a more careful test of "can we predict the next 5 days?"

    python backtest_v2.py           # VALIDATION mode: look at it, change things, repeat
    python backtest_v2.py final     # FINAL TEST: run this ONCE at the very end

WHAT IS DIFFERENT FROM backtest_lstm.py
  1. It predicts the 5-day RETURN direction (up/down), not the price level.
     Returns are comparable across stocks, so all stocks are pooled into
     one big training set instead of 4 tiny ones.
  2. It uses FEATURES, not just raw prices: recent returns, volatility,
     distance from moving averages, RSI, volume.
  3. Two simple, hard-to-overfit models: Logistic Regression and a small
     Gradient-Boosting model (hyper-parameters are FIXED below on purpose).
  4. Time split with a gap, so nothing from the future leaks into training:

        [ ......... TRAIN ......... ][gap][ VALIDATION ][gap][ FINAL TEST ]
                                           (last-but-one year)  (last year)

     Normal mode trains on TRAIN and scores VALIDATION. You may look at
     that as often as you like and improve the features.
     'final' mode trains on TRAIN+VALIDATION and scores FINAL TEST.
     Run it once. If you change the code afterwards because of that
     number, you have turned the test set into a second validation set.
  5. Honest error bars: the 95% interval resamples whole DAYS (stocks move
     together, so 28 stocks on one day are not 28 independent tests).
     A control: the same model trained on SHUFFLED labels. If it scores
     well, something is leaking. It should score about like "always UP".
  6. Tests use every 5th day per stock, so two tests never share days.

Needs data/*.csv from download_prices_v2.py (columns: date, close, volume).
Saves backtest_v2_results.json.
"""
import csv
import glob
import json
import math
import os
import sys
import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_FILE = os.path.join(HERE, "backtest_v2_results.json")

HORIZON   = 5        # predict 5 trading days ahead
LOOKBACK  = 60       # history a row needs before it can be used
EVAL_STEP = 5        # one test per stock every 5 days (no overlapping tests)
SPLIT_LEN = 252      # ~1 trading year for validation, and again for final test
FINAL     = "final" in sys.argv


# ── 1. Load data ──────────────────────────────────────────────────────────────
def load_stock(path):
    dates, close, vol = [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            dates.append(r["date"])
            close.append(float(r["close"]))
            vol.append(float(r["volume"]) if r.get("volume") not in (None, "") else float("nan"))
    c, v = np.array(close), np.array(vol)
    if np.isnan(v).any() or (v <= 0).mean() > 0.05:
        v = None                                   # no usable volume column
    return dates, c, v


# ── 2. Features (each uses ONLY data up to and including day t) ───────────────
FEATURES = ["ret1", "ret5", "ret10", "ret20", "ret60", "vol10", "vol20",
            "dist_sma20", "dist_sma50", "rsi14", "vol_ratio"]

def build_features(c, v):
    n = len(c)
    nan = np.full(n, np.nan)

    def past_ret(k):
        out = nan.copy(); out[k:] = c[k:] / c[:-k] - 1; return out

    d1 = nan.copy(); d1[1:] = c[1:] / c[:-1] - 1          # daily returns

    def roll(fn, x, k, start):
        out = nan.copy()
        for t in range(start, n):
            out[t] = fn(x[t - k + 1: t + 1])
        return out

    sma20 = roll(np.mean, c, 20, 19)
    sma50 = roll(np.mean, c, 50, 49)
    gain  = np.where(np.nan_to_num(d1) > 0, np.nan_to_num(d1), 0.0)
    loss  = np.where(np.nan_to_num(d1) < 0, -np.nan_to_num(d1), 0.0)
    g14, l14 = roll(np.mean, gain, 14, 14), roll(np.mean, loss, 14, 14)
    rsi = 100 - 100 / (1 + g14 / np.where(l14 == 0, 1e-9, l14))

    if v is not None:
        vr = roll(np.mean, v, 20, 19)
        vol_ratio = np.log(v / vr)
    else:
        vol_ratio = np.zeros(n)                       # constant = no information

    cols = [past_ret(1), past_ret(5), past_ret(10), past_ret(20), past_ret(60),
            roll(np.std, d1, 10, 10), roll(np.std, d1, 20, 20),
            c / sma20 - 1, c / sma50 - 1, rsi, vol_ratio]
    return np.column_stack(cols)


def build_dataset():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        sys.exit("No data/*.csv found. Run download_prices_v2.py first.")
    X, y, fwd, d_t, d_end, tick, pos = [], [], [], [], [], [], []
    for path in files:
        tk = os.path.splitext(os.path.basename(path))[0]
        dates, c, v = load_stock(path)
        n = len(c)
        if n < LOOKBACK + HORIZON + 300:
            print(f"  skipping {tk}: only {n} rows")
            continue
        F = build_features(c, v)
        for t in range(LOOKBACK, n - HORIZON):
            row = F[t]
            if np.isnan(row).any():
                continue
            r = c[t + HORIZON] / c[t] - 1
            X.append(row); y.append(int(r > 0)); fwd.append(r * 100)
            d_t.append(dates[t]); d_end.append(dates[t + HORIZON])
            tick.append(tk); pos.append(t)
    return (np.array(X), np.array(y), np.array(fwd), np.array(d_t),
            np.array(d_end), np.array(tick), np.array(pos))


# ── 3. Scoring ────────────────────────────────────────────────────────────────
def score(said_up, went_up, fwd):
    said_up, went_up = np.asarray(said_up, bool), np.asarray(went_up, bool)
    tp = int((said_up & went_up).sum());  fn = int((~said_up & went_up).sum())
    fp = int((said_up & ~went_up).sum()); tn = int((~said_up & ~went_up).sum())
    n = tp + fn + fp + tn
    div = lambda a, b: a / b if b else 0.0
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    up_ret = fwd[said_up].mean() if said_up.any() else float("nan")
    dn_ret = fwd[~said_up].mean() if (~said_up).any() else float("nan")
    return {"n": n, "accuracy": (tp + tn) / n,
            "balanced_acc": (div(tp, tp + fn) + div(tn, tn + fp)) / 2,
            "mcc": (tp * tn - fp * fn) / d if d else 0.0,
            "precision_up": div(tp, tp + fp), "share_said_up": said_up.mean(),
            "ret_after_up": up_ret, "ret_after_down": dn_ret,
            "spread": up_ret - dn_ret,
            "matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn}}


def bootstrap_diff(days, model_ok, base_ok, n_boot=2000, seed=1):
    """95% interval for (model accuracy - baseline accuracy), resampling WHOLE
    DAYS. Stocks move together on the same day, so one day is one 'unit of
    luck', not 28 independent tests. This is the honest way to size the noise."""
    uniq, inv = np.unique(days, return_inverse=True)
    m = np.bincount(inv, weights=model_ok.astype(float))
    b = np.bincount(inv, weights=base_ok.astype(float))
    c = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        diffs[i] = (m[pick].sum() - b[pick].sum()) / c[pick].sum()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def make_models():
    # Hyper-parameters are FIXED here on purpose - do not tune them against
    # the final test. Small and shallow = hard to memorise noise.
    return {
        "Logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(C=0.1, max_iter=1000)),
        "GradBoost": GradientBoostingClassifier(n_estimators=100, max_depth=2,
                                                learning_rate=0.05, subsample=0.8,
                                                random_state=42),
    }


# ── 4. Main ───────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("Building features ...")
    X, y, fwd, d_t, d_end, tick, pos = build_dataset()
    all_dates = sorted(set(d_t))
    if len(all_dates) < 3 * SPLIT_LEN + 300:
        sys.exit(f"Need more history: have {len(all_dates)} days, "
                 f"need {3 * SPLIT_LEN + 300}. Use download_prices_v2.py (10 years).")
    test_start = all_dates[-SPLIT_LEN]
    val_start  = all_dates[-2 * SPLIT_LEN]

    if FINAL:
        mode = "FINAL TEST  (run once - do not tune after this)"
        train_m = d_end < test_start                 # label must be known before test
        eval_m  = d_t >= test_start
    else:
        mode = "VALIDATION  (safe to look at and improve)"
        train_m = d_end < val_start
        eval_m  = (d_t >= val_start) & (d_end < test_start)

    # keep every EVAL_STEP-th row per stock -> tests do not overlap
    keep = np.zeros(len(y), bool)
    for tk in set(tick[eval_m]):
        idx = np.where(eval_m & (tick == tk))[0]
        keep[idx[::EVAL_STEP]] = True
    eval_m = keep

    Xtr, ytr = X[train_m], y[train_m]
    Xev, yev, fev = X[eval_m], y[eval_m], fwd[eval_m]
    n_stocks = len(set(tick))
    n_days_ev = len(set(d_t[eval_m]))
    print(f"\nMode: {mode}")
    print(f"Stocks: {n_stocks} | train rows: {len(ytr):,} "
          f"(labels known up to {max(d_end[train_m])}) | tests: {len(yev):,} "
          f"({min(d_t[eval_m])} .. {max(d_t[eval_m])})")

    # baselines
    always_up = float(yev.mean())
    momentum  = float(((Xev[:, FEATURES.index("ret5")] > 0).astype(int) == yev).mean())
    best_base = max(always_up, momentum)
    margin    = 1.96 * math.sqrt(0.25 / len(yev))      # optimistic: assumes independent tests
    base_ok   = ((np.ones(len(yev), int) == yev) if always_up >= momentum
                 else ((Xev[:, FEATURES.index("ret5")] > 0).astype(int) == yev))
    days_ev   = d_t[eval_m]

    results, models = {}, make_models()
    ok, ci = {}, {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xev) == 1
        results[name] = score(pred, yev == 1, fev)
        ok[name] = (pred == (yev == 1))
        ci[name] = bootstrap_diff(days_ev, ok[name], base_ok)

    # control: train on SHUFFLED labels. Should NOT beat the baselines.
    rng = np.random.default_rng(0)
    ctrl = make_models()["GradBoost"]
    ctrl.fit(Xtr, rng.permutation(ytr))
    results["Control (shuffled)"] = score(ctrl.predict(Xev) == 1, yev == 1, fev)

    # ── print ────────────────────────────────────────────────────────────────
    cols = list(results)
    print("\n" + "=" * (34 + 14 * len(cols)))
    print(f"{'Metric':<34}" + "".join(f"{c:>14}" for c in cols))
    print("=" * (34 + 14 * len(cols)))
    def row(label, key, fmt):
        print(f"{label:<34}" + "".join(f"{fmt.format(results[c][key]):>14}" for c in cols))
    row("Tests",                        "n",              "{:d}")
    row("Directional accuracy",         "accuracy",       "{:.1%}")
    row("Balanced accuracy",            "balanced_acc",   "{:.1%}")
    row("Matthews corr. (0 = no skill)","mcc",            "{:+.3f}")
    row("Precision, UP calls",          "precision_up",   "{:.1%}")
    row("Share of calls saying UP",     "share_said_up",  "{:.0%}")
    row("Avg real 5d return after UP",  "ret_after_up",   "{:+.2f}%")
    row("Avg real 5d return after DOWN","ret_after_down", "{:+.2f}%")
    row("Spread (UP minus DOWN)",       "spread",         "{:+.2f}%")
    print("-" * (34 + 14 * len(cols)))
    print(f"{'Always-UP baseline':<34}{always_up:>14.1%}")
    print(f"{'Momentum baseline (last 5d)':<34}{momentum:>14.1%}")
    print("=" * (34 + 14 * len(cols)))
    print(f"Luck band if every test were independent: +/-{margin:.1%} (optimistic).")
    for name in models:
        lo, hi = ci[name]
        print(f"95% interval, {name:<9} accuracy minus best baseline: "
              f"[{lo:+.1%}, {hi:+.1%}]   (resampling whole days)")

    gb = models["GradBoost"]
    imp = sorted(zip(FEATURES, gb.feature_importances_), key=lambda z: -z[1])[:5]
    print("Top features (GradBoost): " + ", ".join(f"{k} {v:.2f}" for k, v in imp))

    print("\nVERDICT")
    for name in models:
        acc = results[name]["accuracy"]
        lo, hi = ci[name]
        if lo > 0:
            msg = "beats the best baseline, and the 95% interval stays above zero"
        elif acc > best_base:
            msg = "above the baseline, but the interval includes zero - NOT proven"
        else:
            msg = "does not beat the best baseline"
        print(f"  {name:<10} {acc:.1%} vs best baseline {best_base:.1%}: {msg}")
    cacc = results["Control (shuffled)"]["accuracy"]
    flag = "OK" if abs(cacc - always_up) < 0.04 else "SUSPICIOUS - check for leakage"
    print(f"  Control    {cacc:.1%} (should be near always-UP {always_up:.1%}): {flag}")
    if not FINAL:
        print("\nThis was VALIDATION. When you are done improving, run:  "
              "python backtest_v2.py final   (once)")

    with open(OUT_FILE, "w") as f:
        json.dump({"mode": "final" if FINAL else "validation", "stocks": n_stocks,
                   "tests": int(len(yev)), "distinct_days": n_days_ev,
                   "horizon_days": HORIZON, "always_up": round(always_up, 4),
                   "momentum": round(momentum, 4),
                   "luck_margin_if_independent": round(margin, 4),
                   "ci95_vs_best_baseline": {k: [round(v[0], 4), round(v[1], 4)] for k, v in ci.items()},
                   "results": {k: {a: (round(b, 4) if isinstance(b, float) else b)
                                   for a, b in v.items()} for k, v in results.items()}},
                  f, indent=2)
    print(f"\nSaved {OUT_FILE}   (took {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()