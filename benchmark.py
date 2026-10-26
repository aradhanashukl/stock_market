"""
benchmark.py  -  measures TradeOS and prints ONE results table.

    python benchmark.py            # full run
    python benchmark.py quick      # smaller run (faster)

Every number is measured on YOUR machine, so you can say "I measured it".
Run  python backtest_lstm.py  first if you want the forecast rows filled in
(it writes backtest_results.json, which this script reads).

What it measures
  1. Matching engine   : orders/second + latency through the heap order book
  2. Full order path   : match + portfolio + trade log, per order
  3. Anomaly detector  : precision / recall / F1 on INJECTED anomalies
                         (synthetic stream with known labels, 20 random seeds)
  4. News retrieval    : search time + how many results are about the company
  5. Signal latency    : full BUY/SELL/HOLD call (FinBERT + LSTM + RAG)
  6. Data scale        : prices, stocks, headlines
  7. Forecast rows     : read from backtest_results.json
"""
import csv
import json
import math
import os
import random
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
QUICK = "quick" in sys.argv
OUT = os.path.join(HERE, "benchmark_results.json")
TICKERS = ["AAPL", "TSLA", "GOOGL", "INFY"]

rows = []          # (metric, result) for the final table
detail = {}        # raw numbers saved to json


def add(metric, result):
    rows.append((metric, result))


def pct(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(len(vals) * p / 100))]


def section(title):
    print(f"\n-- {title} " + "-" * max(1, 56 - len(title)))


# ═════════════════════════════════════════════════════════════════════════════
# 1 + 2. Matching engine and full order path
# ═════════════════════════════════════════════════════════════════════════════
def bench_engine():
    from trading_system import OrderBook, TradingSystem
    section("1. Matching engine (heaps)")
    n = 5_000 if QUICK else 30_000
    rnd = random.Random(7)
    base = 100.0

    book = OrderBook("BENCH")
    for i in range(1, 1001):                       # 1000 resting orders each side
        book.place_order("SELL", round(base * (1 + 0.0002 * i), 4), rnd.randint(50, 300), rest=True)
        book.place_order("BUY",  round(base * (1 - 0.0002 * i), 4), rnd.randint(50, 300), rest=True)

    lat, trades = [], 0
    t_all = time.perf_counter()
    for _ in range(n):
        side  = rnd.choice(["BUY", "SELL"])
        price = round(base * (1 + rnd.uniform(-0.25, 0.25) / 100 * 10), 4)   # some cross, some rest
        t0 = time.perf_counter_ns()
        r = book.place_order(side, price, rnd.randint(1, 100), rest=True)
        lat.append((time.perf_counter_ns() - t0) / 1000)                      # microseconds
        trades += len(r["trades"])
    total = time.perf_counter() - t_all
    ops = n / total
    print(f"  {n:,} orders in {total:.2f}s  ->  {ops:,.0f} orders/sec, {trades:,} trades")
    print(f"  latency per order (us): median {st.median(lat):.1f} | p95 {pct(lat,95):.1f} | p99 {pct(lat,99):.1f}")
    add("Order-book throughput", f"{ops:,.0f} orders/sec")
    add("Order-book latency (median / p99)", f"{st.median(lat):.0f} us / {pct(lat,99):.0f} us")
    detail["engine"] = {"orders": n, "orders_per_sec": round(ops), "trades": trades,
                        "median_us": round(st.median(lat), 1), "p95_us": round(pct(lat, 95), 1),
                        "p99_us": round(pct(lat, 99), 1)}

    # full pipeline: match + portfolio + price tracker + trade log (anomaly hook off)
    section("2. Full order path (match + portfolio + log)")
    m = 2_000 if QUICK else 10_000
    sysm = TradingSystem(tickers=["BENCH"], starting_cash=1e12)
    sysm._anomaly_manager = None                    # measure the engine, not the ML hook
    sysm.portfolio.holdings["BENCH"] = {"qty": 10**9, "avg_cost": base}
    bk, lat2 = sysm.order_books["BENCH"], []
    for _ in range(m):
        side = rnd.choice(["BUY", "SELL"])
        other = "SELL" if side == "BUY" else "BUY"
        bk.place_order(other, base, 10, rest=True)  # a counter-party to trade against
        t0 = time.perf_counter_ns()
        sysm.place_order(side, "BENCH", base, 10)
        lat2.append((time.perf_counter_ns() - t0) / 1000)
    print(f"  {m:,} full orders: median {st.median(lat2):.1f} us | p99 {pct(lat2,99):.1f} us")
    add("Full order path (median / p99)", f"{st.median(lat2):.0f} us / {pct(lat2,99):.0f} us")
    detail["order_path"] = {"orders": m, "median_us": round(st.median(lat2), 1),
                            "p99_us": round(pct(lat2, 99), 1)}


# ═════════════════════════════════════════════════════════════════════════════
# 3. Anomaly detector: precision / recall / F1 on injected anomalies
# ═════════════════════════════════════════════════════════════════════════════
def make_stream(seed, n=300, rate=0.06):
    """Normal trades with labelled anomalies injected at random positions."""
    rnd = random.Random(seed)
    base, price = 182.0, 182.0
    events, kinds = [], []
    from ml.anomaly import TradeEvent
    for i in range(n):
        price = base + rnd.gauss(0, 0.3)
        qty, spread, kind = rnd.randint(10, 100), rnd.uniform(0.05, 0.2), None
        if i > 40 and rnd.random() < rate:                   # leave a warm-up period
            kind = rnd.choice(["spike", "huge_order", "wide_spread", "crash"])
            if kind == "spike":        price = base + rnd.uniform(4, 12)
            elif kind == "crash":      price = base - rnd.uniform(4, 12)
            elif kind == "huge_order": qty = rnd.randint(2000, 8000)
            else:                      spread = rnd.uniform(2.5, 6.0)
        events.append(TradeEvent("BENCH", round(price, 2), qty, round(spread, 3),
                                 rnd.choice(["BUY", "SELL"])))
        kinds.append(kind)
    return events, kinds


def bench_anomaly():
    from ml.anomaly import AnomalyDetector
    section("3. Anomaly detector (IsolationForest) on injected anomalies")
    seeds = range(5 if QUICK else 20)
    P, R, F, per_kind = [], [], [], {}
    tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for sd in seeds:
        events, kinds = make_stream(sd)
        det = AnomalyDetector("BENCH")
        tp = fp = fn = tn = 0
        for i, (ev, kind) in enumerate(zip(events, kinds)):
            flagged = det.process(ev) is not None
            if i < 40:                                       # model still warming up
                continue
            if kind and flagged:      tp += 1; per_kind.setdefault(kind, [0, 0])[0] += 1; per_kind[kind][1] += 1
            elif kind and not flagged: fn += 1; per_kind.setdefault(kind, [0, 0])[1] += 1
            elif flagged:             fp += 1
            else:                     tn += 1
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        P.append(p); R.append(r); F.append(f)
        for k, v in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)):
            tot[k] += v
    sd_ = lambda x: st.pstdev(x) if len(x) > 1 else 0.0
    print(f"  {len(P)} random streams of 300 trades, ~6% injected anomalies")
    print(f"  precision {st.mean(P):.1%} (+/-{sd_(P):.1%}) | recall {st.mean(R):.1%} (+/-{sd_(R):.1%}) "
          f"| F1 {st.mean(F):.3f} (+/-{sd_(F):.3f})")
    print(f"  totals: caught {tot['tp']}, missed {tot['fn']}, false alarms {tot['fp']}, "
          f"correctly ignored {tot['tn']}")
    for k, (c, n) in sorted(per_kind.items()):
        print(f"    {k:<12} caught {c}/{n}  ({c/n:.0%})")
    add("Anomaly detection precision", f"{st.mean(P):.1%}")
    add("Anomaly detection recall", f"{st.mean(R):.1%}")
    add("Anomaly detection F1", f"{st.mean(F):.3f}")
    add("False alarms on normal trades", f"{tot['fp'] / max(tot['fp'] + tot['tn'], 1):.1%}")
    detail["anomaly"] = {"streams": len(P), "precision": round(st.mean(P), 4),
                         "recall": round(st.mean(R), 4), "f1": round(st.mean(F), 4),
                         "totals": tot,
                         "recall_by_type": {k: round(c / n, 3) for k, (c, n) in per_kind.items()},
                         "note": "synthetic stream, injected anomalies, first 40 trades excluded"}


# ═════════════════════════════════════════════════════════════════════════════
# 4. News retrieval (RAG)
# ═════════════════════════════════════════════════════════════════════════════
QUERIES = ["earnings results and guidance", "analyst upgrade or downgrade",
           "price target change", "risks and lawsuits", "new product or deal announcement"]


def bench_rag():
    import ml.rag as rag_mod
    from ml.rag import rag
    section("4. News retrieval (RAG)")
    s = rag.load()
    if not s["ready"]:
        raise RuntimeError(s["error"])
    rag.retrieve("AAPL", k=8)                                # warm-up (loads embedding model)

    times, on_topic, on_topic_sem = [], [], []
    for tk in TICKERS:
        for q in QUERIES:
            t0 = time.perf_counter()
            hits = rag.retrieve(tk, query=q, k=8)
            times.append((time.perf_counter() - t0) * 1000)
            if hits:
                on_topic.append(sum(h["about_company"] for h in hits) / len(hits))
    saved = rag_mod.MENTION_BONUS
    rag_mod.MENTION_BONUS = 0.0                              # pure semantic search, no mention bonus
    for tk in TICKERS:
        for q in QUERIES:
            hits = rag.retrieve(tk, query=q, k=8)
            if hits:
                on_topic_sem.append(sum(h["about_company"] for h in hits) / len(hits))
    rag_mod.MENTION_BONUS = saved

    print(f"  index: {s['headlines_indexed']:,} headlines  {s['per_ticker']}")
    print(f"  search time: median {st.median(times):.0f} ms | p95 {pct(times,95):.0f} ms")
    print(f"  results that are about the company: {st.mean(on_topic):.0%} with the mention bonus "
          f"vs {st.mean(on_topic_sem):.0%} with plain semantic search")
    add("News headlines indexed", f"{s['headlines_indexed']:,}")
    add("News search time (median)", f"{st.median(times):.0f} ms")
    add("Retrieved headlines on-topic (hybrid)", f"{st.mean(on_topic):.0%}")
    add("Retrieved headlines on-topic (semantic only)", f"{st.mean(on_topic_sem):.0%}")
    detail["rag"] = {"indexed": s["headlines_indexed"], "median_ms": round(st.median(times), 1),
                     "p95_ms": round(pct(times, 95), 1),
                     "on_topic_hybrid": round(st.mean(on_topic), 3),
                     "on_topic_semantic_only": round(st.mean(on_topic_sem), 3),
                     "queries": len(times)}


# ═════════════════════════════════════════════════════════════════════════════
# 5. Signal latency (FinBERT + LSTM + RAG together)
# ═════════════════════════════════════════════════════════════════════════════
def load_closes(t):
    path = os.path.join(HERE, "data", f"{t}.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return [float(r["close"]) for r in csv.DictReader(f)][-250:]


def bench_signal():
    from ml.sentiment import analyzer
    from ml.signal_engine import signal_engine
    from ml.rag import rag
    section("5. Signal latency (FinBERT + LSTM + RAG)")
    closes = load_closes("AAPL")
    if not closes:
        raise RuntimeError("data/AAPL.csv missing - run download_prices.py")
    analyzer.load()
    rag.load()
    signal_engine.generate("AAPL", closes, version=250)       # warm-up: loads/caches everything

    # cold = includes training the 5-model LSTM ensemble
    signal_engine._trained_at.clear()
    t0 = time.perf_counter()
    signal_engine.generate("AAPL", closes, version=250)
    cold = time.perf_counter() - t0

    warm = []
    for _ in range(5 if QUICK else 15):
        t0 = time.perf_counter()
        signal_engine.generate("AAPL", closes, version=250)
        warm.append((time.perf_counter() - t0) * 1000)
    print(f"  first call after new data (trains LSTM x5): {cold:.2f}s")
    print(f"  repeat calls: median {st.median(warm):.0f} ms | p95 {pct(warm,95):.0f} ms")
    add("Signal latency, with LSTM retrain", f"{cold:.1f} s")
    add("Signal latency, models ready (median)", f"{st.median(warm):.0f} ms")
    detail["signal"] = {"cold_s": round(cold, 2), "warm_median_ms": round(st.median(warm), 1),
                        "warm_p95_ms": round(pct(warm, 95), 1), "model": analyzer._loaded and "FinBERT" or "rule-based"}


# ═════════════════════════════════════════════════════════════════════════════
# 6 + 7. Data scale and forecast rows
# ═════════════════════════════════════════════════════════════════════════════
def bench_scale_and_backtest():
    section("6. Data scale")
    days = {}
    for t in TICKERS:
        p = os.path.join(HERE, "data", f"{t}.csv")
        if os.path.exists(p):
            with open(p) as f:
                days[t] = sum(1 for _ in f) - 1
    if days:
        print(f"  real daily prices: {days}")
        add("Real price history", f"{min(days.values()):,} days x {len(days)} stocks (Yahoo Finance)")
        detail["price_days"] = days

    section("7. Forecast evaluation")
    # (a) the rigorous test: pooled stocks, held-out final year, day-level error bars
    p2 = os.path.join(HERE, "backtest_v2_results.json")
    if os.path.exists(p2):
        d = json.load(open(p2))
        res = {k: v for k, v in d["results"].items() if not k.startswith("Control")}
        best = max(res, key=lambda k: res[k]["accuracy"])
        lo, hi = d["ci95_vs_best_baseline"][best]
        tag = "final held-out year" if d["mode"] == "final" else "VALIDATION period (not the final test)"
        base = max(d["always_up"], d["momentum"])
        print(f"  [{tag}] {d['tests']} tests, {d['stocks']} stocks, best model: {best}")
        print(f"  accuracy {res[best]['accuracy']:.1%} vs best baseline {base:.1%}; "
              f"95% interval for the difference [{lo:+.1%}, {hi:+.1%}]")
        add(f"Forecast test set ({tag})", f"{d['tests']:,} tests, {d['stocks']} stocks")
        add(f"Forecast accuracy ({best})", f"{res[best]['accuracy']:.1%}")
        add("Best naive baseline (always-UP / momentum)", f"{base:.1%}")
        add("95% interval, model minus baseline", f"[{lo:+.1%}, {hi:+.1%}]")
        add("Balanced accuracy (50% = no skill)", f"{res[best]['balanced_acc']:.1%}")
        add("Share of calls saying UP", f"{res[best]['share_said_up']:.0%}")
        detail["forecast_v2"] = {"mode": d["mode"], "tests": d["tests"], "best": best,
                                 "accuracy": res[best]["accuracy"], "baseline": base,
                                 "ci95": [lo, hi]}
    else:
        print("  backtest_v2_results.json not found - run  python backtest_v2.py final")

    # (b) the LSTM the app actually uses (4 stocks, walk-forward)
    p = os.path.join(HERE, "backtest_results.json")
    if os.path.exists(p):
        d = json.load(open(p))
        print(f"  LSTM walk-forward: {d['tests']} tests, luck margin +/-{d['luck_margin']:.1%}")
        add("LSTM walk-forward tests (4 stocks)", f"{d['tests']}")
        add("LSTM directional accuracy", f"{d['lstm']['accuracy']:.1%}")
        add("Linear model directional accuracy", f"{d['linear']['accuracy']:.1%}")
        detail["forecast_lstm"] = {"tests": d["tests"], "linear": d["linear"]["accuracy"],
                                   "lstm": d["lstm"]["accuracy"]}
    else:
        print("  LSTM rows skipped - run  python backtest_lstm.py quick  to add them")


# ═════════════════════════════════════════════════════════════════════════════
def main():
    print(f"TradeOS benchmark ({'quick' if QUICK else 'full'} mode)")
    for fn in (bench_engine, bench_anomaly, bench_rag, bench_signal, bench_scale_and_backtest):
        try:
            fn()
        except Exception as e:                  # one failing section must not hide the others
            print(f"  SKIPPED ({fn.__name__}): {e}")
            add(fn.__name__.replace("bench_", "").replace("_", " ") + " (skipped)", str(e)[:60])

    w = max(len(m) for m, _ in rows) + 2
    print("\n" + "=" * (w + 34))
    print(f"{'Metric':<{w}}Result")
    print("=" * (w + 34))
    for m, r in rows:
        print(f"{m:<{w}}{r}")
    print("=" * (w + 34))
    with open(OUT, "w") as f:
        json.dump({"mode": "quick" if QUICK else "full", "table": rows, "detail": detail}, f, indent=2)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()