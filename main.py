# ── main.py (updated) ──────────────────────────────────────────────────────
#
# Run with:   uvicorn main:app --reload
# Docs at:    http://127.0.0.1:8000/docs
#
# NEW ML ENDPOINTS:
#   POST /ml/load                 → load FinBERT
#   GET  /ml/sentiment/{ticker}   → news sentiment
#   GET  /ml/predict/{ticker}     → LSTM price forecast
#   GET  /ml/signal/{ticker}      → BUY/SELL/HOLD signal
#   GET  /ml/signals/all          → all tickers at once
#   GET  /ml/anomalies            → recent anomaly alerts
# ───────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import random, sys, os

from trading_system import TradingSystem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml.ml_router import router as ml_router, set_price_source

app = FastAPI(
    title       = "Stock Trading System + ML",
    description = "DSA engine (heaps, queues, hashmaps) + FinBERT · LSTM · IsolationForest",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml_router)

TICKERS       = ["AAPL", "TSLA", "GOOGL", "INFY"]
BASE_PRICES   = {"AAPL": 182.0, "TSLA": 245.0, "GOOGL": 175.0, "INFY": 19.5}
STARTING_CASH = 50_000.0

# ── Real price history (optional) ────────────────────────────────────────────
# Run  python download_prices.py  once. It saves data/AAPL.csv etc.
# If those files exist we start from REAL prices; otherwise we fall back to
# the made-up random starting prices below.
import csv
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def load_closes(ticker: str, n: int = 250):
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        closes = [float(r["close"]) for r in csv.DictReader(f)]
    return closes[-n:] if len(closes) >= 30 else None

HISTORY = {t: load_closes(t) for t in TICKERS}
for _t, _c in HISTORY.items():
    if _c:
        BASE_PRICES[_t] = round(_c[-1], 2)   # order book is built around the latest real close
print("[TradeOS] price history:",
      {t: ("real, %d days" % len(c)) if c else "random fallback" for t, c in HISTORY.items()})

def make_system() -> TradingSystem:
    sys_ = TradingSystem(tickers=TICKERS, starting_cash=STARTING_CASH)
    for ticker, base in BASE_PRICES.items():
        book = sys_.order_books[ticker]
        for i in range(1, 6):
            book.place_order("SELL", price=round(base*(1+0.001*i),2), quantity=random.randint(50,300), rest=True)
        for i in range(1, 6):
            book.place_order("BUY",  price=round(base*(1-0.001*i),2), quantity=random.randint(50,300), rest=True)
        if HISTORY.get(ticker):
            for p in HISTORY[ticker]:                 # REAL history
                sys_.price_tracker.record(ticker, p)
        else:
            price = base                              # random fallback
            for _ in range(60):
                price = round(max(1, price + random.uniform(-0.4, 0.4)), 2)
                sys_.price_tracker.record(ticker, price)
    return sys_

system = make_system()

# Give the ML layer a window onto THIS system's own price history.
# (The lambdas read the global `system` each time, so they keep working
#  after /reset replaces it with a fresh one.)
set_price_source(
    lambda t: system.price_tracker.history(t, 300),
    lambda t: system.price_tracker.count(t),
)

class OrderRequest(BaseModel):
    side:   Literal["BUY", "SELL"]
    ticker: Literal["AAPL","TSLA","GOOGL","INFY"]
    price:  float = Field(..., ge=0)
    qty:    int   = Field(..., ge=1, le=10000)

class OrderResponse(BaseModel):
    ok:      bool
    status:  str
    message: str
    trades:  int = 0

@app.get("/tickers", tags=["Market"])
def get_tickers():
    return {"tickers": TICKERS, "prices": {t: system.price_tracker.latest(t) for t in TICKERS}}

@app.get("/snapshot/{ticker}", tags=["Market"])
def get_snapshot(ticker: str):
    if ticker not in TICKERS:
        raise HTTPException(404, f"Ticker {ticker} not found")
    snap    = system.snapshot(ticker)
    history = system.price_tracker.history(ticker)
    base    = BASE_PRICES[ticker]
    latest  = snap["last_price"] or base
    change  = round(latest - base, 2)
    pct     = round((change / base) * 100, 2)
    return {**snap, "change": change, "change_pct": pct, "history": history[-20:]}

@app.get("/orderbook/{ticker}", tags=["Market"])
def get_orderbook(ticker: str):
    if ticker not in TICKERS:
        raise HTTPException(404, f"Ticker {ticker} not found")
    book = system.order_books[ticker]
    bids = [{"price": o.price, "qty": o.quantity, "id": o.order_id} for o in book.bids.top_n(5)]
    asks = [{"price": o.price, "qty": o.quantity, "id": o.order_id} for o in book.asks.top_n(5)]
    return {"ticker": ticker, "bids": bids, "asks": asks, "spread": book.spread()}

@app.post("/order", tags=["Trading"], response_model=OrderResponse)
def place_order(req: OrderRequest):
    if req.price == 0:
        # MARKET ORDER: "fill me at the best prices available".
        # We allow walking up/down the book, but never worse than 5% from the
        # best price (a safety cap, like slippage protection on real exchanges).
        book = system.order_books[req.ticker]
        if req.side == "BUY":
            best  = book.best_ask() or BASE_PRICES[req.ticker]
            price = round(best * 1.05, 2)
        else:
            best  = book.best_bid() or BASE_PRICES[req.ticker]
            price = round(best * 0.95, 2)
    else:
        price = req.price
    result = system.place_order(req.side, req.ticker, price, req.qty)
    if not result["ok"]:
        raise HTTPException(400, result["reason"])
    trades  = result.get("trades", 0)
    status  = result.get("status", "UNKNOWN")
    message = result.get("message", "Order processed")
    return OrderResponse(ok=True, status=status, message=message, trades=trades)

@app.get("/portfolio", tags=["Trading"])
def get_portfolio():
    prices  = {t: system.price_tracker.latest(t) or BASE_PRICES[t] for t in TICKERS}
    pnl     = system.portfolio.unrealised_pnl(prices)
    total   = system.portfolio.total_value(prices)
    holdings_out = {}
    for ticker, h in system.portfolio.holdings.items():
        holdings_out[ticker] = {
            "qty":           h["qty"],
            "avg_cost":      round(h["avg_cost"], 2),
            "current_price": prices.get(ticker, 0),
            "value":         round(h["qty"] * prices.get(ticker, h["avg_cost"]), 2),
            "pnl":           pnl.get(ticker, {}).get("pnl", 0),
            "pnl_pct":       pnl.get(ticker, {}).get("pnl_pct", 0),
        }
    return {"cash": round(system.portfolio.cash,2), "total_value": total,
            "realised_pnl": round(system.portfolio.realised_pnl,2),
            "trade_count": system.portfolio.trade_count, "holdings": holdings_out}

@app.get("/trades", tags=["Trading"])
def get_trades():
    return {"total": len(system.trade_log), "trades": system.trade_log.all()[-20:]}

@app.post("/reset", tags=["System"])
def reset_system():
    global system
    system = make_system()
    return {"ok": True, "message": "System reset"}

@app.get("/", tags=["System"])
def root():
    return {
        "message":  "Trading System + ML API v2.0",
        "docs":     "http://127.0.0.1:8000/docs",
        "ml_endpoints": [
            "POST /ml/load",
            "GET  /ml/signal/{ticker}",
            "GET  /ml/signals/all",
            "GET  /ml/sentiment/{ticker}",
            "GET  /ml/predict/{ticker}",
            "GET  /ml/anomalies",
        ]
    }