# ── ml/ml_router.py ────────────────────────────────────────────────────────
#
# FastAPI router that exposes all ML features as REST endpoints.
# Mounted into main.py under the /ml prefix.
#
# ENDPOINTS:
#   GET  /ml/signal/{ticker}      → full BUY/SELL/HOLD signal
#   GET  /ml/sentiment/{ticker}   → FinBERT sentiment for all headlines
#   GET  /ml/predict/{ticker}     → LSTM next-5 price predictions
#   GET  /ml/anomalies            → recent anomaly alerts
#   POST /ml/load                 → load FinBERT model (call once on startup)
# ───────────────────────────────────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi     import APIRouter, HTTPException
from ml.sentiment    import analyzer      as sentiment_analyzer
from ml.predictor    import manager       as predictor_manager
from ml.anomaly      import anomaly_manager, TradeEvent
from ml.signal_engine import signal_engine
from ml.rag           import rag

router = APIRouter(prefix="/ml", tags=["ML"])

TICKERS = ["AAPL", "TSLA", "GOOGL", "INFY"]
MIN_POINTS = 25   # smallest history the models need

# ── Where the prices come from ────────────────────────────────────────────────
# main.py calls set_price_source() once, handing us two small functions:
#   history_fn(ticker) -> list of this system's recent prices
#   version_fn(ticker) -> how many prices were ever recorded
# (We do not import main.py here - that would be a circular import.)
_history_fn = None
_version_fn = None

def set_price_source(history_fn, version_fn):
    global _history_fn, _version_fn
    _history_fn, _version_fn = history_fn, version_fn


def _get_prices(ticker: str, prices_param: str = ""):
    """Return (prices, version). Uses the system's OWN price history unless the
    caller passes ?prices=... by hand."""
    if prices_param:
        try:
            plist = [float(p) for p in prices_param.split(",")]
        except ValueError:
            raise HTTPException(400, "prices must be comma-separated floats")
        if len(plist) < MIN_POINTS:
            raise HTTPException(400, f"Need at least {MIN_POINTS} prices, got {len(plist)}")
        return plist, None          # None = typed by hand, not the system's data
    if _history_fn is None:
        raise HTTPException(503, "Price source not connected yet")
    plist = _history_fn(ticker)
    if len(plist) < MIN_POINTS:
        raise HTTPException(400, f"Not enough price history for {ticker} yet "
                                 f"(have {len(plist)}, need {MIN_POINTS}). Trade a bit first.")
    return plist, _version_fn(ticker)



# ── Load model ───────────────────────────────────────────────────────────────
@router.post("/load")
def load_models():
    """
    Load FinBERT model into memory.
    Call this once after starting the server.
    First call downloads ~400MB — subsequent calls are instant (cached).
    """
    sentiment_analyzer.load()
    rag_status = rag.load()      # first run builds the news index (1-3 min), then cached
    return {
        "ok":      True,
        "message": "FinBERT loaded. Rule-based fallback active if model unavailable.",
        "loaded":  sentiment_analyzer._loaded,
        "rag":     rag_status,
    }


# ── RAG: retrieved news ───────────────────────────────────────────────────────
@router.get("/news/{ticker}")
def get_news(ticker: str, query: str = "", k: int = 8):
    """Headlines RAG retrieves for a ticker (optionally for your own ?query=...)."""
    if ticker not in TICKERS:
        raise HTTPException(404, f"Ticker {ticker} not found")
    if not rag.ready:
        raise HTTPException(503, f"News index not ready: {rag.error or 'call POST /ml/load first'}")
    hits = rag.retrieve(ticker, query=query or "earnings results, guidance, analyst rating or price target change, risks, major company news", k=k)
    return {"ticker": ticker, "count": len(hits or []), "headlines": hits or [],
            "index": rag.status()}


# ── Sentiment ─────────────────────────────────────────────────────────────────
@router.get("/sentiment/{ticker}")
def get_sentiment(ticker: str):
    """FinBERT sentiment analysis for a ticker (per-headline breakdown)."""
    if ticker not in TICKERS:
        raise HTTPException(404, f"Ticker {ticker} not found")
    return sentiment_analyzer.analyze_ticker(ticker)


# ── Price prediction ─────────────────────────────────────────────────────────
@router.get("/predict/{ticker}")
def get_prediction(ticker: str, prices: str = ""):
    """
    Price forecast for the next 5 ticks, learned from THIS system's own
    recorded prices (optionally pass ?prices=1,2,3,... to override).
    """
    if ticker not in TICKERS:
        raise HTTPException(404, f"Ticker {ticker} not found")

    price_list, version = _get_prices(ticker, prices)
    train_info = signal_engine.ensure_trained(ticker, price_list, version)
    if not train_info.get("ok"):
        raise HTTPException(400, train_info.get("reason", "Training failed"))

    result = predictor_manager.predict(ticker, price_list)
    if not result.get("ok"):
        raise HTTPException(400, result.get("reason", "Prediction failed"))
    result["data"] = {"points_used": len(price_list),
                      "total_recorded": version,
                      "retrained_now": train_info.get("retrained", False)}
    return result


# ── Signal ────────────────────────────────────────────────────────────────────
@router.get("/signal/{ticker}")
def get_signal(ticker: str, prices: str = ""):
    """BUY / SELL / HOLD from sentiment + price forecast + anomaly check."""
    if ticker not in TICKERS:
        raise HTTPException(404, f"Ticker {ticker} not found")
    price_list, version = _get_prices(ticker, prices)
    return signal_engine.generate(ticker, price_list, version)


# ── Anomalies ─────────────────────────────────────────────────────────────────
@router.get("/anomalies")
def get_anomalies(n: int = 10):
    """Recent anomaly alerts across all tickers."""
    alerts = anomaly_manager.recent_alerts(n)
    return {
        "count":  len(alerts),
        "alerts": [
            {
                "ticker":       a.ticker,
                "type":         a.anomaly_type,
                "severity":     a.severity,
                "score":        a.score,
                "detail":       a.detail,
                "price":        a.trade.price,
                "qty":          a.trade.quantity,
            }
            for a in reversed(alerts)
        ],
    }


# ── All signals at once ───────────────────────────────────────────────────────
@router.get("/signals/all")
def get_all_signals():
    """Signals for every ticker, each based on its own recorded history."""
    results = {}
    for ticker in TICKERS:
        try:
            price_list, version = _get_prices(ticker)
            results[ticker] = signal_engine.generate(ticker, price_list, version)
        except HTTPException as e:
            results[ticker] = {"ticker": ticker, "signal": "HOLD",
                               "error": e.detail}
    return results