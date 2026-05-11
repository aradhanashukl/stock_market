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

router = APIRouter(prefix="/ml", tags=["ML"])

TICKERS = ["AAPL", "TSLA", "GOOGL", "INFY"]


# ── Load model ───────────────────────────────────────────────────────────────
@router.post("/load")
def load_models():
    """
    Load FinBERT model into memory.
    Call this once after starting the server.
    First call downloads ~400MB — subsequent calls are instant (cached).
    """
    sentiment_analyzer.load()
    return {
        "ok":      True,
        "message": "FinBERT loaded. Rule-based fallback active if model unavailable.",
        "loaded":  sentiment_analyzer._loaded,
    }


# ── Sentiment ─────────────────────────────────────────────────────────────────
@router.get("/sentiment/{ticker}")
def get_sentiment(ticker: str):
    """
    FinBERT sentiment analysis for a ticker.
    Returns aggregate label + per-headline breakdown.
    """
    if ticker not in TICKERS:
        raise HTTPException(404, f"Ticker {ticker} not found")

    result = sentiment_analyzer.analyze_ticker(ticker)
    return result


# ── Price prediction ─────────────────────────────────────────────────────────
@router.get("/predict/{ticker}")
def get_prediction(ticker: str, prices: str = ""):
    """
    LSTM price prediction for next 5 ticks.
    Pass recent prices as comma-separated string: ?prices=182.1,182.3,...
    Or omit to use auto-generated test prices.
    """
    if ticker not in TICKERS:
        raise HTTPException(404, f"Ticker {ticker} not found")

    if prices:
        try:
            price_list = [float(p) for p in prices.split(",")]
        except ValueError:
            raise HTTPException(400, "prices must be comma-separated floats")
    else:
        # Use synthetic prices for demo
        import random
        base = {"AAPL": 182, "TSLA": 245, "GOOGL": 175, "INFY": 19.5}[ticker]
        price_list = []
        p = base
        for _ in range(60):
            p += random.gauss(0, 0.4)
            price_list.append(round(max(1, p), 2))

    # Train if needed
    if ticker not in signal_engine._trained_tickers:
        train_result = predictor_manager.train(ticker, price_list)
        if train_result.get("ok"):
            signal_engine._trained_tickers.add(ticker)

    result = predictor_manager.predict(ticker, price_list)
    if not result.get("ok"):
        raise HTTPException(400, result.get("reason", "Prediction failed"))

    return result


# ── Signal ────────────────────────────────────────────────────────────────────
@router.get("/signal/{ticker}")
def get_signal(ticker: str, prices: str = ""):
    """
    Full ML signal: BUY / SELL / HOLD with confidence and reasoning.
    Combines FinBERT + LSTM + AnomalyDetector.
    """
    if ticker not in TICKERS:
        raise HTTPException(404, f"Ticker {ticker} not found")

    if prices:
        try:
            price_list = [float(p) for p in prices.split(",")]
        except ValueError:
            raise HTTPException(400, "prices must be comma-separated floats")
    else:
        import random
        base = {"AAPL": 182, "TSLA": 245, "GOOGL": 175, "INFY": 19.5}[ticker]
        price_list = []
        p = base
        for _ in range(60):
            p += random.gauss(0, 0.4)
            price_list.append(round(max(1, p), 2))

    result = signal_engine.generate(ticker, price_list)
    return result


# ── Anomalies ─────────────────────────────────────────────────────────────────
@router.get("/anomalies")
def get_anomalies(n: int = 10):
    """
    Recent anomaly alerts across all tickers.
    """
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
    """
    Generate signals for all tickers at once.
    Used by the frontend dashboard to populate all signal cards.
    """
    results = {}
    for ticker in TICKERS:
        import random
        base = {"AAPL": 182, "TSLA": 245, "GOOGL": 175, "INFY": 19.5}[ticker]
        prices = []
        p = base
        for _ in range(60):
            p += random.gauss(0, 0.4)
            prices.append(round(max(1, p), 2))
        results[ticker] = signal_engine.generate(ticker, prices)
    return results