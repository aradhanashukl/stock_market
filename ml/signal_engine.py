# ── ml/signal_engine.py ────────────────────────────────────────────────────
#
# WHAT THIS DOES:
#   Combines three ML signals into one actionable trading decision:
#
#   1. SENTIMENT  (FinBERT / rule-based)
#      How is the news? Positive news → bullish pressure
#
#   2. LSTM PREDICTION
#      Where is the price going? Upward trend → buy signal
#
#   3. ANOMALY DETECTOR (IsolationForest)
#      Is something unusual happening? Anomalies → caution / HOLD
#
# SIGNAL FUSION:
#   Each sub-signal contributes a score in [-1, +1]:
#     sentiment_score:  +1 = strong positive, -1 = strong negative
#     trend_score:      +1 = strong uptrend,  -1 = strong downtrend
#     anomaly_penalty:  -0.3 applied if recent anomaly detected
#
#   final_score = 0.4 * sentiment + 0.5 * trend + 0.1 * anomaly_penalty
#
#   Score thresholds:
#     > +0.15  → BUY
#     < -0.15  → SELL
#     else     → HOLD
# ───────────────────────────────────────────────────────────────────────────

from typing import List, Dict
import time

from ml.sentiment  import analyzer      as sentiment_analyzer
from ml.predictor  import manager       as predictor_manager
from ml.anomaly    import anomaly_manager
from ml.rag        import rag


# Weights for signal fusion
W_SENTIMENT = 0.40
W_TREND     = 0.50
W_ANOMALY   = 0.10

BUY_THRESHOLD  =  0.15
SELL_THRESHOLD = -0.15


class SignalEngine:
    """
    Fuses FinBERT + LSTM + IsolationForest into BUY / SELL / HOLD signals.
    """

    RETRAIN_EVERY = 10     # retrain after this many NEW prices arrive

    def __init__(self):
        self._trained_at: Dict[str, int] = {}   # ticker -> price-count at last training

    def ensure_trained(self, ticker: str, prices: List[float], version) -> Dict:
        """
        Train (or retrain) the price model on the latest history.
        `version` = how many prices the system has ever recorded for this
        ticker. When 10+ new prices have arrived since the last training,
        the model learns from the fresh data again.
        """
        last = self._trained_at.get(ticker)
        # version=None means "prices typed in by hand": always train on them,
        # then forget it, so the next real call retrains on the system's data.
        if version is not None and last is not None and version - last < self.RETRAIN_EVERY:
            return {"ok": True, "retrained": False}
        res = predictor_manager.train(ticker, prices)
        if version is None:
            self._trained_at.pop(ticker, None)
        elif res.get("ok"):
            self._trained_at[ticker] = version
        res["retrained"] = bool(res.get("ok"))
        return res

    def generate(self, ticker: str, prices: List[float], version=0) -> Dict:
        """
        Generate a trading signal for `ticker`.
        `prices` should be at least 25 recent price points.
        `version` = total prices recorded so far (decides when to retrain).
        """
        reasons = []
        sub_signals = {}

        # ── 1. Sentiment ─────────────────────────────────────────────────────
        # RAG: search the news corpus for the most relevant recent headlines.
        # If the index is not loaded (or the stock has no news), fall back
        # to the 5 built-in sample headlines.
        news = rag.retrieve(ticker)
        if news:
            sentiment_result = sentiment_analyzer.analyze_ticker(
                ticker, headlines=[n["title"] for n in news])
            news_source = f"RAG: {len(news)} retrieved headlines (historical corpus)"
        else:
            sentiment_result = sentiment_analyzer.analyze_ticker(ticker)
            news_source = "sample headlines (RAG index not loaded)"
        sentiment_score  = sentiment_result.get("sentiment_score", 0.0)  # -1 to +1
        sub_signals["sentiment"] = {
            "score":      round(sentiment_score, 3),
            "label":      sentiment_result.get("aggregate_label", "NEUTRAL"),
            "confidence": sentiment_result.get("aggregate_score", 0.5),
            "model":      sentiment_result.get("model", "rule-based"),
            "source":     news_source,
        }
        # attach each retrieved headline with its date, similarity and FinBERT label
        if news:
            by_title = {b["headline"]: b for b in sentiment_result.get("breakdown", [])}
            sub_signals["news"] = [
                {**n, "label": by_title.get(n["title"], {}).get("label", "?")}
                for n in news
            ]
        if sentiment_score > 0.2:
            reasons.append(f"Positive news sentiment ({sentiment_result.get('positive_count',0)} bullish headlines)")
        elif sentiment_score < -0.2:
            reasons.append(f"Negative news sentiment ({sentiment_result.get('negative_count',0)} bearish headlines)")
        else:
            reasons.append("Neutral news sentiment")

        # ── 2. LSTM / Linear Prediction ──────────────────────────────────────
        trend_score = 0.0
        if len(prices) >= 25:
            # Learn from the latest data (retrains when new prices arrived)
            train_res = self.ensure_trained(ticker, prices, version)

            pred_result = predictor_manager.predict(ticker, prices)
            sub_signals["prediction"] = pred_result

            if pred_result.get("ok"):
                delta_pct = pred_result.get("delta_pct", 0.0)
                trend     = pred_result.get("trend", "FLAT")

                # Map delta_pct → [-1, +1] score (cap at ±2%)
                trend_score = max(-1.0, min(1.0, delta_pct / 2.0))

                reasons.append(
                    f"Price forecast: {trend} trend, "
                    f"Δ{delta_pct:+.2f}% over next 5 ticks "
                    f"(model: {pred_result.get('model','?')})"
                )
            else:
                reasons.append(f"Prediction unavailable: {pred_result.get('reason','?')}")
        else:
            sub_signals["prediction"] = {"ok": False, "reason": "Not enough price data"}
            reasons.append("Not enough price history for LSTM forecast")

        # ── 3. Anomaly check ─────────────────────────────────────────────────
        recent_alerts = anomaly_manager.recent_alerts(5)
        ticker_alerts = [a for a in recent_alerts if a.ticker == ticker]
        anomaly_penalty = 0.0

        if ticker_alerts:
            # Penalise by severity
            severity_map = {"HIGH": -0.5, "MEDIUM": -0.3, "LOW": -0.1}
            worst = max(ticker_alerts, key=lambda a: abs(severity_map.get(a.severity, 0)))
            anomaly_penalty = severity_map.get(worst.severity, 0.0)
            sub_signals["anomaly"] = {
                "detected":  True,
                "type":      worst.anomaly_type,
                "severity":  worst.severity,
                "detail":    worst.detail,
            }
            reasons.append(f"⚠️ Anomaly detected: {worst.anomaly_type} ({worst.severity}) — {worst.detail}")
        else:
            sub_signals["anomaly"] = {"detected": False}
            reasons.append("No recent anomalies detected")

        # ── Signal fusion ─────────────────────────────────────────────────────
        final_score = (
            W_SENTIMENT * sentiment_score +
            W_TREND     * trend_score +
            W_ANOMALY   * anomaly_penalty
        )
        final_score = round(final_score, 4)

        if final_score > BUY_THRESHOLD:
            signal     = "BUY"
            confidence = round(min(0.99, 0.5 + final_score * 2), 2)
        elif final_score < SELL_THRESHOLD:
            signal     = "SELL"
            confidence = round(min(0.99, 0.5 + abs(final_score) * 2), 2)
        else:
            signal     = "HOLD"
            confidence = round(max(0.50, 1.0 - abs(final_score) * 3), 2)

        return {
            "ticker":      ticker,
            "signal":      signal,
            "confidence":  confidence,
            "final_score": final_score,
            "sub_signals": sub_signals,
            "reasons":     reasons,
            "data":        {"points_used": len(prices),
                            "source": "this system's own recorded prices",
                            "total_recorded": version},
            "weights":     {
                "sentiment": W_SENTIMENT,
                "trend":     W_TREND,
                "anomaly":   W_ANOMALY,
            },
            "timestamp":   round(time.time(), 2),
        }


# Singleton
signal_engine = SignalEngine()