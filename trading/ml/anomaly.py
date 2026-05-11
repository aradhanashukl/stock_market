# ── ml/anomaly.py ──────────────────────────────────────────────────────────
#
# WHAT THIS DOES:
#   Watches every trade and order book snapshot for unusual activity.
#   Uses IsolationForest — an unsupervised ML algorithm that learns
#   what "normal" looks like, then flags anything that deviates.
#
# HOW ISOLATION FOREST WORKS:
#   Imagine you're trying to find a needle in a haystack.
#   Normal points (hay) need many cuts to isolate.
#   Anomalies (needle) are isolated in very few cuts.
#   IsolationForest builds random trees and measures cut depth.
#   Short path = anomaly. Long path = normal.
#
# FEATURES TRACKED per trade:
#   - price deviation from rolling average
#   - quantity (unusually large orders = suspicious)
#   - price velocity (how fast price is moving)
#   - spread (abnormally wide spread = liquidity problem)
#
# ANOMALY TYPES DETECTED:
#   PRICE_SPIKE    → price moved too far too fast
#   LARGE_ORDER    → unusually large quantity
#   WIDE_SPREAD    → spread much wider than normal
#   VELOCITY       → price changing too rapidly
# ───────────────────────────────────────────────────────────────────────────

import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque
from dataclasses import dataclass


@dataclass
class TradeEvent:
    ticker:   str
    price:    float
    quantity: int
    spread:   float
    side:     str


@dataclass
class AnomalyAlert:
    ticker:      str
    anomaly_type: str
    severity:    str       # LOW / MEDIUM / HIGH
    score:       float     # closer to -1 = more anomalous
    detail:      str
    trade:       TradeEvent


class AnomalyDetector:
    """
    Per-ticker anomaly detection using IsolationForest.

    Workflow:
      1. Collect at least MIN_SAMPLES trade events to train
      2. Retrain model every RETRAIN_EVERY new events
      3. Score each incoming event — flag if anomalous
    """

    MIN_SAMPLES    = 20     # minimum events before training
    RETRAIN_EVERY  = 15     # retrain after this many new events
    CONTAMINATION  = 0.08   # expected anomaly rate (8%)

    def __init__(self, ticker: str):
        self.ticker   = ticker
        self._history = deque(maxlen=200)   # rolling window of events
        self._model: IsolationForest | None = None
        self._since_retrain = 0
        self._price_history = deque(maxlen=30)

    def _extract_features(self, event: TradeEvent) -> list[float]:
        """
        Convert a trade event into a feature vector for the model.

        Features:
          [0] price_z_score     how far price is from rolling mean (in std devs)
          [1] log_quantity       log-scaled quantity (handles large orders)
          [2] price_velocity     rate of price change
          [3] spread_ratio       spread as % of price
        """
        prices = list(self._price_history)

        # Price z-score
        if len(prices) >= 5:
            mu    = np.mean(prices)
            sigma = np.std(prices) + 1e-6
            price_z = (event.price - mu) / sigma
        else:
            price_z = 0.0

        # Log quantity (log scale handles outliers better)
        log_qty = np.log1p(event.quantity)

        # Price velocity (how much price moved in last tick)
        if len(prices) >= 2:
            velocity = abs(prices[-1] - prices[-2])
        else:
            velocity = 0.0

        # Spread ratio (spread as % of current price)
        spread_ratio = (event.spread / event.price * 100) if event.price > 0 else 0.0

        return [price_z, log_qty, velocity, spread_ratio]

    def _train(self):
        """Train IsolationForest on collected history."""
        if len(self._history) < self.MIN_SAMPLES:
            return

        X = np.array([h["features"] for h in self._history])
        self._model = IsolationForest(
            n_estimators  = 100,
            contamination = self.CONTAMINATION,
            random_state  = 42,
        )
        self._model.fit(X)
        self._since_retrain = 0

    def process(self, event: TradeEvent) -> AnomalyAlert | None:
        """
        Process one trade event.
        Returns AnomalyAlert if anomaly detected, else None.
        """
        self._price_history.append(event.price)
        features = self._extract_features(event)

        self._history.append({"features": features, "event": event})
        self._since_retrain += 1

        # Train / retrain model
        if (self._model is None and len(self._history) >= self.MIN_SAMPLES) or \
           (self._since_retrain >= self.RETRAIN_EVERY and len(self._history) >= self.MIN_SAMPLES):
            self._train()

        if self._model is None:
            return None   # not enough data yet

        # Score this event
        x     = np.array(features).reshape(1, -1)
        pred  = self._model.predict(x)[0]       # 1 = normal, -1 = anomaly
        score = self._model.score_samples(x)[0] # more negative = more anomalous

        if pred == -1:
            return self._build_alert(features, score, event)
        return None

    def _build_alert(self, features: list, score: float,
                     event: TradeEvent) -> AnomalyAlert:
        """Classify what kind of anomaly this is."""
        price_z, log_qty, velocity, spread_ratio = features

        # Determine anomaly type by which feature is most extreme
        severity = "HIGH" if score < -0.15 else "MEDIUM" if score < -0.10 else "LOW"

        if abs(price_z) > 2.0:
            atype  = "PRICE_SPIKE"
            detail = f"Price ₹{event.price:.2f} is {abs(price_z):.1f}σ from mean"
        elif log_qty > 5.5:   # e+exp(5.5) ≈ 244 shares
            atype  = "LARGE_ORDER"
            detail = f"Unusual qty {event.quantity} on {event.side} order"
        elif spread_ratio > 1.5:
            atype  = "WIDE_SPREAD"
            detail = f"Spread {event.spread:.2f} is {spread_ratio:.1f}% of price"
        elif velocity > 1.0:
            atype  = "VELOCITY"
            detail = f"Price moving too fast: Δ{velocity:.2f} per tick"
        else:
            atype  = "UNUSUAL"
            detail = f"Anomaly score {score:.4f}"

        return AnomalyAlert(
            ticker       = event.ticker,
            anomaly_type = atype,
            severity     = severity,
            score        = round(score, 4),
            detail       = detail,
            trade        = event,
        )

    def stats(self) -> dict:
        return {
            "ticker":          self.ticker,
            "events_seen":     len(self._history),
            "model_trained":   self._model is not None,
            "contamination":   self.CONTAMINATION,
        }


# ── Multi-ticker manager ─────────────────────────────────────────────────────
class AnomalyManager:
    def __init__(self):
        self._detectors: dict[str, AnomalyDetector] = {}
        self.alerts: list[AnomalyAlert] = []

    def get_or_create(self, ticker: str) -> AnomalyDetector:
        if ticker not in self._detectors:
            self._detectors[ticker] = AnomalyDetector(ticker)
        return self._detectors[ticker]

    def process(self, event: TradeEvent) -> AnomalyAlert | None:
        alert = self.get_or_create(event.ticker).process(event)
        if alert:
            self.alerts.append(alert)
        return alert

    def recent_alerts(self, n=10) -> list[AnomalyAlert]:
        return self.alerts[-n:]


anomaly_manager = AnomalyManager()


# ── Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    print("=" * 55)
    print("IsolationForest Anomaly Detector")
    print("=" * 55)

    detector = AnomalyDetector("AAPL")
    alerts_found = []
    base_price = 182.0

    print("\n  Feeding 40 normal trades then injecting anomalies...\n")

    # 40 normal trades
    for i in range(40):
        price  = base_price + random.gauss(0, 0.3)
        qty    = random.randint(10, 100)
        spread = random.uniform(0.05, 0.2)
        event  = TradeEvent("AAPL", round(price,2), qty, round(spread,3), "BUY")
        alert  = detector.process(event)
        if alert:
            alerts_found.append(alert)

    print(f"  Normal trades processed: 40, alerts: {len(alerts_found)}")

    # Inject obvious anomalies
    anomalies = [
        TradeEvent("AAPL", 195.00,  10,   0.10, "BUY"),   # price spike
        TradeEvent("AAPL", 182.10,  5000, 0.10, "SELL"),  # huge order
        TradeEvent("AAPL", 175.00,  10,   0.10, "SELL"),  # price crash
        TradeEvent("AAPL", 182.00,  20,   4.50, "BUY"),   # wide spread
    ]

    print("\n  Injecting anomalous trades:")
    for ev in anomalies:
        alert = detector.process(ev)
        if alert:
            icon = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(alert.severity,"⚪")
            print(f"\n  {icon} ANOMALY DETECTED")
            print(f"     Type     : {alert.anomaly_type}")
            print(f"     Severity : {alert.severity}")
            print(f"     Score    : {alert.score}")
            print(f"     Detail   : {alert.detail}")
        else:
            print(f"  ✓  No alert for price={ev.price} qty={ev.quantity}")

    print(f"\n  Stats: {detector.stats()}")