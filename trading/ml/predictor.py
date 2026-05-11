# ── ml/predictor.py ────────────────────────────────────────────────────────
#
# WHAT THIS DOES:
#   Trains a lightweight LSTM on recent price history and forecasts
#   the next N price points.
#
# HOW LSTM WORKS (simple version):
#   LSTM = Long Short-Term Memory — a recurrent neural network designed
#   to remember patterns over sequences (time-series data).
#   - It reads prices one at a time: [p1, p2, p3, ... p60]
#   - It maintains a "memory cell" that learns what's important to keep
#   - Output: predicted next price(s)
#
# TWO MODES:
#   1. PyTorch LSTM        → real ML, trains on GPU/CPU
#   2. Numpy linear model  → linear regression fallback (no deep learning)
#
# WINDOW = 20: model sees last 20 prices to predict next 5
# ───────────────────────────────────────────────────────────────────────────

import numpy as np
from typing import List, Dict, Optional


# ── Numpy linear regression fallback ─────────────────────────────────────────
class LinearPredictor:
    """
    Simple linear trend extrapolation.
    Good enough as a fallback and for explaining the concept.
    Fits y = a*x + b on recent prices, extrapolates N steps.
    """

    def __init__(self, window: int = 20, forecast: int = 5):
        self.window   = window
        self.forecast = forecast
        self._slope   = 0.0
        self._intercept = 0.0
        self._last_prices: List[float] = []

    def train(self, prices: List[float]) -> Dict:
        if len(prices) < self.window:
            return {"ok": False, "reason": f"Need at least {self.window} prices, got {len(prices)}"}

        recent = np.array(prices[-self.window:])
        x = np.arange(len(recent))
        # Least squares: [slope, intercept]
        coeffs = np.polyfit(x, recent, 1)
        self._slope     = coeffs[0]
        self._intercept = coeffs[1]
        self._last_prices = prices
        return {"ok": True, "model": "linear", "points": len(recent)}

    def predict(self, prices: List[float]) -> Dict:
        if not self._last_prices:
            train_result = self.train(prices)
            if not train_result["ok"]:
                return train_result

        recent = np.array(prices[-self.window:])
        x = np.arange(len(recent))
        coeffs = np.polyfit(x, recent, 1)

        # Extrapolate
        start  = len(recent)
        future_x = np.arange(start, start + self.forecast)
        predictions = (coeffs[0] * future_x + coeffs[1]).tolist()
        predictions = [round(max(0.01, p), 4) for p in predictions]

        last  = float(recent[-1])
        trend = "UP" if predictions[-1] > last else "DOWN" if predictions[-1] < last else "FLAT"
        delta = round(predictions[-1] - last, 4)

        return {
            "ok":          True,
            "model":       "linear-regression",
            "ticker":      "?",
            "last_price":  last,
            "predictions": predictions,
            "trend":       trend,
            "delta":       delta,
            "delta_pct":   round((delta / last) * 100, 3) if last else 0,
        }


# ── PyTorch LSTM ───────────────────────────────────────────────────────────────
class LSTMPredictor:
    """
    LSTM using PyTorch for sequence-to-sequence price prediction.

    Architecture:
      Input  → [batch, seq_len, 1]   (normalized prices)
      LSTM   → hidden_size=32, num_layers=2
      FC     → hidden_size → forecast_steps
      Output → [batch, forecast_steps]
    """

    def __init__(self, window: int = 20, forecast: int = 5,
                 hidden: int = 32, layers: int = 2, epochs: int = 30):
        self.window   = window
        self.forecast = forecast
        self.hidden   = hidden
        self.layers   = layers
        self.epochs   = epochs
        self._model   = None
        self._scaler_min: float = 0.0
        self._scaler_rng: float = 1.0

    def _build_model(self):
        import torch
        import torch.nn as nn

        class _LSTM(nn.Module):
            def __init__(self, hidden, layers, forecast):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size  = 1,
                    hidden_size = hidden,
                    num_layers  = layers,
                    batch_first = True,
                    dropout     = 0.1 if layers > 1 else 0,
                )
                self.fc = nn.Linear(hidden, forecast)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        return _LSTM(self.hidden, self.layers, self.forecast)

    def _normalize(self, prices: np.ndarray) -> np.ndarray:
        self._scaler_min = float(prices.min())
        self._scaler_rng = float(prices.max() - prices.min()) + 1e-9
        return (prices - self._scaler_min) / self._scaler_rng

    def _denormalize(self, arr: np.ndarray) -> np.ndarray:
        return arr * self._scaler_rng + self._scaler_min

    def _make_sequences(self, prices: np.ndarray):
        """Sliding window sequences for LSTM training."""
        X, y = [], []
        for i in range(len(prices) - self.window - self.forecast + 1):
            X.append(prices[i: i + self.window])
            y.append(prices[i + self.window: i + self.window + self.forecast])
        return np.array(X), np.array(y)

    def train(self, prices: List[float]) -> Dict:
        if len(prices) < self.window + self.forecast:
            return {
                "ok":     False,
                "reason": f"Need {self.window + self.forecast} prices, got {len(prices)}"
            }
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            arr = self._normalize(np.array(prices, dtype=np.float32))
            X, y = self._make_sequences(arr)
            if len(X) == 0:
                return {"ok": False, "reason": "Not enough data after windowing"}

            X_t = torch.tensor(X.reshape(len(X), self.window, 1), dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)

            self._model = self._build_model()
            optimizer   = optim.Adam(self._model.parameters(), lr=0.01)
            criterion   = nn.MSELoss()

            self._model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                output = self._model(X_t)
                loss   = criterion(output, y_t)
                loss.backward()
                optimizer.step()

            self._model.eval()
            return {"ok": True, "model": "LSTM-PyTorch", "epochs": self.epochs,
                    "final_loss": round(float(loss.item()), 6)}
        except ImportError:
            return {"ok": False, "reason": "PyTorch not installed"}
        except Exception as e:
            return {"ok": False, "reason": str(e)}

    def predict(self, prices: List[float]) -> Dict:
        if self._model is None:
            return {"ok": False, "reason": "Model not trained. Call train() first."}
        try:
            import torch
            recent = np.array(prices[-self.window:], dtype=np.float32)
            norm   = (recent - self._scaler_min) / self._scaler_rng
            x_t    = torch.tensor(norm.reshape(1, self.window, 1), dtype=torch.float32)
            with torch.no_grad():
                pred_norm = self._model(x_t).numpy()[0]
            predictions = self._denormalize(pred_norm).tolist()
            predictions = [round(max(0.01, p), 4) for p in predictions]
            last  = float(recent[-1])
            delta = round(predictions[-1] - last, 4)
            trend = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"
            return {
                "ok":          True,
                "model":       "LSTM-PyTorch",
                "last_price":  last,
                "predictions": predictions,
                "trend":       trend,
                "delta":       delta,
                "delta_pct":   round((delta / last) * 100, 3) if last else 0,
            }
        except Exception as e:
            return {"ok": False, "reason": str(e)}


# ── PredictorManager: tries LSTM, falls back to Linear ───────────────────────
class PredictorManager:
    """
    Manages one predictor per ticker.
    Tries PyTorch LSTM first; automatically falls back to LinearPredictor.
    """

    def __init__(self):
        self._lstm:   Dict[str, LSTMPredictor]   = {}
        self._linear: Dict[str, LinearPredictor] = {}
        self._use_lstm: Dict[str, bool]           = {}

    def train(self, ticker: str, prices: List[float]) -> Dict:
        # Try LSTM
        if ticker not in self._lstm:
            self._lstm[ticker]   = LSTMPredictor()
            self._linear[ticker] = LinearPredictor()

        result = self._lstm[ticker].train(prices)
        if result.get("ok"):
            self._use_lstm[ticker] = True
        else:
            # Fallback to linear
            fallback = self._linear[ticker].train(prices)
            self._use_lstm[ticker] = False
            return fallback
        return result

    def predict(self, ticker: str, prices: List[float]) -> Dict:
        if ticker not in self._lstm:
            train_result = self.train(ticker, prices)
            if not train_result.get("ok"):
                return train_result

        if self._use_lstm.get(ticker):
            result = self._lstm[ticker].predict(prices)
            if result.get("ok"):
                result["ticker"] = ticker
                return result
            # LSTM predict failed — try linear
            self._use_lstm[ticker] = False

        result = self._linear[ticker].predict(prices)
        result["ticker"] = ticker
        return result


# Singleton
manager = PredictorManager()