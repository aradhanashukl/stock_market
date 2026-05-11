TradeOS — DSA-Powered Stock Trading System with ML

A full-stack algorithmic trading simulator built from scratch using core Data Structures and Algorithms (heaps, queues, hashmaps, circular buffers), combined with a Machine Learning layer for real-time signals.

The system simulates a live order book for 4 tickers (AAPL, TSLA, GOOGL, INFY) — matching buy/sell orders, tracking portfolio P&L, detecting price anomalies, and generating BUY/SELL/HOLD signals — all served via a REST API with a terminal-styled trading dashboard UI.
Key DSA concepts implemented from scratch:

Max-Heap / Min-Heap — bid/ask priority queues with O(log n) insert and O(1) best-price lookup
Order Matching Engine — price-time priority matching across heap levels
HashMap — O(1) ticker lookup for order books, price history, and portfolio holdings
Circular Buffer + FIFO Queue — bounded memory price history and trade log

ML layer (3 models fused into one signal):

FinBERT (HuggingFace) — financial news sentiment analysis; rule-based fallback
LSTM (PyTorch) — next-5-tick price forecasting; linear regression fallback
IsolationForest (scikit-learn) — unsupervised anomaly detection on trade events

How to run:
bash# Terminal 1 — API server
uvicorn main:app --reload

# Terminal 2 — Frontend
python -m http.server 3000
Then open http://localhost:3000 and API docs at http://localhost:8000/docs.

Resume — Technologies to List
Languages & Frameworks

Python, FastAPI, Pydantic, Uvicorn
HTML5, CSS3, Vanilla JavaScript (Canvas API for charting)

Data Structures & Algorithms (mention this explicitly — it's a strong signal)

Heaps (Max/Min), FIFO Queue, Circular Buffer, HashMap — all implemented from scratch

Machine Learning

PyTorch (LSTM neural network for time-series forecasting)
scikit-learn (IsolationForest anomaly detection)
HuggingFace Transformers — FinBERT (financial NLP / sentiment analysis)
NumPy (linear regression fallback, feature engineering)

System Design

REST API design, CORS middleware, modular ML router
Price-time priority order matching engine
Real-time portfolio P&L with weighted average cost accounting

