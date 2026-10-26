# ── trading_system.py ──────────────────────────────────────────────────────
#
# Core DSA Engine — ties together all data structures:
#
#   OrderBook      → per-ticker MaxHeap (bids) + MinHeap (asks)
#   PriceTracker   → HashMap of deque (rolling price history)
#   Portfolio      → HashMap of holdings + cash management
#   TradeLog       → Circular buffer (deque) of executed trades
#   TradingSystem  → Orchestrates all of the above
#
# DATA STRUCTURES USED:
#   MaxHeap  → best bid always O(1) accessible, insert/remove O(log n)
#   MinHeap  → best ask always O(1) accessible, insert/remove O(log n)
#   HashMap  → O(1) ticker lookup for books, prices, holdings
#   Deque    → O(1) append/pop for price history & trade log
# ───────────────────────────────────────────────────────────────────────────

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

from heap import Order, MaxHeap, MinHeap


# ── Order Book (per ticker) ──────────────────────────────────────────────────
class OrderBook:
    """
    Limit Order Book for one ticker.

    bids → MaxHeap  (buyers: highest price wins)
    asks → MinHeap  (sellers: lowest price wins)

    Matching logic:
      A BUY  order at price P matches against asks where ask.price <= P
      A SELL order at price P matches against bids where bid.price >= P
    """

    def __init__(self, ticker: str):
        self.ticker   = ticker
        self.bids     = MaxHeap()
        self.asks     = MinHeap()
        self._counter = 0           # monotonic order-id counter

    def _next_id(self) -> int:
        self._counter += 1
        return self._counter

    def place_order(self, side: str, price: float, quantity: int,
                    rest: bool = False) -> dict:
        """
        Place a limit order. Returns match result.

        rest=False → "immediate-or-cancel": fill what we can RIGHT NOW,
                     throw away whatever could not be filled.
        rest=True  → the unfilled part waits in the book as a bid / ask.
        """
        order = Order(
            order_id = self._next_id(),
            ticker   = self.ticker,
            side     = side,
            price    = price,
            quantity = quantity,
        )

        trades = []
        remaining = quantity

        if side == "BUY":
            # Match against asks (lowest ask first)
            while remaining > 0 and self.asks.peek() and self.asks.peek().price <= price:
                ask = self.asks.pop()
                fill = min(remaining, ask.quantity)
                trades.append({
                    "price":    ask.price,
                    "qty":      fill,
                    "side":     "BUY",
                    "ticker":   self.ticker,
                    "order_id": order.order_id,
                })
                remaining -= fill
                if ask.quantity > fill:
                    # Partially filled — push remainder back
                    leftover = Order(ask.order_id, ask.ticker, ask.side,
                                     ask.price, ask.quantity - fill)
                    self.asks.push(leftover)

            if remaining > 0 and rest:
                # Unmatched portion goes into book
                order.quantity = remaining
                self.bids.push(order)

        else:  # SELL
            while remaining > 0 and self.bids.peek() and self.bids.peek().price >= price:
                bid = self.bids.pop()
                fill = min(remaining, bid.quantity)
                trades.append({
                    "price":    bid.price,
                    "qty":      fill,
                    "side":     "SELL",
                    "ticker":   self.ticker,
                    "order_id": order.order_id,
                })
                remaining -= fill
                if bid.quantity > fill:
                    leftover = Order(bid.order_id, bid.ticker, bid.side,
                                     bid.price, bid.quantity - fill)
                    self.bids.push(leftover)

            if remaining > 0 and rest:
                order.quantity = remaining
                self.asks.push(order)

        status = "EXECUTED" if trades and remaining == 0 else \
                 "PARTIAL"  if trades else \
                 "QUEUED"   if rest   else "UNFILLED"

        return {"ok": True, "status": status, "trades": trades,
                "order": order, "remaining": remaining}

    def best_bid(self) -> Optional[float]:
        top = self.bids.peek()
        return top.price if top else None

    def best_ask(self) -> Optional[float]:
        top = self.asks.peek()
        return top.price if top else None

    def spread(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        if b and a:
            return round(a - b, 4)
        return None

    def mid_price(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        if b and a:
            return round((a + b) / 2, 4)
        return None


# ── Price Tracker (HashMap of deques) ────────────────────────────────────────
class PriceTracker:
    """
    Stores rolling price history per ticker.

    HashMap[ticker] → deque(maxlen=500)
    O(1) insert, O(1) latest, O(n) history (n = window)
    """

    def __init__(self, maxlen: int = 500):
        self._data: Dict[str, deque] = {}
        self._count: Dict[str, int] = {}    # total prices ever recorded
        self._maxlen = maxlen

    def record(self, ticker: str, price: float):
        if ticker not in self._data:
            self._data[ticker] = deque(maxlen=self._maxlen)
        self._data[ticker].append(round(price, 4))
        self._count[ticker] = self._count.get(ticker, 0) + 1

    def count(self, ticker: str) -> int:
        """How many prices were ever recorded (keeps growing even when the
        rolling window is full) - the ML layer uses this to know when new
        data has arrived and the models should be retrained."""
        return self._count.get(ticker, 0)

    def latest(self, ticker: str) -> Optional[float]:
        d = self._data.get(ticker)
        return d[-1] if d else None

    def history(self, ticker: str, n: int = 60) -> List[float]:
        d = self._data.get(ticker, deque())
        return list(d)[-n:]

    def all_tickers(self) -> List[str]:
        return list(self._data.keys())


# ── Trade Log (circular buffer) ───────────────────────────────────────────────
class TradeLog:
    """
    Append-only circular buffer for executed trades.
    maxlen ensures memory stays bounded even for millions of trades.
    """

    def __init__(self, maxlen: int = 1000):
        self._log: deque = deque(maxlen=maxlen)

    def append(self, trade: dict):
        trade["ts"] = time.time()
        self._log.append(trade)

    def all(self) -> List[dict]:
        return list(self._log)

    def __len__(self):
        return len(self._log)


# ── Portfolio ─────────────────────────────────────────────────────────────────
class Portfolio:
    """
    Tracks cash + holdings.

    holdings: HashMap[ticker] → {qty, avg_cost}
    Supports:
      - buy / sell with avg-cost accounting
      - unrealised PnL per position
      - total portfolio value
    """

    def __init__(self, starting_cash: float = 50_000.0):
        self.cash          = starting_cash
        self.holdings: Dict[str, dict] = {}
        self.realised_pnl  = 0.0
        self.trade_count   = 0

    def buy(self, ticker: str, price: float, qty: int) -> dict:
        cost = price * qty
        if cost > self.cash:
            return {"ok": False, "reason": f"Insufficient cash. Need ₹{cost:.2f}, have ₹{self.cash:.2f}"}

        self.cash -= cost
        if ticker in self.holdings:
            h = self.holdings[ticker]
            total_qty  = h["qty"] + qty
            total_cost = h["avg_cost"] * h["qty"] + price * qty
            h["qty"]      = total_qty
            h["avg_cost"] = total_cost / total_qty
        else:
            self.holdings[ticker] = {"qty": qty, "avg_cost": price}

        self.trade_count += 1
        return {"ok": True, "cost": cost}

    def sell(self, ticker: str, price: float, qty: int) -> dict:
        h = self.holdings.get(ticker)
        if not h or h["qty"] < qty:
            held = h["qty"] if h else 0
            return {"ok": False, "reason": f"Insufficient holdings. Have {held}, need {qty}"}

        proceeds = price * qty
        pnl      = (price - h["avg_cost"]) * qty
        self.cash         += proceeds
        self.realised_pnl += pnl
        h["qty"]          -= qty
        if h["qty"] == 0:
            del self.holdings[ticker]

        self.trade_count += 1
        return {"ok": True, "proceeds": proceeds, "pnl": round(pnl, 2)}

    def unrealised_pnl(self, current_prices: Dict[str, float]) -> Dict[str, dict]:
        result = {}
        for ticker, h in self.holdings.items():
            cp  = current_prices.get(ticker, h["avg_cost"])
            pnl = (cp - h["avg_cost"]) * h["qty"]
            pnl_pct = ((cp - h["avg_cost"]) / h["avg_cost"]) * 100 if h["avg_cost"] else 0
            result[ticker] = {
                "pnl":     round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
            }
        return result

    def total_value(self, current_prices: Dict[str, float]) -> float:
        equity = sum(
            h["qty"] * current_prices.get(t, h["avg_cost"])
            for t, h in self.holdings.items()
        )
        return round(self.cash + equity, 2)


# ── TradingSystem ─────────────────────────────────────────────────────────────
class TradingSystem:
    """
    Top-level orchestrator.

    Combines:
      - OrderBook per ticker   (HashMap of OrderBooks)
      - PriceTracker           (HashMap of deques)
      - Portfolio              (cash + holdings)
      - TradeLog               (circular buffer)
      - AnomalyManager         (IsolationForest per ticker)
    """

    def __init__(self, tickers: List[str], starting_cash: float = 50_000.0):
        self.tickers       = tickers
        self.order_books   = {t: OrderBook(t) for t in tickers}   # HashMap
        self.price_tracker = PriceTracker()
        self.portfolio     = Portfolio(starting_cash)
        self.trade_log     = TradeLog()

        # Lazy-import anomaly manager to avoid circular imports
        try:
            from ml.anomaly import anomaly_manager, TradeEvent
            self._anomaly_manager = anomaly_manager
            self._TradeEvent      = TradeEvent
        except ImportError:
            self._anomaly_manager = None
            self._TradeEvent      = None

    def place_order(self, side: str, ticker: str,
                    price: float, qty: int) -> dict:
        """
        Order pipeline (matching FIRST, accounting SECOND):
          1. Safety check   - could you pay for / deliver this at worst case?
                              (nothing is changed yet)
          2. Match against the order book - get the REAL fills
          3. For every fill: update cash + shares at the FILL price
          4. Record price + trade log ONLY for real trades
          5. Anomaly detection on each fill
        """
        # 1. Safety check only - do NOT touch cash or holdings yet
        if side == "BUY":
            worst_cost = price * qty
            if worst_cost > self.portfolio.cash:
                return {"ok": False, "reason":
                        f"Insufficient cash. Need up to Rs{worst_cost:.2f}, "
                        f"have Rs{self.portfolio.cash:.2f}"}
        else:
            h    = self.portfolio.holdings.get(ticker)
            held = h["qty"] if h else 0
            if held < qty:
                return {"ok": False, "reason":
                        f"Insufficient holdings. Have {held}, need {qty}"}

        # 2. Match. Unfilled part is cancelled (rest=False)
        book   = self.order_books[ticker]
        result = book.place_order(side, price, qty, rest=False)

        # 3 + 4 + 5. One pass over the REAL fills
        filled = 0
        for t in result["trades"]:
            fill_price, fill_qty = t["price"], t["qty"]
            filled += fill_qty

            if side == "BUY":
                self.portfolio.buy(ticker, fill_price, fill_qty)
                pnl = None
            else:
                pnl = self.portfolio.sell(ticker, fill_price, fill_qty)["pnl"]

            self.price_tracker.record(ticker, fill_price)       # price moves only on real trades
            entry = {"ticker": ticker, "side": side, "price": fill_price,
                     "qty": fill_qty, "status": "EXECUTED"}
            if pnl is not None:
                entry["pnl"] = pnl
            self.trade_log.append(entry)

            if self._anomaly_manager and self._TradeEvent:
                event = self._TradeEvent(ticker, fill_price, fill_qty,
                                         book.spread() or 0.0, side)
                self._anomaly_manager.process(event)

        status = result["status"]
        if filled == 0:
            message = (f"{side} {qty}x{ticker} not filled - no "
                       f"{'seller at or below' if side == 'BUY' else 'buyer at or above'} "
                       f"Rs{price:.2f}. Nothing was charged.")
        elif filled < qty:
            message = (f"{side} {filled} of {qty}x{ticker} filled - "
                       f"rest cancelled (not enough on the other side).")
        else:
            message = f"{side} {qty}x{ticker} - EXECUTED"

        return {"ok": True, "status": status, "trades": len(result["trades"]),
                "filled": filled, "message": message}

    def snapshot(self, ticker: str) -> dict:
        """Full snapshot of one ticker."""
        book    = self.order_books[ticker]
        return {
            "ticker":     ticker,
            "last_price": self.price_tracker.latest(ticker),
            "best_bid":   book.best_bid(),
            "best_ask":   book.best_ask(),
            "spread":     book.spread(),
            "mid_price":  book.mid_price(),
            "bid_depth":  len(book.bids),
            "ask_depth":  len(book.asks),
        }