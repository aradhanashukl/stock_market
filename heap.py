# ── STEP 1: Heap (Priority Queue) ──────────────────────────────────────────
#
# WHY A HEAP?
#   - Buyers want to pay the HIGHEST price  → Max-Heap (best bid on top)
#   - Sellers want the LOWEST ask price     → Min-Heap (best ask on top)
#   - Both give us O(log n) insert/remove, O(1) peek
#
# Python's `heapq` is a Min-Heap by default.
# For a Max-Heap we just negate the price.
# ───────────────────────────────────────────────────────────────────────────

import heapq
from dataclasses import dataclass, field
from typing import Literal

OrderSide = Literal["BUY", "SELL"]


@dataclass
class Order:
    order_id: int
    ticker: str
    side: OrderSide
    price: float
    quantity: int

    def __repr__(self):
        return f"Order(#{self.order_id} {self.side} {self.quantity}×{self.ticker} @{self.price})"


class MaxHeap:
    """
    Max-Heap for BID orders.
    Highest price order is always at the top (best bid).
    """

    def __init__(self):
        self._heap = []  # stores (-price, order_id, Order)  ← negate for max behaviour

    def push(self, order: Order):
        # Negate price so heapq (min-heap) gives us max behaviour
        heapq.heappush(self._heap, (-order.price, order.order_id, order))

    def pop(self) -> Order:
        _, _, order = heapq.heappop(self._heap)
        return order

    def peek(self) -> Order | None:
        if self._heap:
            return self._heap[0][2]
        return None

    def __len__(self):
        return len(self._heap)

    def top_n(self, n=5) -> list[Order]:
        # Return top-n items without destroying the heap
        sorted_items = sorted(self._heap)          # sort by (-price, id)
        return [item[2] for item in sorted_items[:n]]


class MinHeap:
    """
    Min-Heap for ASK orders.
    Lowest price order is always at the top (best ask).
    """

    def __init__(self):
        self._heap = []  # stores (price, order_id, Order)

    def push(self, order: Order):
        heapq.heappush(self._heap, (order.price, order.order_id, order))

    def pop(self) -> Order:
        _, _, order = heapq.heappop(self._heap)
        return order

    def peek(self) -> Order | None:
        if self._heap:
            return self._heap[0][2]
        return None

    def __len__(self):
        return len(self._heap)

    def top_n(self, n=5) -> list[Order]:
        sorted_items = sorted(self._heap)
        return [item[2] for item in sorted_items[:n]]


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Max-Heap (BID side) ===")
    bids = MaxHeap()
    bids.push(Order(1, "AAPL", "BUY", price=181.50, quantity=100))
    bids.push(Order(2, "AAPL", "BUY", price=182.00, quantity=50))   # ← best bid
    bids.push(Order(3, "AAPL", "BUY", price=180.75, quantity=200))

    print(f"Best bid (peek): {bids.peek()}")        # should be 182.00
    print(f"All bids (top 3): {bids.top_n()}")
    print(f"Pop best bid: {bids.pop()}")             # removes 182.00
    print(f"New best bid: {bids.peek()}")            # now 181.50

    print("\n=== Min-Heap (ASK side) ===")
    asks = MinHeap()
    asks.push(Order(4, "AAPL", "SELL", price=182.50, quantity=75))
    asks.push(Order(5, "AAPL", "SELL", price=181.80, quantity=120))  # ← best ask
    asks.push(Order(6, "AAPL", "SELL", price=183.00, quantity=60))

    print(f"Best ask (peek): {asks.peek()}")         # should be 181.80
    print(f"All asks (top 3): {asks.top_n()}")
    print(f"Pop best ask: {asks.pop()}")              # removes 181.80
    print(f"New best ask: {asks.peek()}")             # now 182.50

    print("\n=== Spread ===")
    best_bid = bids.peek().price
    best_ask = asks.peek().price
    print(f"Bid: {best_bid}  Ask: {best_ask}  Spread: {best_ask - best_bid:.2f}")
    bids.push(Order(7, "AAPL", "BUY", price=185.00, quantity=10))
    print(f"New best bid after 185.00 insert: {bids.peek()}")