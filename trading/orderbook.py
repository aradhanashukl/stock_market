# ── STEP 2: OrderBook ──────────────────────────────────────────────────────
#
# HOW MATCHING WORKS:
#   1. BUY  orders go into the Max-Heap  (highest price on top)
#   2. SELL orders go into the Min-Heap  (lowest price on top)
#   3. After every new order we check:
#        if best_bid.price >= best_ask.price → MATCH → trade executes
#
# Example:
#   BUY  @ 182.00  ──┐
#                     ├── 182.00 >= 181.50 → TRADE at 181.50 ✅
#   SELL @ 181.50  ──┘
#
# ───────────────────────────────────────────────────────────────────────────

from heap import MaxHeap, MinHeap, Order   # ← your Step 1 file


class Trade:
    """Represents a matched trade between a buyer and seller."""

    def __init__(self, buy_order: Order, sell_order: Order, price: float, quantity: int):
        self.buy_order  = buy_order
        self.sell_order = sell_order
        self.price      = price
        self.quantity   = quantity

    def __repr__(self):
        return (
            f"TRADE  {self.quantity}×{self.buy_order.ticker}"
            f"  Buyer #{self.buy_order.order_id}"
            f" ←→ Seller #{self.sell_order.order_id}"
            f"  @ ₹{self.price}"
        )


class OrderBook:
    """
    Maintains bids and asks for ONE ticker.
    Automatically matches orders when prices cross.
    """

    def __init__(self, ticker: str):
        self.ticker     = ticker
        self.bids       = MaxHeap()   # BUY  side — highest price on top
        self.asks       = MinHeap()   # SELL side — lowest  price on top
        self.trade_log  = []          # list of executed Trade objects
        self._order_id  = 1

    def new_order_id(self) -> int:
        oid = self._order_id
        self._order_id += 1
        return oid

    # ── Place an order ──────────────────────────────────────────────────────
    def place_order(self, side: str, price: float, quantity: int) -> list[Trade]:
        """
        Add a new order and immediately try to match it.
        Returns a list of trades that were executed (can be empty).
        """
        order = Order(
            order_id = self.new_order_id(),
            ticker   = self.ticker,
            side     = side,
            price    = price,
            quantity = quantity,
        )

        if side == "BUY":
            self.bids.push(order)
        elif side == "SELL":
            self.asks.push(order)
        else:
            raise ValueError(f"side must be BUY or SELL, got: {side}")

        # Try to match after every new order
        return self._match()

    # ── Matching engine ─────────────────────────────────────────────────────
    def _match(self) -> list[Trade]:
        """
        Core matching loop.
        Keeps matching as long as best_bid >= best_ask.
        """
        executed = []

        while self.bids and self.asks:
            best_bid = self.bids.peek()   # highest buy price
            best_ask = self.asks.peek()   # lowest  sell price

            # No match possible — spread is positive
            if best_bid.price < best_ask.price:
                break

            # ── MATCH FOUND ─────────────────────────────────────────────────
            # Trade happens at the ask price (price taker rule)
            trade_price = best_ask.price
            trade_qty   = min(best_bid.quantity, best_ask.quantity)

            trade = Trade(
                buy_order  = best_bid,
                sell_order = best_ask,
                price      = trade_price,
                quantity   = trade_qty,
            )
            executed.append(trade)
            self.trade_log.append(trade)

            # ── Update quantities or remove fully filled orders ─────────────
            best_bid.quantity  -= trade_qty
            best_ask.quantity  -= trade_qty

            if best_bid.quantity == 0:
                self.bids.pop()    # fully filled — remove from heap

            if best_ask.quantity == 0:
                self.asks.pop()    # fully filled — remove from heap

        return executed

    # ── Snapshot helpers ────────────────────────────────────────────────────
    def best_bid(self) -> float | None:
        o = self.bids.peek()
        return o.price if o else None

    def best_ask(self) -> float | None:
        o = self.asks.peek()
        return o.price if o else None

    def spread(self) -> float | None:
        if self.best_bid() and self.best_ask():
            return round(self.best_ask() - self.best_bid(), 2)
        return None

    def display(self):
        """Print a simple order book snapshot."""
        print(f"\n{'─'*40}")
        print(f"  ORDER BOOK — {self.ticker}")
        print(f"{'─'*40}")

        print("  ASK side (sellers):")
        asks = self.asks.top_n(5)
        for o in reversed(asks):
            print(f"    ₹{o.price:<10.2f}  qty: {o.quantity}  #{o.order_id}")

        spread = self.spread()
        if spread is not None:
            print(f"\n  {'─'*20}")
            print(f"  Spread: ₹{spread}")
            print(f"  {'─'*20}\n")

        print("  BID side (buyers):")
        for o in self.bids.top_n(5):
            print(f"    ₹{o.price:<10.2f}  qty: {o.quantity}  #{o.order_id}")

        print(f"{'─'*40}")
        print(f"  Total trades executed: {len(self.trade_log)}")
        print(f"{'─'*40}\n")


# ── Test it ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    book = OrderBook("AAPL")

    print("=== Adding BUY orders ===")
    book.place_order("BUY",  price=181.00, quantity=100)
    book.place_order("BUY",  price=182.00, quantity=50)
    book.place_order("BUY",  price=180.50, quantity=200)

    print("=== Adding SELL orders (no match yet) ===")
    book.place_order("SELL", price=183.00, quantity=75)
    book.place_order("SELL", price=184.00, quantity=60)

    book.display()  # should show clean spread, no trades yet

    print("=== Adding a SELL @ 181.50 — will match with BUY @ 182.00 ===")
    trades = book.place_order("SELL", price=181.50, quantity=30)
    for t in trades:
        print(f"  ✅ {t}")

    book.display()  # BUY #2 quantity drops from 50 → 20

    print("=== Adding a BUY @ 184.50 — will match with SELL @ 183.00 ===")
    trades = book.place_order("BUY", price=184.50, quantity=100)
    for t in trades:
        print(f"  ✅ {t}")

    book.display()  # SELL @ 183.00 fully filled, BUY partially filled

    print("=== Partial fill test — big BUY sweeps multiple SELL levels ===")
    book.place_order("SELL", price=181.00, quantity=10)
    book.place_order("SELL", price=181.50, quantity=10)
    book.place_order("SELL", price=182.00, quantity=10)
    trades = book.place_order("BUY",  price=185.00, quantity=100)
    print(f"  Trades from one BUY order: {len(trades)}")
    for t in trades:
        print(f"  ✅ {t}")