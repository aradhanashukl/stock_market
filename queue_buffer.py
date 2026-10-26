# ── STEP 3: Queue + CircularBuffer ─────────────────────────────────────────
#
# TradeQueue     → FIFO queue for trade execution log
#                  first trade in = first trade out
#                  max size prevents memory growing forever
#
# CircularBuffer → fixed-size ring buffer for price history
#                  when full, oldest price is overwritten
#                  used to power price charts
#
# ───────────────────────────────────────────────────────────────────────────


# ── FIFO Queue ──────────────────────────────────────────────────────────────
class TradeQueue:
    """
    A simple FIFO queue with a max size.
    When full, the oldest entry is dropped automatically.

    Internal structure: Python list used as a queue
      enqueue → append to right  (newest)
      dequeue → pop from left    (oldest)
    """

    def __init__(self, max_size: int = 100):
        self._queue   = []
        self.max_size = max_size

    def enqueue(self, item):
        """Add a new item to the back of the queue."""
        self._queue.append(item)
        if len(self._queue) > self.max_size:
            self._queue.pop(0)   # drop oldest when full

    def dequeue(self):
        """Remove and return the front item (oldest)."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._queue.pop(0)

    def peek(self):
        """Look at the front item without removing it."""
        if self.is_empty():
            return None
        return self._queue[0]

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self):
        return len(self._queue)

    def all(self) -> list:
        """Return all items oldest → newest."""
        return list(self._queue)

    def display(self, label="Trade Queue"):
        print(f"\n── {label} (oldest → newest) ──")
        if self.is_empty():
            print("  (empty)")
            return
        for i, item in enumerate(self._queue):
            print(f"  [{i}] {item}")


# ── Circular Buffer ─────────────────────────────────────────────────────────
class CircularBuffer:
    """
    Fixed-size ring buffer for price history.

    Uses a single list of fixed size + a head pointer.
    head always points to where the NEXT write will go.
    When head reaches the end it wraps back to 0.

    Memory is always exactly `capacity` slots — never grows.

    Visual with capacity=5:
      After 3 pushes:  [10, 20, 30,  _,  _]   head=3  size=3
      After 5 pushes:  [10, 20, 30, 40, 50]   head=0  size=5  (full)
      After 6 pushes:  [60, 20, 30, 40, 50]   head=1  size=5  (60 overwrote 10)
      After 7 pushes:  [60, 70, 30, 40, 50]   head=2  size=5  (70 overwrote 20)
    """

    def __init__(self, capacity: int):
        self._buf      = [None] * capacity   # fixed-size list
        self._capacity = capacity
        self._head     = 0                   # next write position
        self._size     = 0                   # how many slots are filled

    def push(self, value):
        """Write value at head, then advance head (wrapping around)."""
        self._buf[self._head] = value
        self._head = (self._head + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def to_list(self) -> list:
        """
        Return all stored values in chronological order (oldest → newest).
        Uses modular arithmetic to unwrap the ring.
        """
        if self._size == 0:
            return []

        # oldest item starts at (head - size) wrapping around
        start = (self._head - self._size) % self._capacity
        return [self._buf[(start + i) % self._capacity] for i in range(self._size)]

    def latest(self):
        """Return the most recently pushed value."""
        if self._size == 0:
            return None
        return self._buf[(self._head - 1) % self._capacity]

    def is_full(self) -> bool:
        return self._size == self._capacity

    def __len__(self):
        return self._size

    def display(self, label="Circular Buffer"):
        print(f"\n── {label} ──")
        print(f"  Raw storage : {self._buf}")
        print(f"  Head pointer: {self._head}")
        print(f"  Size/Cap    : {self._size}/{self._capacity}")
        print(f"  In order    : {self.to_list()}")


# ── Tests ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Test 1: TradeQueue ───────────────────────────────────────────────────
    print("=" * 45)
    print("TEST 1 — TradeQueue (FIFO)")
    print("=" * 45)

    q = TradeQueue(max_size=5)

    q.enqueue("TRADE: BUY  10×AAPL @ ₹182.00")
    q.enqueue("TRADE: SELL 5×AAPL  @ ₹183.00")
    q.enqueue("TRADE: BUY  20×TSLA @ ₹245.00")
    q.display()

    print(f"\n  peek (front) : {q.peek()}")
    print(f"  dequeue      : {q.dequeue()}")   # removes oldest
    print(f"  peek after   : {q.peek()}")       # now 2nd trade is front
    q.display("After dequeue")

    print("\n  Adding 4 more to test max_size=5 overflow...")
    q.enqueue("TRADE: SELL 15×AAPL @ ₹184.00")
    q.enqueue("TRADE: BUY  8×GOOGL @ ₹175.00")
    q.enqueue("TRADE: SELL 12×TSLA @ ₹246.00")
    q.enqueue("TRADE: BUY  30×INFY @ ₹19.50")   # oldest gets dropped
    q.display("After overflow (max=5, oldest dropped)")

    # ── Test 2: CircularBuffer ───────────────────────────────────────────────
    print("\n" + "=" * 45)
    print("TEST 2 — CircularBuffer (price history)")
    print("=" * 45)

    buf = CircularBuffer(capacity=5)

    print("\n  Pushing prices one by one...")
    for price in [100, 101, 102]:
        buf.push(price)
        buf.display(f"After push({price})")

    print("\n  Filling to capacity...")
    buf.push(103)
    buf.push(104)
    buf.display("Full buffer (5/5)")

    print("\n  Pushing more — watch oldest get overwritten...")
    buf.push(105)
    buf.display("After push(105) — 100 is gone")
    buf.push(106)
    buf.display("After push(106) — 101 is gone")

    print(f"\n  Latest price : {buf.latest()}")
    print(f"  All prices   : {buf.to_list()}")
    print(f"  Is full      : {buf.is_full()}")

    # ── Test 3: Realistic price feed simulation ──────────────────────────────
    print("\n" + "=" * 45)
    print("TEST 3 — Simulated price feed (60 ticks, buffer=10)")
    print("=" * 45)

    import random
    price_history = CircularBuffer(capacity=10)
    price = 182.00

    for tick in range(60):
        price += round(random.uniform(-0.5, 0.5), 2)
        price = round(max(1, price), 2)
        price_history.push(price)

    print(f"\n  60 prices pushed into a size-10 buffer")
    print(f"  Memory used : always 10 slots (never more)")
    print(f"  Last 10 prices: {price_history.to_list()}")
    print(f"  Current price : ₹{price_history.latest()}")