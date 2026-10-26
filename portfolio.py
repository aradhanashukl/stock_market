# ── STEP 4: Portfolio ──────────────────────────────────────────────────────
#
# DSA USED: HashMap (Python dict)
#
#   holdings = {
#       "AAPL": { "qty": 150, "avg_cost": 181.50 },
#       "TSLA": { "qty":  50, "avg_cost": 244.00 },
#   }
#
#   O(1) lookup, insert, update — by ticker key
#
# FEATURES:
#   - buy()        → adds shares, recalculates average cost
#   - sell()       → removes shares, calculates realised P&L
#   - unrealised() → live P&L based on current market price
#   - summary()    → full portfolio snapshot
#
# ───────────────────────────────────────────────────────────────────────────


class Portfolio:

    def __init__(self, starting_cash: float = 10_000.0):
        self.cash          = starting_cash
        self.holdings      = {}      # HashMap: ticker → {qty, avg_cost}
        self.realised_pnl  = 0.0    # total profit/loss from closed trades
        self.trade_count   = 0

    # ── Buy ─────────────────────────────────────────────────────────────────
    def buy(self, ticker: str, price: float, qty: int) -> dict:
        """
        Buy `qty` shares of `ticker` at `price`.
        Updates average cost using weighted average formula:
            new_avg = (old_qty * old_avg + new_qty * price) / (old_qty + new_qty)
        Returns a result dict.
        """
        cost = price * qty

        # ── Validation ──────────────────────────────────────────────────────
        if qty <= 0:
            return {"ok": False, "reason": "Quantity must be positive"}
        if price <= 0:
            return {"ok": False, "reason": "Price must be positive"}
        if cost > self.cash:
            return {
                "ok":     False,
                "reason": f"Insufficient funds — need ₹{cost:.2f}, have ₹{self.cash:.2f}"
            }

        # ── Execute ─────────────────────────────────────────────────────────
        self.cash -= cost

        if ticker in self.holdings:
            h = self.holdings[ticker]
            total_spent  = h["qty"] * h["avg_cost"] + cost   # old + new
            h["qty"]     += qty
            h["avg_cost"] = total_spent / h["qty"]            # weighted avg
        else:
            # First purchase — create new entry in HashMap
            self.holdings[ticker] = {"qty": qty, "avg_cost": price}

        self.trade_count += 1
        return {
            "ok":      True,
            "action":  "BUY",
            "ticker":  ticker,
            "qty":     qty,
            "price":   price,
            "cost":    cost,
            "avg_cost": round(self.holdings[ticker]["avg_cost"], 2),
            "cash_left": round(self.cash, 2),
        }

    # ── Sell ────────────────────────────────────────────────────────────────
    def sell(self, ticker: str, price: float, qty: int) -> dict:
        """
        Sell `qty` shares of `ticker` at `price`.
        Calculates realised P&L = (sell_price - avg_cost) * qty
        """
        # ── Validation ──────────────────────────────────────────────────────
        if qty <= 0:
            return {"ok": False, "reason": "Quantity must be positive"}
        if price <= 0:
            return {"ok": False, "reason": "Price must be positive"}
        if ticker not in self.holdings:
            return {"ok": False, "reason": f"You don't own any {ticker}"}
        if self.holdings[ticker]["qty"] < qty:
            return {
                "ok":     False,
                "reason": f"Not enough shares — have {self.holdings[ticker]['qty']}, want to sell {qty}"
            }

        # ── Execute ─────────────────────────────────────────────────────────
        h          = self.holdings[ticker]
        avg_cost   = h["avg_cost"]
        pnl        = (price - avg_cost) * qty       # profit or loss on this sale
        proceeds   = price * qty

        self.cash          += proceeds
        self.realised_pnl  += pnl
        h["qty"]           -= qty
        self.trade_count   += 1

        if h["qty"] == 0:
            del self.holdings[ticker]               # remove key from HashMap

        return {
            "ok":           True,
            "action":       "SELL",
            "ticker":       ticker,
            "qty":          qty,
            "price":        price,
            "proceeds":     round(proceeds, 2),
            "avg_cost":     round(avg_cost, 2),
            "realised_pnl": round(pnl, 2),
            "cash_now":     round(self.cash, 2),
        }

    # ── Unrealised P&L ──────────────────────────────────────────────────────
    def unrealised_pnl(self, market_prices: dict) -> dict:
        """
        Calculate live (unrealised) P&L for all holdings.
        market_prices = { "AAPL": 185.00, "TSLA": 250.00, ... }
        """
        result = {}
        for ticker, h in self.holdings.items():
            if ticker not in market_prices:
                continue
            current   = market_prices[ticker]
            pnl       = (current - h["avg_cost"]) * h["qty"]
            pnl_pct   = ((current - h["avg_cost"]) / h["avg_cost"]) * 100
            result[ticker] = {
                "qty":       h["qty"],
                "avg_cost":  round(h["avg_cost"], 2),
                "current":   current,
                "pnl":       round(pnl, 2),
                "pnl_pct":   round(pnl_pct, 2),
            }
        return result

    # ── Portfolio value ──────────────────────────────────────────────────────
    def total_value(self, market_prices: dict) -> float:
        equity = sum(
            h["qty"] * market_prices.get(ticker, h["avg_cost"])
            for ticker, h in self.holdings.items()
        )
        return round(self.cash + equity, 2)

    # ── Summary display ──────────────────────────────────────────────────────
    def summary(self, market_prices: dict = None):
        mp = market_prices or {}
        pnl_data = self.unrealised_pnl(mp)

        print(f"\n{'═'*50}")
        print(f"  PORTFOLIO SUMMARY")
        print(f"{'═'*50}")
        print(f"  Cash            : ₹{self.cash:>10.2f}")
        print(f"  Realised P&L    : ₹{self.realised_pnl:>10.2f}")
        print(f"  Trades executed : {self.trade_count}")
        print(f"\n  {'Ticker':<8} {'Qty':>5} {'AvgCost':>9} {'Current':>9} {'P&L':>10} {'%':>7}")
        print(f"  {'─'*8} {'─'*5} {'─'*9} {'─'*9} {'─'*10} {'─'*7}")

        if not self.holdings:
            print("  (no holdings)")
        else:
            for ticker, h in self.holdings.items():
                cur  = mp.get(ticker, h["avg_cost"])
                pnl  = pnl_data.get(ticker, {}).get("pnl", 0)
                pct  = pnl_data.get(ticker, {}).get("pnl_pct", 0)
                sign = "+" if pnl >= 0 else ""
                print(f"  {ticker:<8} {h['qty']:>5} ₹{h['avg_cost']:>8.2f} ₹{cur:>8.2f} {sign}₹{pnl:>8.2f} {sign}{pct:.1f}%")

        if mp:
            print(f"\n  Total value     : ₹{self.total_value(mp):>10.2f}")
        print(f"{'═'*50}\n")


# ── Tests ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    p = Portfolio(starting_cash=10_000)

    print("=" * 50)
    print("TEST 1 — Basic buy and average cost")
    print("=" * 50)

    r = p.buy("AAPL", price=180.00, qty=20)
    print(f"\n  {r['action']} {r['qty']}×{r['ticker']} @ ₹{r['price']}")
    print(f"  Avg cost   : ₹{r['avg_cost']}")
    print(f"  Cash left  : ₹{r['cash_left']}")

    # Buy more AAPL at a higher price — avg cost should rise
    r = p.buy("AAPL", price=184.00, qty=30)
    print(f"\n  {r['action']} {r['qty']}×{r['ticker']} @ ₹{r['price']}")
    print(f"  New avg cost: ₹{r['avg_cost']}  (was 180.00, now blended)")
    print(f"  Cash left   : ₹{r['cash_left']}")

    # Buy a second stock
    r = p.buy("TSLA", price=245.00, qty=3)
    print(f"\n  {r['action']} {r['qty']}×{r['ticker']} @ ₹{r['price']}")
    print(f"  Cash left  : ₹{r['cash_left']}")

    p.summary(market_prices={"AAPL": 186.00, "TSLA": 250.00})

    print("=" * 50)
    print("TEST 2 — Sell and realised P&L")
    print("=" * 50)

    r = p.sell("AAPL", price=186.00, qty=10)
    print(f"\n  {r['action']} {r['qty']}×{r['ticker']} @ ₹{r['price']}")
    print(f"  Avg cost      : ₹{r['avg_cost']}")
    print(f"  Realised P&L  : ₹{r['realised_pnl']}")
    print(f"  Cash now      : ₹{r['cash_now']}")

    p.summary(market_prices={"AAPL": 186.00, "TSLA": 250.00})

    print("=" * 50)
    print("TEST 3 — Sell all shares (ticker removed from HashMap)")
    print("=" * 50)

    remaining = p.holdings.get("TSLA", {}).get("qty", 0)
    r = p.sell("TSLA", price=255.00, qty=remaining)
    print(f"\n  Sold all {remaining} TSLA @ ₹255.00")
    print(f"  Realised P&L  : ₹{r['realised_pnl']}")
    print(f"  TSLA in holdings: {'TSLA' in p.holdings}  ← removed from HashMap")

    p.summary(market_prices={"AAPL": 186.00})

    print("=" * 50)
    print("TEST 4 — Validation checks")
    print("=" * 50)

    print(f"\n  Try selling stock not owned:")
    r = p.sell("GOOGL", price=175.00, qty=10)
    print(f"  → {r}")

    print(f"\n  Try buying more than cash allows:")
    r = p.buy("AAPL", price=500.00, qty=1000)
    print(f"  → {r}")

    print(f"\n  Try selling more than owned:")
    r = p.sell("AAPL", price=186.00, qty=9999)
    print(f"  → {r}")