"""
backtest.py — Prosperity 4 Round 1 custom backtester

Usage:
    python backtest.py          # days -2, -1, 0
    python backtest.py -v       # verbose (prints trader logs)
"""

import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

from datamodel import Listing, Observation, Order, OrderDepth, Trade, TradingState
from trader import Trader

# ─────────────────────────────────────────────────────────────
PRODUCTS = {"ASH_COATED_OSMIUM": 50, "INTARIAN_PEPPER_ROOT": 50}
DATA_DIR = Path("data")
DAYS = (-2, -1, 0)


# ─────────────────────────────────────────────────────────────
#  Load price CSV  →  ts → {product: (OrderDepth, mid)}
# ─────────────────────────────────────────────────────────────
def load_prices(day: int) -> dict:
    ticks: dict = defaultdict(dict)
    path = DATA_DIR / f"prices_round_1_day_{day}.csv"
    with open(path) as f:
        for row in csv.DictReader(f, delimiter=";"):
            ts = int(row["timestamp"])
            product = row["product"]
            od = OrderDepth()
            for i in (1, 2, 3):
                bp = row.get(f"bid_price_{i}", "").strip()
                bv = row.get(f"bid_volume_{i}", "").strip()
                ap = row.get(f"ask_price_{i}", "").strip()
                av = row.get(f"ask_volume_{i}", "").strip()
                if bp and bv:
                    od.buy_orders[int(float(bp))] = int(bv)
                if ap and av:
                    od.sell_orders[int(float(ap))] = -int(av)  # negative convention
            mid_str = row.get("mid_price", "").strip()
            mid = float(mid_str) if mid_str else None
            ticks[ts][product] = (od, mid)
    return ticks


# ─────────────────────────────────────────────────────────────
#  Load trades CSV  →  ts → {symbol: [Trade, ...]}
# ─────────────────────────────────────────────────────────────
def load_market_trades(day: int) -> dict:
    trades: dict = defaultdict(lambda: defaultdict(list))
    path = DATA_DIR / f"trades_round_1_day_{day}.csv"
    with open(path) as f:
        for row in csv.DictReader(f, delimiter=";"):
            ts = int(row["timestamp"])
            sym = row["symbol"]
            price = int(float(row["price"]))
            qty = int(row["quantity"])
            buyer = row.get("buyer", "") or ""
            seller = row.get("seller", "") or ""
            trades[ts][sym].append(Trade(sym, price, qty, buyer, seller, ts))
    return trades


# ─────────────────────────────────────────────────────────────
#  Match trader orders against the orderbook snapshot
#  Returns: list of (price, signed_qty), new position
#    signed_qty > 0 = bought, < 0 = sold
# ─────────────────────────────────────────────────────────────
def fill_orders(
    orders: list,
    od: OrderDepth,
    position: int,
    limit: int,
) -> tuple:
    fills = []
    pos = position

    # process buys then sells
    for order in sorted(orders, key=lambda o: -o.quantity):
        if order.quantity > 0:  # buy order
            qty_left = min(order.quantity, limit - pos)
            for ask in sorted(od.sell_orders):
                if ask > order.price or qty_left <= 0:
                    break
                available = -od.sell_orders[ask]
                filled = min(qty_left, available)
                fills.append((ask, filled))
                pos += filled
                qty_left -= filled

        elif order.quantity < 0:  # sell order
            qty_left = min(-order.quantity, limit + pos)
            for bid in sorted(od.buy_orders, reverse=True):
                if bid < order.price or qty_left <= 0:
                    break
                available = od.buy_orders[bid]
                filled = min(qty_left, available)
                fills.append((bid, -filled))
                pos -= filled
                qty_left -= filled

    return fills, pos


# ─────────────────────────────────────────────────────────────
#  Run backtest across all days
# ─────────────────────────────────────────────────────────────
def run_backtest(verbose: bool = False):
    trader = Trader()
    positions: dict = defaultdict(int)
    cash: dict = defaultdict(float)
    trader_data = ""

    print(f"\n{'Day':>5}  {'Day PnL':>12}  {'Cumul PnL':>12}  {'Max PnL':>12}")
    print("-" * 50)

    cumul_pnl_start = 0.0

    for day in DAYS:
        ticks = load_prices(day)
        market_trades = load_market_trades(day)
        own_trades_prev: dict = defaultdict(list)
        day_pnls = []

        for ts in sorted(ticks):
            tick = ticks[ts]

            listings = {p: Listing(p, p, 1) for p in tick}
            order_depths = {p: od for p, (od, _) in tick.items()}
            mkt = {sym: list(tlist) for sym, tlist in market_trades[ts].items()}

            state = TradingState(
                traderData=trader_data,
                timestamp=ts,
                listings=listings,
                order_depths=order_depths,
                own_trades=dict(own_trades_prev),
                market_trades=mkt,
                position=dict(positions),
                observations=Observation({}, {}),
            )

            # suppress trader's logger.flush() stdout unless verbose
            if verbose:
                orders_dict, _, trader_data = trader.run(state)
            else:
                buf = io.StringIO()
                _real_stdout = sys.stdout
                sys.stdout = buf
                try:
                    orders_dict, _, trader_data = trader.run(state)
                finally:
                    sys.stdout = _real_stdout

            own_trades_prev = defaultdict(list)

            for product, orders in orders_dict.items():
                if product not in order_depths:
                    continue
                limit = PRODUCTS.get(product, 50)
                fills, new_pos = fill_orders(
                    orders, order_depths[product], positions[product], limit
                )
                for price, qty in fills:
                    cash[product] -= price * qty   # buy: cash down; sell: cash up
                    positions[product] += qty
                    own_trades_prev[product].append(
                        Trade(product, price, abs(qty), "", "", ts)
                    )

            # Mark-to-market PnL = total cash + position * mid price
            mtm = sum(cash[p] for p in PRODUCTS)
            for product, (_, mid) in tick.items():
                if mid is not None:
                    mtm += positions[product] * mid
            day_pnls.append(mtm)

        cumul_pnl_end = day_pnls[-1] if day_pnls else 0.0
        day_pnl = cumul_pnl_end - cumul_pnl_start
        cumul_pnl_start = cumul_pnl_end

        print(
            f"{day:>5}  {day_pnl:>12.1f}  {cumul_pnl_end:>12.1f}  {max(day_pnls):>12.1f}"
        )

    print("-" * 50)
    print(f"\nFinal positions : {dict(positions)}")
    print(f"Final cash      : { {p: round(v, 1) for p, v in cash.items()} }")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    verbose = "-v" in sys.argv
    run_backtest(verbose=verbose)
