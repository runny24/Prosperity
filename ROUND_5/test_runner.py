"""
test_runner.py — Fast local backtest engine for strategy testing.
Does NOT modify backtester_r5.py; standalone simulation.
"""
import csv, importlib.util, json, os, sys
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from typing import Dict, List

from datamodel import Listing, Observation, Order, OrderDepth, Trade, TradingState

LIMIT = 10

def load_prices(data_dir: str, days: List[int]):
    prices = {}
    for day in days:
        path = os.path.join(data_dir, f"prices_round_5_day_{day}.csv")
        with open(path) as f:
            for row in csv.DictReader(f, delimiter=";"):
                key = (day, int(row["timestamp"]))
                prices.setdefault(key, {})[row["product"]] = row
    return prices

def load_market_trades(data_dir: str, days: List[int]):
    result = defaultdict(list)
    for day in days:
        path = os.path.join(data_dir, f"trades_round_5_day_{day}.csv")
        with open(path) as f:
            for row in csv.DictReader(f, delimiter=";"):
                key = (day, int(row["timestamp"]))
                result[key].append(Trade(
                    symbol=row["symbol"], price=int(float(row["price"])),
                    quantity=int(row["quantity"]),
                    buyer=row.get("buyer") or None,
                    seller=row.get("seller") or None,
                    timestamp=int(row["timestamp"]),
                ))
    return result

def build_order_depth(row: dict) -> OrderDepth:
    od = OrderDepth()
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}", "").strip()
        bv = row.get(f"bid_volume_{i}", "").strip()
        ap = row.get(f"ask_price_{i}", "").strip()
        av = row.get(f"ask_volume_{i}", "").strip()
        if bp and bv: od.buy_orders[int(float(bp))] = int(float(bv))
        if ap and av: od.sell_orders[int(float(ap))] = -int(float(av))
    return od

def match_orders(orders, od, market_trades, product, positions, realized, timestamp):
    fills = []
    pos = positions.get(product, 0)
    total_qty = sum(o.quantity for o in orders)
    if pos + total_qty > LIMIT or pos + total_qty < -LIMIT:
        return []

    for t in market_trades:
        if t.symbol != product: continue
        if t.quantity > 0:
            for px in sorted(od.sell_orders):
                if px <= t.price:
                    vol = -od.sell_orders[px]; take = min(vol, t.quantity)
                    od.sell_orders[px] += take; t.quantity -= take
                    if t.quantity <= 0: break
        else:
            for px in sorted(od.buy_orders, reverse=True):
                if px >= t.price:
                    vol = od.buy_orders[px]; take = min(vol, -t.quantity)
                    od.buy_orders[px] -= take; t.quantity += take
                    if t.quantity >= 0: break

    for order in orders:
        if order.quantity > 0:
            remaining = order.quantity
            for ask_px in sorted(od.sell_orders):
                if ask_px > order.price or remaining <= 0: break
                avail = -od.sell_orders[ask_px]; fill = min(remaining, avail)
                if fill > 0:
                    od.sell_orders[ask_px] += fill; pos += fill
                    realized[product] -= fill * ask_px; remaining -= fill
                    fills.append(Trade(product, ask_px, fill, "SUBMISSION", None, timestamp))
        elif order.quantity < 0:
            remaining = -order.quantity
            for bid_px in sorted(od.buy_orders, reverse=True):
                if bid_px < order.price or remaining <= 0: break
                avail = od.buy_orders[bid_px]; fill = min(remaining, avail)
                if fill > 0:
                    od.buy_orders[bid_px] -= fill; pos -= fill
                    realized[product] += fill * bid_px; remaining -= fill
                    fills.append(Trade(product, bid_px, -fill, None, "SUBMISSION", timestamp))

    positions[product] = pos
    return fills

def run_simulation(trader, prices, market_trades):
    positions = defaultdict(int)
    realized  = defaultdict(float)
    trader_data = ""
    prev_own_trades = defaultdict(list)

    for (day, ts) in sorted(prices.keys()):
        tick_prices = prices[(day, ts)]
        order_depths = {p: build_order_depth(r) for p, r in tick_prices.items()}
        mkt_by_sym = defaultdict(list)
        for t in market_trades.get((day, ts), []):
            mkt_by_sym[t.symbol].append(t)

        state = TradingState(
            timestamp=ts, traderData=trader_data,
            listings={p: Listing(p, p, "XIRECS") for p in tick_prices},
            order_depths=order_depths,
            own_trades=dict(prev_own_trades),
            market_trades=dict(mkt_by_sym),
            position=dict(positions),
            observations=Observation(),
        )
        try:
            buf = StringIO()
            with redirect_stdout(buf), redirect_stderr(buf):
                result = trader.run(state)
            orders_dict, _, new_td = result if len(result) == 3 else (*result, "")
            trader_data = new_td or ""
        except Exception:
            orders_dict = {}

        this_tick_own = defaultdict(list)
        for product, orders in (orders_dict or {}).items():
            if product in order_depths and orders:
                for f in match_orders(orders, order_depths[product],
                                      mkt_by_sym[product], product,
                                      positions, realized, ts):
                    this_tick_own[product].append(f)
        prev_own_trades = this_tick_own

    last_mid = {}
    for tick_prices in prices.values():
        for p, row in tick_prices.items():
            mp = row.get("mid_price", "").strip()
            if mp: last_mid[p] = float(mp)

    final_pnl = {}
    for p in set(list(realized) + list(last_mid)):
        final_pnl[p] = realized.get(p, 0.0) + positions.get(p, 0) * last_mid.get(p, 0.0)
    return final_pnl, dict(positions)

def print_results(pnl, positions, label=""):
    rows = sorted(pnl.items(), key=lambda x: -x[1])
    total = sum(pnl.values())
    print(f"\n{'='*60}")
    if label: print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'Product':<35} {'PnL':>10}  {'Pos':>5}")
    print(f"  {'-'*53}")
    for p, v in rows:
        if v != 0 or positions.get(p, 0) != 0:
            print(f"  {p:<35} {v:>10,.0f}  {positions.get(p,0):>5}")
    print(f"  {'-'*53}")
    print(f"  {'TOTAL':<35} {total:>10,.0f}")
    print(f"{'='*60}\n")

def load_trader(path):
    spec = importlib.util.spec_from_file_location("trader_module", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Trader()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--trader", required=True)
    p.add_argument("--days", default="2,3,4")
    p.add_argument("--data", default=".")
    args = p.parse_args()

    days = [int(d) for d in args.days.split(",")]
    trader = load_trader(args.trader)
    prices = load_prices(args.data, days)
    mkt    = load_market_trades(args.data, days)
    print(f"Loaded {len(prices):,} ticks | days={days}")
    pnl, pos = run_simulation(trader, prices, mkt)
    print_results(pnl, pos, label=f"{os.path.basename(args.trader)} | days={days}")

    os.makedirs("outputs", exist_ok=True)
    results = {"total_pnl": sum(pnl.values()), "pnl_by_product": pnl,
               "final_positions": pos, "days": days,
               "trader": os.path.basename(args.trader)}
    with open("outputs/latest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to outputs/latest_results.json")
