"""
compare_1000tick.py — Compare two traders on first 1000 ticks of each day.

Usage: python compare_1000tick.py trader_r5_v13.py trader_r5_v14.py
"""
import csv, importlib.util, json, sys, os
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from datamodel import Listing, Observation, Trade, OrderDepth, TradingState

LIMIT = 10
DATA_DIR = "."
DAYS = [2, 3, 4]
MAX_TS = 99900


def load_prices(days, max_ts=None):
    prices = {}
    for day in days:
        path = os.path.join(DATA_DIR, f"prices_round_5_day_{day}.csv")
        with open(path) as f:
            for row in csv.DictReader(f, delimiter=";"):
                ts = int(row["timestamp"])
                if max_ts is not None and ts > max_ts:
                    continue
                key = (day, ts)
                prices.setdefault(key, {})[row["product"]] = row
    return prices


def load_market_trades(days, max_ts=None):
    result = defaultdict(list)
    for day in days:
        path = os.path.join(DATA_DIR, f"trades_round_5_day_{day}.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f, delimiter=";"):
                ts = int(row["timestamp"])
                if max_ts is not None and ts > max_ts:
                    continue
                key = (day, ts)
                result[key].append(Trade(
                    symbol=row["symbol"], price=int(float(row["price"])),
                    quantity=int(row["quantity"]),
                    buyer=row.get("buyer") or None, seller=row.get("seller") or None,
                    timestamp=ts,
                ))
    return result


def build_od(row):
    od = OrderDepth()
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}", "").strip()
        bv = row.get(f"bid_volume_{i}", "").strip()
        ap = row.get(f"ask_price_{i}", "").strip()
        av = row.get(f"ask_volume_{i}", "").strip()
        if bp and bv: od.buy_orders[int(float(bp))] = int(float(bv))
        if ap and av: od.sell_orders[int(float(ap))] = -int(float(av))
    return od


def match_orders(orders, od, mkt_trades, product, positions, realized, ts):
    pos = positions.get(product, 0)
    total_qty = sum(o.quantity for o in orders)
    if pos + total_qty > LIMIT or pos + total_qty < -LIMIT:
        return

    for t in mkt_trades:
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
        elif order.quantity < 0:
            remaining = -order.quantity
            for bid_px in sorted(od.buy_orders, reverse=True):
                if bid_px < order.price or remaining <= 0: break
                avail = od.buy_orders[bid_px]; fill = min(remaining, avail)
                if fill > 0:
                    od.buy_orders[bid_px] -= fill; pos -= fill
                    realized[product] += fill * bid_px; remaining -= fill
    positions[product] = pos


def run_trader_on_day(trader_path, prices, mkt, day):
    spec = importlib.util.spec_from_file_location("trader", trader_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    trader = mod.Trader()

    positions = defaultdict(int)
    realized = defaultdict(float)
    trader_data = ""

    last_mid = {}
    prev_own = defaultdict(list)

    for (d, ts) in sorted(prices.keys()):
        if d != day:
            continue
        tick_prices = prices[(d, ts)]
        order_depths = {p: build_od(r) for p, r in tick_prices.items()}
        mkt_by_sym = defaultdict(list)
        for t in mkt.get((d, ts), []):
            mkt_by_sym[t.symbol].append(t)

        for p, row in tick_prices.items():
            mp = row.get("mid_price", "")
            if mp: last_mid[p] = float(mp)

        state = TradingState(
            timestamp=ts, traderData=trader_data,
            listings={p: Listing(p, p, "XIRECS") for p in tick_prices},
            order_depths=order_depths,
            own_trades=dict(prev_own), market_trades=dict(mkt_by_sym),
            position=dict(positions), observations=Observation(),
        )

        buf = StringIO()
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                result = trader.run(state)
            orders_dict, _, new_td = result if len(result) == 3 else (*result, "")
            trader_data = new_td or ""
        except Exception as e:
            orders_dict = {}

        this_tick = defaultdict(list)
        for product, orders in (orders_dict or {}).items():
            if product not in order_depths: continue
            match_orders(orders, order_depths[product], mkt_by_sym[product],
                         product, positions, realized, ts)
        prev_own = this_tick

    pnl_by_prod = {}
    for p in set(list(realized) + list(last_mid)):
        pnl_by_prod[p] = realized.get(p, 0.0) + positions.get(p, 0) * last_mid.get(p, 0.0)
    return pnl_by_prod


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_1000tick.py trader1.py trader2.py")
        sys.exit(1)

    t1, t2 = sys.argv[1], sys.argv[2]
    prices = load_prices(DAYS, max_ts=MAX_TS)
    mkt = load_market_trades(DAYS, max_ts=MAX_TS)

    print(f"\nFirst-1000-tick comparison: {os.path.basename(t1)} vs {os.path.basename(t2)}")

    all_products = set()
    results = {}

    for trader_path, label in [(t1, "v_old"), (t2, "v_new")]:
        day_totals = []
        prod_results = defaultdict(dict)
        for day in DAYS:
            pnl = run_trader_on_day(trader_path, prices, mkt, day)
            total = sum(pnl.values())
            day_totals.append(total)
            for p, v in pnl.items():
                prod_results[p][day] = v
                all_products.add(p)
        results[label] = {"totals": day_totals, "by_prod": prod_results}
        print(f"\n{label} ({os.path.basename(trader_path)}):")
        print(f"  D2={day_totals[0]:+,.0f}  D3={day_totals[1]:+,.0f}  D4={day_totals[2]:+,.0f}  avg={(sum(day_totals)/3):+,.0f}")

    # Per-product diff
    old_b = results["v_old"]["by_prod"]
    new_b = results["v_new"]["by_prod"]

    print(f"\n{'Product':<42} {'D2 old':>8} {'D2 new':>8} {'D3 old':>8} {'D3 new':>8} {'D4 old':>8} {'D4 new':>8}")
    print("-" * 105)
    # Show products where there's a difference
    changed = []
    for p in sorted(all_products):
        old_d2 = old_b.get(p, {}).get(2, 0)
        new_d2 = new_b.get(p, {}).get(2, 0)
        old_d3 = old_b.get(p, {}).get(3, 0)
        new_d3 = new_b.get(p, {}).get(3, 0)
        old_d4 = old_b.get(p, {}).get(4, 0)
        new_d4 = new_b.get(p, {}).get(4, 0)
        if abs(new_d4 - old_d4) > 10 or abs(new_d3 - old_d3) > 10 or abs(new_d2 - old_d2) > 10:
            changed.append((p, old_d2, new_d2, old_d3, new_d3, old_d4, new_d4))

    changed.sort(key=lambda x: -(x[6] - x[5]))
    for row in changed:
        p, od2, nd2, od3, nd3, od4, nd4 = row
        d4_diff = nd4 - od4
        marker = " +" if d4_diff > 100 else (" -" if d4_diff < -100 else "  ")
        print(f"{p:<42} {od2:>8,.0f} {nd2:>8,.0f} {od3:>8,.0f} {nd3:>8,.0f} {od4:>8,.0f} {nd4:>8,.0f}{marker}")

    old_d4 = sum(results["v_old"]["by_prod"].get(p, {}).get(4, 0) for p in all_products)
    new_d4 = sum(results["v_new"]["by_prod"].get(p, {}).get(4, 0) for p in all_products)
    print(f"\nD4 total: {os.path.basename(t1)}={old_d4:+,.0f}  {os.path.basename(t2)}={new_d4:+,.0f}  delta={new_d4-old_d4:+,.0f}")


if __name__ == "__main__":
    main()
