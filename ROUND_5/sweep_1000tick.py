"""
sweep_1000tick.py — Sweep all products for first-1000-tick PnL across D2/D3/D4.

This is the correct metric for predicting online performance because online = first
1000 ticks of day 4. We test each product in isolation with multiple alpha values.
Products consistently positive across D2/D3/D4 first 1000 ticks are genuine signals.
"""
import csv, importlib.util, json, sys, os
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from datamodel import Listing, Observation, Trade, OrderDepth, TradingState

LIMIT = 10
DATA_DIR = "."
DAYS = [2, 3, 4]
MAX_TS = 99900  # first 1000 ticks: ts=0..99900


def load_prices(days):
    prices = {}
    for day in days:
        path = os.path.join(DATA_DIR, f"prices_round_5_day_{day}.csv")
        with open(path) as f:
            for row in csv.DictReader(f, delimiter=";"):
                ts = int(row["timestamp"])
                if ts > MAX_TS:
                    continue
                key = (day, ts)
                prices.setdefault(key, {})[row["product"]] = row
    return prices


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


def match_orders(orders, od, product, positions, realized, ts):
    pos = positions.get(product, 0)
    total_qty = sum(o.quantity for o in orders)
    if pos + total_qty > LIMIT or pos + total_qty < -LIMIT:
        return
    for order in orders:
        if order.quantity > 0:
            remaining = order.quantity
            for ask_px in sorted(od.sell_orders):
                if ask_px > order.price or remaining <= 0: break
                avail = -od.sell_orders[ask_px]
                fill = min(remaining, avail)
                if fill > 0:
                    od.sell_orders[ask_px] += fill
                    pos += fill
                    realized[product] -= fill * ask_px
                    remaining -= fill
        elif order.quantity < 0:
            remaining = -order.quantity
            for bid_px in sorted(od.buy_orders, reverse=True):
                if bid_px < order.price or remaining <= 0: break
                avail = od.buy_orders[bid_px]
                fill = min(remaining, avail)
                if fill > 0:
                    od.buy_orders[bid_px] -= fill
                    pos -= fill
                    realized[product] += fill * bid_px
                    remaining -= fill
    positions[product] = pos


def run_single_product(product, slow_alpha, prices, day):
    """Run simulation for a single product on one day, first 1000 ticks only."""
    FAST_ALPHA = 0.02
    WARMUP_TICKS = 3000
    SIGNAL_STRONG = 55
    SIGNAL_WEAK = 20
    DRAWDOWN_PTS = 600
    COOLDOWN_TICKS = 2000
    RAMP_PER_TICK = 5
    PASSIVE_SIZE = 2

    positions = defaultdict(int)
    realized = defaultdict(float)
    fe = None; se = None
    cb_until = 0; cb_peak = None; cb_dir = 0

    last_mid = None

    for (d, ts) in sorted(prices.keys()):
        if d != day:
            continue
        tick_prices = prices[(d, ts)]
        if product not in tick_prices:
            continue

        row = tick_prices[product]
        od = build_od(row)

        mid_raw = row.get("mid_price", "")
        if not mid_raw:
            continue
        mid = float(mid_raw)
        last_mid = mid

        # EMA update
        if fe is None:
            fe = mid; se = mid
        else:
            fe = FAST_ALPHA * mid + (1 - FAST_ALPHA) * fe
            se = slow_alpha * mid + (1 - slow_alpha) * se

        # EMA target
        if ts < WARMUP_TICKS:
            raw_target = 0
        else:
            signal = fe - se
            if signal > SIGNAL_STRONG: raw_target = LIMIT
            elif signal > SIGNAL_WEAK: raw_target = 5
            elif signal < -SIGNAL_STRONG: raw_target = -LIMIT
            elif signal < -SIGNAL_WEAK: raw_target = -5
            else: raw_target = 0

        # Circuit breaker
        if ts < cb_until:
            target = 0
        else:
            if raw_target != cb_dir:
                cb_peak = mid
            cb_dir = raw_target
            target = raw_target

            if raw_target > 0:
                if mid > (cb_peak or mid): cb_peak = mid
                if cb_peak - mid > DRAWDOWN_PTS:
                    cb_until = ts + COOLDOWN_TICKS; cb_peak = mid; cb_dir = 0; target = 0
            elif raw_target < 0:
                if mid < (cb_peak or mid): cb_peak = mid
                if mid - cb_peak > DRAWDOWN_PTS:
                    cb_until = ts + COOLDOWN_TICKS; cb_peak = mid; cb_dir = 0; target = 0

        # Generate orders
        pos = positions[product]
        orders = []
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        buy_cap = LIMIT - pos; sell_cap = LIMIT + pos

        if pos < target:
            need = target - pos
            if best_ask is not None and buy_cap > 0:
                from datamodel import Order
                vol = min(-od.sell_orders[best_ask], buy_cap, need, RAMP_PER_TICK)
                if vol > 0: orders.append(Order(product, best_ask, vol))
            if best_bid is not None and buy_cap > 0 and need > 0:
                orders.append(Order(product, best_bid + 1, min(buy_cap, PASSIVE_SIZE, need)))
        elif pos > target:
            need = pos - target
            if best_bid is not None and sell_cap > 0:
                from datamodel import Order
                vol = min(od.buy_orders[best_bid], sell_cap, need, RAMP_PER_TICK)
                if vol > 0: orders.append(Order(product, best_bid, -vol))
            if best_ask is not None and sell_cap > 0 and need > 0:
                orders.append(Order(product, best_ask - 1, -min(sell_cap, PASSIVE_SIZE, need)))

        if orders:
            match_orders(orders, od, product, positions, realized, ts)

    pos = positions[product]
    pnl = realized[product] + pos * (last_mid or 0)
    return pnl


def get_all_products(prices):
    prods = set()
    for tick in prices.values():
        prods.update(tick.keys())
    return sorted(prods)


def main():
    prices = load_prices(DAYS)
    all_prods = get_all_products(prices)

    CURRENT_WHITELIST = {
        "PEBBLES_XL",
        "OXYGEN_SHAKE_MORNING_BREATH",
        "ROBOT_MOPPING",
        "MICROCHIP_CIRCLE",
        "SLEEP_POD_COTTON",
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "MICROCHIP_OVAL",
        "ROBOT_IRONING",
        "PANEL_1X4",
        "UV_VISOR_AMBER",
        "TRANSLATOR_GRAPHITE_MIST",
    }

    alphas = [0.0006, 0.0008, 0.001, 0.002, 0.003, 0.005]

    # Test ALL products (including whitelist to find better alpha)
    print(f"\n{'Product':<42} {'alpha':>7} {'D2':>8} {'D3':>8} {'D4':>8} {'avg':>8}  note")
    print("-" * 95)

    results = []
    for prod in all_prods:
        if prod in ("PEBBLES_XL",):  # skip constraint arb
            continue

        best_avg = -999999
        best_a = None
        best_d2 = best_d3 = best_d4 = 0

        for a in alphas:
            d2 = run_single_product(prod, a, prices, 2)
            d3 = run_single_product(prod, a, prices, 3)
            d4 = run_single_product(prod, a, prices, 4)
            avg = (d2 + d3 + d4) / 3
            if avg > best_avg:
                best_avg = avg
                best_a = a
                best_d2, best_d3, best_d4 = d2, d3, d4

        note = ""
        if prod in CURRENT_WHITELIST:
            note = "IN_WL"
        elif best_d2 > 0 and best_d3 > 0 and best_d4 > 0:
            note = "ALL_POS"
        elif best_d4 > 500:
            note = "D4_pos"

        results.append((prod, best_a, best_d2, best_d3, best_d4, best_avg, note))

    # Sort by D4 (best predictor of online)
    results.sort(key=lambda x: -x[4])

    for prod, a, d2, d3, d4, avg, note in results:
        print(f"{prod:<42} {a:>7.4f} {d2:>8,.0f} {d3:>8,.0f} {d4:>8,.0f} {avg:>8,.0f}  {note}")

    print("\n--- Top candidates (D4 > 2000, not in whitelist) ---")
    for prod, a, d2, d3, d4, avg, note in results:
        if "IN_WL" not in note and d4 > 2000:
            print(f"  {prod:<42} α={a:.4f}  D4={d4:+,.0f}  D3={d3:+,.0f}  D2={d2:+,.0f}  avg={avg:+,.0f}")

    print("\n--- Already in whitelist: first-1000-tick with best alpha ---")
    for prod, a, d2, d3, d4, avg, note in results:
        if "IN_WL" in note:
            print(f"  {prod:<42} α={a:.4f}  D4={d4:+,.0f}  D3={d3:+,.0f}  D2={d2:+,.0f}  avg={avg:+,.0f}")


if __name__ == "__main__":
    main()
