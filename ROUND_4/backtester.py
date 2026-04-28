"""
backtester.py — Prosperity Round 4 simulator
=============================================
Reads price + trade CSVs, runs a Trader class tick-by-tick,
matches orders against the order book, and reports per-product P&L.

Usage
-----
  # Baseline (no overrides):
  python backtester.py --trader v23.py --data .

  # VEV_5500 SELL_MIN sweep (called by SLURM array):
  python backtester.py --trader v23.py --data . --sell_min 6

  # Override HYDROGEL SELL extreme threshold:
  python backtester.py --trader v23.py --data . --hyd_sell 10030
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ── datamodel must be importable (same directory) ────────────────────────────
from datamodel import (
    Listing, Observation, Order, OrderDepth,
    Trade, TradingState,
)

# ── Position limits (hard) ───────────────────────────────────────────────────
LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK":       200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300, "VEV_4500": 300,
    "VEV_5000": 300, "VEV_5100": 300, "VEV_5200": 300,
    "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
    "VEV_6000": 300, "VEV_6500": 300,
}

DAYS = [1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_prices(data_dir: str) -> Dict[Tuple[int, int], Dict[str, dict]]:
    """Return {(day, timestamp): {product: row_dict}}"""
    prices: Dict[Tuple[int, int], Dict[str, dict]] = {}
    for day in DAYS:
        path = os.path.join(data_dir, f"prices_round_4_day_{day}.csv")
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                key = (day, int(row["timestamp"]))
                if key not in prices:
                    prices[key] = {}
                prices[key][row["product"]] = row
    return prices


def load_market_trades(data_dir: str) -> Dict[Tuple[int, int], List[Trade]]:
    """Return {(day, timestamp): [Trade, ...]}"""
    result: Dict[Tuple[int, int], List[Trade]] = defaultdict(list)
    for day in DAYS:
        path = os.path.join(data_dir, f"trades_round_4_day_{day}.csv")
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                key = (day, int(row["timestamp"]))
                result[key].append(Trade(
                    symbol    = row["symbol"],
                    price     = int(float(row["price"])),
                    quantity  = int(row["quantity"]),
                    buyer     = row["buyer"]  or None,
                    seller    = row["seller"] or None,
                    timestamp = int(row["timestamp"]),
                ))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Order book builder
# ─────────────────────────────────────────────────────────────────────────────

def build_order_depth(row: dict) -> OrderDepth:
    od = OrderDepth()
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}", "").strip()
        bv = row.get(f"bid_volume_{i}", "").strip()
        ap = row.get(f"ask_price_{i}", "").strip()
        av = row.get(f"ask_volume_{i}", "").strip()
        if bp and bv:
            od.buy_orders[int(float(bp))] = int(float(bv))
        if ap and av:
            od.sell_orders[int(float(ap))] = -int(float(av))   # negative!
    return od


# ─────────────────────────────────────────────────────────────────────────────
# Order matching
# ─────────────────────────────────────────────────────────────────────────────

def match_orders(
    orders:    List[Order],
    od:        OrderDepth,
    product:   str,
    positions: Dict[str, int],
    realized:  Dict[str, float],
    timestamp: int,
) -> List[Trade]:
    """Fill orders against the snapshot order book. Mutates positions + realized."""
    fills: List[Trade] = []
    limit = LIMITS.get(product, 200)
    pos   = positions.get(product, 0)

    for order in orders:
        if order.quantity > 0:                          # BUY
            max_buy  = max(0, limit - pos)
            remaining = min(order.quantity, max_buy)
            for ask_px in sorted(od.sell_orders):
                if ask_px > order.price or remaining <= 0:
                    break
                avail = -od.sell_orders[ask_px]
                fill  = min(remaining, avail)
                if fill > 0:
                    od.sell_orders[ask_px] += fill      # reduce available
                    pos       += fill
                    realized[product] -= fill * ask_px
                    remaining -= fill
                    fills.append(Trade(product, ask_px, fill,
                                       "SUBMISSION", None, timestamp))

        elif order.quantity < 0:                        # SELL
            qty      = -order.quantity
            max_sell = max(0, limit + pos)
            remaining = min(qty, max_sell)
            for bid_px in sorted(od.buy_orders, reverse=True):
                if bid_px < order.price or remaining <= 0:
                    break
                avail = od.buy_orders[bid_px]
                fill  = min(remaining, avail)
                if fill > 0:
                    od.buy_orders[bid_px] -= fill
                    pos       -= fill
                    realized[product] += fill * bid_px
                    remaining -= fill
                    fills.append(Trade(product, bid_px, -fill,
                                       None, "SUBMISSION", timestamp))

    positions[product] = pos
    return fills


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    trader,
    prices:         Dict[Tuple[int, int], Dict[str, dict]],
    market_trades:  Dict[Tuple[int, int], List[Trade]],
) -> Tuple[Dict[str, float], Dict[str, int]]:

    positions: Dict[str, int]   = defaultdict(int)
    realized:  Dict[str, float] = defaultdict(float)
    trader_data = ""
    prev_own_trades: Dict[str, List[Trade]] = defaultdict(list)

    all_ticks = sorted(prices.keys())   # sorted by (day, ts)

    for (day, ts) in all_ticks:
        tick_prices = prices[(day, ts)]
        tick_mktrades = market_trades.get((day, ts), [])

        # Build order depths
        order_depths = {
            product: build_order_depth(row)
            for product, row in tick_prices.items()
        }

        # Group market trades by symbol
        mkt_by_sym: Dict[str, List[Trade]] = defaultdict(list)
        for t in tick_mktrades:
            mkt_by_sym[t.symbol].append(t)

        # Build listings
        listings = {
            p: Listing(p, p, "XIRECS") for p in tick_prices
        }

        state = TradingState(
            timestamp     = ts,
            traderData    = trader_data,
            listings      = listings,
            order_depths  = order_depths,
            own_trades    = dict(prev_own_trades),
            market_trades = dict(mkt_by_sym),
            position      = dict(positions),
            observations  = Observation(),
        )

        # ── Call trader ───────────────────────────────────────────────────
        try:
            result = trader.run(state)
            if isinstance(result, tuple) and len(result) == 3:
                orders_dict, _conversions, new_td = result
            else:
                orders_dict, new_td = result, ""
            trader_data = new_td or ""
        except Exception as exc:
            print(f"[WARN] Trader error at day={day} ts={ts}: {exc}",
                  file=sys.stderr)
            orders_dict = {}

        # ── Match orders ─────────────────────────────────────────────────
        this_tick_own: Dict[str, List[Trade]] = defaultdict(list)
        for product, orders in (orders_dict or {}).items():
            if product not in order_depths or not orders:
                continue
            od    = order_depths[product]
            fills = match_orders(orders, od, product, positions, realized, ts)
            this_tick_own[product].extend(fills)

        prev_own_trades = this_tick_own

    # ── Final P&L (realized + unrealized at last mid) ────────────────────
    last_mid: Dict[str, float] = {}
    for (_d, _t), tick_prices in sorted(prices.items()):
        for product, row in tick_prices.items():
            mp = row.get("mid_price", "").strip()
            if mp:
                last_mid[product] = float(mp)

    final_pnl: Dict[str, float] = {}
    for product in set(list(realized) + list(last_mid)):
        unrealized      = positions.get(product, 0) * last_mid.get(product, 0.0)
        final_pnl[product] = realized.get(product, 0.0) + unrealized

    return final_pnl, dict(positions)


# ─────────────────────────────────────────────────────────────────────────────
# Trader loader
# ─────────────────────────────────────────────────────────────────────────────

def load_trader(trader_path: str):
    spec   = importlib.util.spec_from_file_location("trader_module", trader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Trader()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Prosperity R4 backtester")
    p.add_argument("--trader",   required=True,  help="Path to trader .py file")
    p.add_argument("--data",     required=True,  help="Directory with price/trade CSVs")
    p.add_argument("--output",   default=None,   help="Write JSON results to this file")
    # Optional overrides (applied after loading the trader)
    p.add_argument("--sell_min", type=int, default=None,
                   help="Override VEV_5500_SELL_MIN (sweep: 5,6,7,8)")
    p.add_argument("--hyd_sell", type=int, default=None,
                   help="Override HYDROGEL_PACK EXTREME_SELL threshold")
    p.add_argument("--vf_buy",   type=int, default=None,
                   help="Override VELVETFRUIT_EXTRACT EXTREME_BUY threshold")
    p.add_argument("--mark01_5300_floor", type=int, default=None,
                   help="Override MARK01_MIN_BID[VEV_5300]")
    p.add_argument("--mark01_5400_floor", type=int, default=None,
                   help="Override MARK01_MIN_BID[VEV_5400]")
    p.add_argument("--vev5500_sell_min", type=int, default=None,
                   help="Override VEV_5500_SELL_MIN (alias for --sell_min)")
    p.add_argument("--vev5500_buy_max", type=int, default=None,
                   help="Override VEV_5500_BUY_MAX")
    return p.parse_args()


def apply_overrides(trader, args):
    if args.sell_min is not None:
        trader.VEV_5500_SELL_MIN = args.sell_min
        print(f"[override] VEV_5500_SELL_MIN = {args.sell_min}")
    if args.vev5500_sell_min is not None:
        trader.VEV_5500_SELL_MIN = args.vev5500_sell_min
        print(f"[override] VEV_5500_SELL_MIN = {args.vev5500_sell_min}")
    if args.vev5500_buy_max is not None:
        trader.VEV_5500_BUY_MAX = args.vev5500_buy_max
        print(f"[override] VEV_5500_BUY_MAX = {args.vev5500_buy_max}")
    if args.hyd_sell is not None:
        trader.EXTREME_SELL["HYDROGEL_PACK"] = args.hyd_sell
        print(f"[override] EXTREME_SELL[HYDROGEL_PACK] = {args.hyd_sell}")
    if args.vf_buy is not None:
        trader.EXTREME_BUY["VELVETFRUIT_EXTRACT"] = args.vf_buy
        print(f"[override] EXTREME_BUY[VELVETFRUIT] = {args.vf_buy}")
    if args.mark01_5300_floor is not None:
        trader.MARK01_MIN_BID["VEV_5300"] = args.mark01_5300_floor
        print(f"[override] MARK01_MIN_BID[VEV_5300] = {args.mark01_5300_floor}")
    if args.mark01_5400_floor is not None:
        trader.MARK01_MIN_BID["VEV_5400"] = args.mark01_5400_floor
        print(f"[override] MARK01_MIN_BID[VEV_5400] = {args.mark01_5400_floor}")


def print_results(pnl: Dict[str, float], positions: Dict[str, int], label: str = ""):
    print(f"\n{'='*55}")
    if label:
        print(f"  {label}")
    print(f"{'='*55}")
    print(f"  {'Product':<28} {'PnL':>12}  {'Pos':>6}")
    print(f"  {'-'*48}")
    total = 0.0
    for product in sorted(pnl):
        v   = pnl[product]
        pos = positions.get(product, 0)
        total += v
        print(f"  {product:<28} {v:>12,.1f}  {pos:>6}")
    print(f"  {'-'*48}")
    print(f"  {'TOTAL':<28} {total:>12,.1f}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()

    # Build label for output
    label_parts = [os.path.basename(args.trader)]
    if args.sell_min is not None:
        label_parts.append(f"SELL_MIN={args.sell_min}")
    if args.hyd_sell is not None:
        label_parts.append(f"HYD_SELL={args.hyd_sell}")
    label = " | ".join(label_parts)

    print(f"Loading trader:  {args.trader}")
    trader = load_trader(args.trader)
    apply_overrides(trader, args)

    print(f"Loading data:    {args.data}")
    prices        = load_prices(args.data)
    mkt_trades    = load_market_trades(args.data)
    print(f"  {len(prices):,} ticks loaded")

    print("Running simulation …")
    pnl, positions = run_simulation(trader, prices, mkt_trades)

    print_results(pnl, positions, label)

    if args.output:
        out = {
            "label":     label,
            "pnl":       pnl,
            "positions": positions,
            "total":     sum(pnl.values()),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
