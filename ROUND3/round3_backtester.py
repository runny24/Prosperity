"""
Round 3 local replay backtester.

This is intentionally a conservative research tool, not a perfect exchange
simulator. It fills only orders that cross visible historical liquidity:

- buy order fills against sell orders with price <= order.price
- sell order fills against buy orders with price >= order.price

Passive orders inside the spread are left unfilled. The IMC platform can fill
some passive orders with future bot flow, so platform PnL may be higher than
this backtest for market-making strategies. This tool is mainly for fast
strategy comparisons, risk checks, and parameter search.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DATA_DIR = Path("ROUND3_Data")
DEFAULT_LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
}


class Order:
    def __init__(self, symbol: str, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = int(price)
        self.quantity = int(quantity)

    def __repr__(self) -> str:
        return f"({self.symbol}, {self.price}, {self.quantity})"


class OrderDepth:
    def __init__(self) -> None:
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class TradingState:
    def __init__(
        self,
        traderData: str,
        timestamp: int,
        listings: Dict,
        order_depths: Dict[str, OrderDepth],
        own_trades: Dict,
        market_trades: Dict,
        position: Dict[str, int],
        observations,
    ) -> None:
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations


@dataclass
class Observation:
    plainValueObservations: Dict = field(default_factory=dict)
    conversionObservations: Dict = field(default_factory=dict)


@dataclass
class Fill:
    timestamp: int
    product: str
    price: int
    quantity: int


def install_datamodel_stub() -> None:
    module = types.ModuleType("datamodel")
    module.Order = Order
    module.OrderDepth = OrderDepth
    module.TradingState = TradingState
    module.Observation = Observation
    module.UserId = str
    module.Symbol = str
    module.Product = str
    module.Position = int
    module.Listing = object
    module.Trade = object
    sys.modules["datamodel"] = module


def load_trader(path: Path):
    install_datamodel_stub()
    module_name = "loaded_trader_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import trader from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.Trader()


def load_prices(days: Iterable[int]) -> pd.DataFrame:
    frames = []
    for day in days:
        path = DATA_DIR / f"prices_round_3_day_{day}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, sep=";"))
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["day", "timestamp", "product"]).reset_index(drop=True)
    return prices


def order_depth_from_row(row: pd.Series) -> OrderDepth:
    od = OrderDepth()
    for i in (1, 2, 3):
        bid_price = row.get(f"bid_price_{i}")
        bid_volume = row.get(f"bid_volume_{i}")
        if pd.notna(bid_price) and pd.notna(bid_volume) and int(bid_volume) != 0:
            od.buy_orders[int(bid_price)] = int(bid_volume)

        ask_price = row.get(f"ask_price_{i}")
        ask_volume = row.get(f"ask_volume_{i}")
        if pd.notna(ask_price) and pd.notna(ask_volume) and int(ask_volume) != 0:
            # Historical CSV stores ask volume as positive; datamodel uses negative.
            od.sell_orders[int(ask_price)] = -abs(int(ask_volume))
    return od


def make_state(
    day: int,
    timestamp: int,
    group: pd.DataFrame,
    position: Dict[str, int],
    trader_data: str,
) -> TradingState:
    depths = {}
    for _, row in group.iterrows():
        depths[str(row["product"])] = order_depth_from_row(row)
    return TradingState(
        traderData=trader_data,
        timestamp=int(timestamp),
        listings={},
        order_depths=depths,
        own_trades={},
        market_trades={},
        position=dict(position),
        observations=Observation(),
    )


def fill_orders(
    timestamp: int,
    order_depths: Dict[str, OrderDepth],
    orders_by_product: Dict[str, List[Order]],
    position: Dict[str, int],
    cash: Dict[str, float],
    limits: Dict[str, int],
) -> List[Fill]:
    fills: List[Fill] = []
    for product, orders in orders_by_product.items():
        if product not in order_depths:
            continue
        od = order_depths[product]
        buy_book = dict(od.buy_orders)
        sell_book = dict(od.sell_orders)

        for order in orders:
            if order.quantity == 0:
                continue
            if order.quantity > 0:
                remaining = order.quantity
                for ask_price in sorted(sell_book):
                    if ask_price > order.price or remaining <= 0:
                        break
                    available = -sell_book[ask_price]
                    capacity = limits[product] - position.get(product, 0)
                    qty = min(remaining, available, capacity)
                    if qty <= 0:
                        break
                    position[product] = position.get(product, 0) + qty
                    cash[product] = cash.get(product, 0.0) - qty * ask_price
                    sell_book[ask_price] += qty
                    remaining -= qty
                    fills.append(Fill(timestamp, product, ask_price, qty))
            else:
                remaining = -order.quantity
                for bid_price in sorted(buy_book, reverse=True):
                    if bid_price < order.price or remaining <= 0:
                        break
                    available = buy_book[bid_price]
                    capacity = limits[product] + position.get(product, 0)
                    qty = min(remaining, available, capacity)
                    if qty <= 0:
                        break
                    position[product] = position.get(product, 0) - qty
                    cash[product] = cash.get(product, 0.0) + qty * bid_price
                    buy_book[bid_price] -= qty
                    remaining -= qty
                    fills.append(Fill(timestamp, product, bid_price, -qty))
    return fills


def mark_to_market(
    group: pd.DataFrame,
    position: Dict[str, int],
    cash: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    mids = {str(row["product"]): float(row["mid_price"]) for _, row in group.iterrows()}
    pnl_by_product = {}
    total = 0.0
    for product in sorted(set(cash) | set(position) | set(mids)):
        value = cash.get(product, 0.0) + position.get(product, 0) * mids.get(product, 0.0)
        pnl_by_product[product] = value
        total += value
    return total, pnl_by_product


def run_backtest(
    trader_path: Path,
    days: List[int],
    max_ticks: Optional[int] = None,
) -> Dict:
    trader = load_trader(trader_path)
    prices = load_prices(days)
    grouped = prices.groupby(["day", "timestamp"], sort=True)

    position: Dict[str, int] = {}
    cash: Dict[str, float] = {}
    trader_data = ""
    equity_curve = []
    all_fills: List[Fill] = []
    last_pnl_by_product: Dict[str, float] = {}

    for tick_index, ((day, timestamp), group) in enumerate(grouped):
        if max_ticks is not None and tick_index >= max_ticks:
            break
        state = make_state(int(day), int(timestamp), group, position, trader_data)
        output = trader.run(state)
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError("Trader.run must return (orders, conversions, traderData)")
        orders_by_product, _conversions, trader_data = output
        if orders_by_product is None:
            orders_by_product = {}

        fills = fill_orders(
            int(timestamp),
            state.order_depths,
            orders_by_product,
            position,
            cash,
            DEFAULT_LIMITS,
        )
        all_fills.extend(fills)
        total, last_pnl_by_product = mark_to_market(group, position, cash)
        equity_curve.append(
            {
                "day": int(day),
                "timestamp": int(timestamp),
                "equity": total,
            }
        )

    final_equity = equity_curve[-1]["equity"] if equity_curve else 0.0
    max_equity = max((x["equity"] for x in equity_curve), default=0.0)
    min_equity = min((x["equity"] for x in equity_curve), default=0.0)
    max_drawdown = 0.0
    peak = -10**18
    for point in equity_curve:
        peak = max(peak, point["equity"])
        max_drawdown = min(max_drawdown, point["equity"] - peak)

    fills_by_product: Dict[str, Dict[str, int]] = {}
    for fill in all_fills:
        stats = fills_by_product.setdefault(fill.product, {"fills": 0, "buy_qty": 0, "sell_qty": 0})
        stats["fills"] += 1
        if fill.quantity > 0:
            stats["buy_qty"] += fill.quantity
        else:
            stats["sell_qty"] += -fill.quantity

    return {
        "trader": str(trader_path),
        "days": days,
        "ticks": len(equity_curve),
        "final_equity": final_equity,
        "max_equity": max_equity,
        "min_equity": min_equity,
        "max_drawdown": max_drawdown,
        "position": dict(sorted(position.items())),
        "pnl_by_product": dict(sorted(last_pnl_by_product.items())),
        "fills_by_product": dict(sorted(fills_by_product.items())),
    }


def print_summary(result: Dict) -> None:
    print(f"Trader: {result['trader']}")
    print(f"Days: {result['days']}  ticks: {result['ticks']}")
    print(f"Final equity: {result['final_equity']:.2f}")
    print(f"Min equity:   {result['min_equity']:.2f}")
    print(f"Max equity:   {result['max_equity']:.2f}")
    print(f"Max drawdown: {result['max_drawdown']:.2f}")
    print("\nPositions:")
    for product, qty in result["position"].items():
        print(f"  {product:22s} {qty:6d}")
    print("\nPnL by product:")
    for product, pnl in result["pnl_by_product"].items():
        fills = result["fills_by_product"].get(product, {})
        print(
            f"  {product:22s} {pnl:10.2f}"
            f"  fills={fills.get('fills', 0):4d}"
            f" buy={fills.get('buy_qty', 0):5d}"
            f" sell={fills.get('sell_qty', 0):5d}"
        )


def parse_days(raw: str) -> List[int]:
    if raw.lower() == "all":
        return [0, 1, 2]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trader", type=Path, help="Path to trader .py file")
    parser.add_argument("--days", default="2", help="Comma-separated days or 'all'. Default: 2")
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    result = run_backtest(args.trader, parse_days(args.days), args.max_ticks)
    print_summary(result)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

