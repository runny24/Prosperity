"""
Replay backtest for r5_v1.py using ROUND5_data CSV snapshots.

This is intentionally conservative by default:
- orders are sent to the trader at each timestamp
- only orders crossing the visible book are filled
- resting/passive orders are not assumed to fill
- remaining inventory is marked to the latest mid price

Usage:
    python3 test_backtest.py
    python3 test_backtest.py r5_v1.py
    python3 test_backtest.py r5_v1.py --days 2 3 4
    python3 test_backtest.py --strategy r5_v1.py --days 4 --max-ticks 1000
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import types
from functools import lru_cache
from itertools import islice
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DATA_DIR = Path("ROUND5_data")
DEFAULT_STRATEGY = Path("r5_v1.py")


class ProsperityEncoder(json.JSONEncoder):
    def default(self, obj):
        return obj.__dict__ if hasattr(obj, "__dict__") else str(obj)


class Order:
    def __init__(self, symbol: str, price: int, quantity: int):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity


class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Trade:
    def __init__(
        self,
        symbol: str,
        price: float,
        quantity: int,
        buyer: str = "",
        seller: str = "",
        timestamp: int = 0,
    ):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp


class Listing:
    def __init__(self, symbol: str, product: str, denomination: str = "XIRECS"):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination


class Observation:
    def __init__(self):
        self.plainValueObservations = {}
        self.conversionObservations = {}


class TradingState:
    def __init__(
        self,
        timestamp: int,
        traderData: str,
        listings: Dict[str, Listing],
        order_depths: Dict[str, OrderDepth],
        own_trades: Dict[str, List[Trade]],
        market_trades: Dict[str, List[Trade]],
        position: Dict[str, int],
        observations: Observation,
    ):
        self.timestamp = timestamp
        self.traderData = traderData
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations


Symbol = str


@dataclass
class Fill:
    day: int
    timestamp: int
    product: str
    price: float
    quantity: int


def install_datamodel_stub() -> None:
    module = types.ModuleType("datamodel")
    for name, obj in {
        "Order": Order,
        "OrderDepth": OrderDepth,
        "TradingState": TradingState,
        "ProsperityEncoder": ProsperityEncoder,
        "Listing": Listing,
        "Observation": Observation,
        "Symbol": Symbol,
        "Trade": Trade,
    }.items():
        setattr(module, name, obj)
    sys.modules["datamodel"] = module


def import_strategy(path: Path, overrides: Optional[Dict[str, object]] = None):
    install_datamodel_stub()
    module_name = f"strategy_{path.stem}_{abs(hash((str(path), str(overrides))))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import strategy from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if overrides:
        for key, value in overrides.items():
            setattr(module.Trader, key, value)
    return module


def price_file_for_day(day: int) -> Path:
    return DATA_DIR / f"prices_round_5_day_{day}.csv"


def trade_file_for_day(day: int) -> Path:
    return DATA_DIR / f"trades_round_5_day_{day}.csv"


@lru_cache(maxsize=None)
def _load_prices_cached(days: Tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for day in days:
        path = price_file_for_day(day)
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, sep=";"))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["day", "timestamp", "product"]).reset_index(drop=True)


def load_prices(days: Iterable[int]) -> pd.DataFrame:
    return _load_prices_cached(tuple(days)).copy()


@lru_cache(maxsize=None)
def _load_trades_cached(days: Tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for day in days:
        path = trade_file_for_day(day)
        if not path.exists():
            continue
        frame = pd.read_csv(path, sep=";")
        frame["day"] = day
        frame = frame.rename(columns={"symbol": "product"})
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["day", "timestamp", "product", "price", "quantity"])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["day", "timestamp", "product"]).reset_index(drop=True)


def load_trades(days: Iterable[int]) -> pd.DataFrame:
    return _load_trades_cached(tuple(days)).copy()


def row_to_depth(row) -> OrderDepth:
    depth = OrderDepth()
    for level in (1, 2, 3):
        bid_price = getattr(row, f"bid_price_{level}")
        bid_volume = getattr(row, f"bid_volume_{level}")
        ask_price = getattr(row, f"ask_price_{level}")
        ask_volume = getattr(row, f"ask_volume_{level}")
        if pd.notna(bid_price) and pd.notna(bid_volume) and int(bid_volume) > 0:
            depth.buy_orders[int(bid_price)] = int(bid_volume)
        if pd.notna(ask_price) and pd.notna(ask_volume) and int(ask_volume) > 0:
            depth.sell_orders[int(ask_price)] = -int(ask_volume)
    return depth


def market_trades_for_group(group: pd.DataFrame, timestamp: int) -> Dict[str, List[Trade]]:
    trades: Dict[str, List[Trade]] = {}
    if group.empty:
        return trades
    for row in group.itertuples(index=False):
        trade = Trade(
            symbol=row.product,
            price=float(row.price),
            quantity=int(row.quantity),
            buyer=getattr(row, "buyer", "") or "",
            seller=getattr(row, "seller", "") or "",
            timestamp=timestamp,
        )
        trades.setdefault(row.product, []).append(trade)
    return trades


def match_order(
    order: Order,
    depth: OrderDepth,
    day: int,
    timestamp: int,
) -> List[Fill]:
    fills: List[Fill] = []
    remaining = int(order.quantity)

    if remaining > 0:
        for ask_price in sorted(depth.sell_orders):
            if remaining <= 0 or order.price < ask_price:
                break
            available = -depth.sell_orders[ask_price]
            qty = min(remaining, available)
            if qty <= 0:
                continue
            fills.append(Fill(day, timestamp, order.symbol, float(ask_price), qty))
            remaining -= qty
            depth.sell_orders[ask_price] += qty

    elif remaining < 0:
        sell_qty = -remaining
        for bid_price in sorted(depth.buy_orders, reverse=True):
            if sell_qty <= 0 or order.price > bid_price:
                break
            available = depth.buy_orders[bid_price]
            qty = min(sell_qty, available)
            if qty <= 0:
                continue
            fills.append(Fill(day, timestamp, order.symbol, float(bid_price), -qty))
            sell_qty -= qty
            depth.buy_orders[bid_price] -= qty

    return fills


def mark_to_market(
    cash: Dict[str, float],
    position: Dict[str, int],
    last_mid: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    per_product = {}
    for product in sorted(set(cash) | set(position) | set(last_mid)):
        per_product[product] = cash.get(product, 0.0) + position.get(product, 0) * last_mid.get(product, 0.0)
    return sum(per_product.values()), per_product


def run_backtest(
    strategy_path: Path = DEFAULT_STRATEGY,
    days: Iterable[int] = (4,),
    max_ticks: Optional[int] = None,
    overrides: Optional[Dict[str, object]] = None,
    quiet: bool = True,
) -> Dict[str, object]:
    days = list(days)
    prices = load_prices(days)
    trades = load_trades(days)
    trade_groups = {key: group for key, group in trades.groupby(["day", "timestamp"])}

    strategy_module = import_strategy(strategy_path, overrides=overrides)
    trader = strategy_module.Trader()

    products = sorted(prices["product"].unique())
    listings = {p: Listing(p, p, "XIRECS") for p in products}
    observations = Observation()

    trader_data = ""
    position = {p: 0 for p in products}
    cash = {p: 0.0 for p in products}
    last_mid: Dict[str, float] = {}
    all_fills: List[Fill] = []
    pnl_path = []
    own_trades_next: Dict[str, List[Trade]] = {}

    if max_ticks is None:
        grouped_iter = prices.groupby(["day", "timestamp"], sort=True)
    else:
        per_day_iters = []
        for day in days:
            day_prices = prices[prices["day"].eq(day)]
            per_day_iters.extend(islice(day_prices.groupby(["day", "timestamp"], sort=True), max_ticks))
        grouped_iter = iter(per_day_iters)
    tick_count = 0

    for (day, timestamp), group in grouped_iter:
        tick_count += 1
        order_depths = {row.product: row_to_depth(row) for row in group.itertuples(index=False)}
        for row in group.itertuples(index=False):
            last_mid[row.product] = float(row.mid_price)

        state = TradingState(
            timestamp=int(timestamp),
            traderData=trader_data,
            listings=listings,
            order_depths=order_depths,
            own_trades=own_trades_next,
            market_trades=market_trades_for_group(trade_groups.get((day, timestamp), pd.DataFrame()), int(timestamp)),
            position={k: v for k, v in position.items() if v != 0},
            observations=observations,
        )

        with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
            orders, _, trader_data = trader.run(state)

        tick_fills: List[Fill] = []
        for product, product_orders in orders.items():
            if product not in order_depths:
                continue
            depth = order_depths[product]
            for order in product_orders:
                fills = match_order(order, depth, int(day), int(timestamp))
                tick_fills.extend(fills)
                for fill in fills:
                    position[fill.product] = position.get(fill.product, 0) + fill.quantity
                    cash[fill.product] = cash.get(fill.product, 0.0) - fill.price * fill.quantity

        own_trades_next = {}
        for fill in tick_fills:
            if fill.quantity > 0:
                trade = Trade(fill.product, fill.price, fill.quantity, "SUBMISSION", "", int(timestamp))
            else:
                trade = Trade(fill.product, fill.price, -fill.quantity, "", "SUBMISSION", int(timestamp))
            own_trades_next.setdefault(fill.product, []).append(trade)
        all_fills.extend(tick_fills)

        total_pnl, _ = mark_to_market(cash, position, last_mid)
        pnl_path.append({"day": int(day), "timestamp": int(timestamp), "pnl": total_pnl})

    total_pnl, per_product = mark_to_market(cash, position, last_mid)
    pnl_df = pd.DataFrame(pnl_path)
    if pnl_df.empty:
        max_drawdown = 0.0
    else:
        max_drawdown = float((pnl_df["pnl"] - pnl_df["pnl"].cummax()).min())

    fill_df = pd.DataFrame([fill.__dict__ for fill in all_fills])
    return {
        "strategy": str(strategy_path),
        "days": days,
        "ticks": tick_count,
        "total_pnl": float(total_pnl),
        "max_drawdown": max_drawdown,
        "per_product": per_product,
        "positions": {k: v for k, v in position.items() if v != 0},
        "fills": fill_df,
        "pnl_path": pnl_df,
    }


def print_summary(result: Dict[str, object]) -> None:
    print(f"strategy: {result['strategy']}")
    print(f"days: {result['days']} ticks: {result['ticks']}")
    print(f"total_pnl: {result['total_pnl']:.2f}")
    print(f"max_drawdown: {result['max_drawdown']:.2f}")
    print(f"fills: {len(result['fills'])}")

    per_product = pd.Series(result["per_product"]).sort_values(ascending=False)
    active = per_product[per_product.abs() > 1e-9]
    if not active.empty:
        print("\nper-product pnl:")
        print(active.to_string(float_format=lambda x: f"{x:,.2f}"))

    positions = result["positions"]
    if positions:
        print("\nending positions:")
        print(pd.Series(positions).sort_index().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy_file", nargs="?", default=None)
    parser.add_argument("--strategy", default=str(DEFAULT_STRATEGY))
    parser.add_argument("--days", type=int, nargs="+", default=[4])
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--verbose-trader-logs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy = args.strategy_file or args.strategy
    result = run_backtest(
        strategy_path=Path(strategy),
        days=args.days,
        max_ticks=args.max_ticks,
        quiet=not args.verbose_trader_logs,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
