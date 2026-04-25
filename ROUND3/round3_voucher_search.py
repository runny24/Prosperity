"""
Voucher parameter search for Round 3.

This script searches simple single-voucher rules:

fair = max(VELVETFRUIT_EXTRACT_mid - strike, 0) + time_value

Buy visible asks when ask < fair - edge.
Sell visible bids when bid > fair + edge.

It is not a final strategy and it does not simulate passive fill. The point is
to quickly identify which vouchers and rough parameter ranges have enough
historical edge to justify a proper trader implementation.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


DATA_DIR = Path("ROUND3_Data")
STRIKES = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}


@dataclass(frozen=True)
class Config:
    time_value: float
    edge: float
    cap: int
    max_take: int


@dataclass
class Result:
    product: str
    config: Config
    pnl: float
    min_equity: float
    max_drawdown: float
    final_position: int
    buy_qty: int
    sell_qty: int
    fills: int


def parse_days(raw: str) -> List[int]:
    if raw.lower() == "all":
        return [0, 1, 2]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_joined(days: Iterable[int]) -> pd.DataFrame:
    frames = []
    for day in days:
        frames.append(pd.read_csv(DATA_DIR / f"prices_round_3_day_{day}.csv", sep=";"))
    prices = pd.concat(frames, ignore_index=True)
    spot = prices.loc[
        prices["product"] == "VELVETFRUIT_EXTRACT",
        ["day", "timestamp", "mid_price"],
    ].rename(columns={"mid_price": "spot_mid"})
    joined = prices.merge(spot, on=["day", "timestamp"], how="left")
    return joined.sort_values(["day", "timestamp", "product"]).reset_index(drop=True)


def materialize_product_rows(joined: pd.DataFrame, product: str) -> List[Tuple]:
    cols = [
        "day",
        "timestamp",
        "spot_mid",
        "bid_price_1",
        "bid_volume_1",
        "bid_price_2",
        "bid_volume_2",
        "bid_price_3",
        "bid_volume_3",
        "ask_price_1",
        "ask_volume_1",
        "ask_price_2",
        "ask_volume_2",
        "ask_price_3",
        "ask_volume_3",
        "mid_price",
    ]
    df = joined.loc[joined["product"] == product, cols]
    return list(df.itertuples(index=False, name=None))


def is_valid_number(value) -> bool:
    return value == value


def simulate_product(rows: List[Tuple], product: str, config: Config) -> Result:
    strike = STRIKES[product]
    pos = 0
    cash = 0.0
    buy_qty = 0
    sell_qty = 0
    fills = 0
    min_equity = 0.0
    max_drawdown = 0.0
    peak = 0.0

    last_mid = 0.0
    for row in rows:
        (
            _day,
            _timestamp,
            spot_mid,
            bid_price_1,
            bid_volume_1,
            bid_price_2,
            bid_volume_2,
            bid_price_3,
            bid_volume_3,
            ask_price_1,
            ask_volume_1,
            ask_price_2,
            ask_volume_2,
            ask_price_3,
            ask_volume_3,
            mid_price,
        ) = row

        last_mid = float(mid_price)
        fair = max(float(spot_mid) - strike, 0.0) + config.time_value

        bought = 0
        asks = [
            (ask_price_1, ask_volume_1),
            (ask_price_2, ask_volume_2),
            (ask_price_3, ask_volume_3),
        ]
        for price, volume in asks:
            if not is_valid_number(price) or not is_valid_number(volume):
                continue
            price = int(price)
            volume = int(volume)
            if price >= fair - config.edge:
                continue
            qty = min(volume, config.cap - pos, config.max_take - bought)
            if qty <= 0:
                continue
            pos += qty
            cash -= qty * price
            buy_qty += qty
            bought += qty
            fills += 1

        sold = 0
        bids = [
            (bid_price_1, bid_volume_1),
            (bid_price_2, bid_volume_2),
            (bid_price_3, bid_volume_3),
        ]
        for price, volume in bids:
            if not is_valid_number(price) or not is_valid_number(volume):
                continue
            price = int(price)
            volume = int(volume)
            if price <= fair + config.edge:
                continue
            qty = min(volume, config.cap + pos, config.max_take - sold)
            if qty <= 0:
                continue
            pos -= qty
            cash += qty * price
            sell_qty += qty
            sold += qty
            fills += 1

        equity = cash + pos * last_mid
        min_equity = min(min_equity, equity)
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)

    pnl = cash + pos * last_mid
    return Result(
        product=product,
        config=config,
        pnl=pnl,
        min_equity=min_equity,
        max_drawdown=max_drawdown,
        final_position=pos,
        buy_qty=buy_qty,
        sell_qty=sell_qty,
        fills=fills,
    )


def default_grid(product: str) -> Tuple[List[float], List[float], List[int], List[int]]:
    if product == "VEV_5300":
        return (
            [38, 42, 46, 50, 54],
            [2.0, 3.0, 4.0],
            [200, 300],
            [20, 80],
        )
    if product == "VEV_5400":
        return (
            [10, 14, 16, 18, 20],
            [1.0, 2.0, 3.0],
            [150, 300],
            [20, 80],
        )
    if product == "VEV_5500":
        return (
            [4, 5, 6, 7, 8],
            [0.5, 1.0, 1.5],
            [150, 300],
            [20, 80],
        )
    if product == "VEV_5200":
        return (
            [35, 45, 55],
            [3.0, 6.0, 8.0],
            [50, 100],
            [20, 40],
        )
    return (
        [0, 5, 10, 15, 20, 30, 45],
        [1.0, 2.0, 4.0, 8.0],
        [50, 100, 200, 300],
        [10, 20, 40],
    )


def search_product(joined: pd.DataFrame, product: str, top_n: int) -> List[Result]:
    time_values, edges, caps, max_takes = default_grid(product)
    rows = materialize_product_rows(joined, product)
    results = []
    total = len(time_values) * len(edges) * len(caps) * len(max_takes)
    print(f"Searching {product}: {total} configs")
    for tv, edge, cap, max_take in itertools.product(time_values, edges, caps, max_takes):
        config = Config(float(tv), float(edge), int(cap), int(max_take))
        result = simulate_product(rows, product, config)
        results.append(result)
    results.sort(key=lambda r: r.pnl, reverse=True)
    return results[:top_n]


def print_results(results: List[Result]) -> None:
    for r in results:
        c = r.config
        print(
            f"{r.product:8s} pnl={r.pnl:10.2f}"
            f" min={r.min_equity:10.2f} dd={r.max_drawdown:10.2f}"
            f" pos={r.final_position:5d}"
            f" buy={r.buy_qty:6d} sell={r.sell_qty:6d} fills={r.fills:5d}"
            f" tv={c.time_value:5.1f} edge={c.edge:4.1f}"
            f" cap={c.cap:3d} max_take={c.max_take:3d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", default="all")
    parser.add_argument(
        "--products",
        default="VEV_5200,VEV_5300,VEV_5400,VEV_5500",
        help="Comma-separated voucher products",
    )
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    joined = load_joined(parse_days(args.days))
    products = [x.strip() for x in args.products.split(",") if x.strip()]
    for product in products:
        top = search_product(joined, product, args.top)
        print_results(top)
        print()


if __name__ == "__main__":
    main()
