"""
IMC Prosperity 4 — Round 1 Trader
===================================
Products: ASH_COATED_OSMIUM (ACO) | INTARIAN_PEPPER_ROOT (IPR)

Strategy:
  ACO  → Pure market-making around hardcoded fair value 10000.
          Half-life ≈ 2-3 ticks, spread ≈ 16 ticks.
          Take any crossing orders, then penny the best MM.

  IPR  → Price has a DETERMINISTIC linear trend: +1000 per day
          (slope = 0.001 per timestamp, R² = 0.9999 across all 3 training days).
          running_fair(t) = book_mid ≈ day_open + t * 0.001
          Two layers:
            1. Trend capture: stay max-long (+50) at all times.
               Buying at ask is fine — 50 pos * 1000 price rise >> friction.
            2. Market-making around running_fair to earn extra spread.

Data-derived constants (from analysis.ipynb):
  ACO spread mean=16, p10=16 → quote at offset 3 (inside spread ~98% of time)
  IPR spread mean=13, residual_std=2 → quote at offset 3
  Position limit = 50 for both (standard Prosperity R1)
"""

import json
import math
from typing import Any
from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState
)

# ─────────────────────────────────────────────────────────────
#  Logger (CMU-style, keeps logs under the 3750-char limit)
# ─────────────────────────────────────────────────────────────
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict, conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions, "", "",
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list:
        return [
            state.timestamp, trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades):
        out = []
        for arr in trades.values():
            for t in arr:
                out.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return out

    def compress_observations(self, obs: Observation) -> list:
        co = {}
        for p, o in obs.conversionObservations.items():
            co[p] = [o.bidPrice, o.askPrice, o.transportFees,
                     o.exportTariff, o.importTariff, o.sunlight, o.humidity]
        return [obs.plainValueObservations, co]

    def compress_orders(self, orders):
        out = []
        for arr in orders.values():
            for o in arr:
                out.append([o.symbol, o.price, o.quantity])
        return out

    def to_json(self, value) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[:max_length - 3] + "..."


logger = Logger()


# ─────────────────────────────────────────────────────────────
#  Trader
# ─────────────────────────────────────────────────────────────
class Trader:

    # position limits
    LIMIT = {"ASH_COATED_OSMIUM": 20, "INTARIAN_PEPPER_ROOT": 80}

    # ACO: Fallback fair value
    ACO_FAIR = 10000

    # Quote offsets (ticks from fair value).
    ACO_OFFSET = 1
    IPR_OFFSET = 3

    def run(self, state: TradingState):
        orders: dict[Symbol, list[Order]] = {}
        conversions = 0

        # ── persist IPR day-open price across ticks via traderData ──────────
        trader_data_out = {}
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        for product in state.order_depths:
            if product == "ASH_COATED_OSMIUM":
                orders[product], td = self._trade_aco(state, td)
            elif product == "INTARIAN_PEPPER_ROOT":
                orders[product], td = self._trade_ipr(state, td)

        trader_data_out = json.dumps(td)
        logger.flush(state, orders, conversions, trader_data_out)
        return orders, conversions, trader_data_out

    # ─────────────────────────────────────────────────────────
    #  ACO — Modified with Dynamic Weighted Mid-Price
    # ─────────────────────────────────────────────────────────
    def _trade_aco(self, state: TradingState, td: dict) -> tuple[list[Order], dict]:
        product = "ASH_COATED_OSMIUM"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT[product]
        result: list[Order] = []

        buy_capacity  = limit - pos
        sell_capacity = limit + pos

        # ── DYNAMIC FAIR VALUE (Order Book Imbalance) ───────────────────────
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        
        if best_bid is not None and best_ask is not None:
            bid_vol = od.buy_orders[best_bid]
            ask_vol = -od.sell_orders[best_ask]
            # Calculate Weighted Mid-Price
            fair = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
        else:
            fair = self.ACO_FAIR

        # ── Layer 1: Take any order that crosses fair value ──────────────────
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price < fair and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol))
                buy_capacity -= vol
                logger.print(f"ACO TAKE BUY  {vol}@{ask_price}  (fair={fair:.2f})")
            else:
                break

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price > fair and sell_capacity > 0:
                vol = min(od.buy_orders[bid_price], sell_capacity)
                result.append(Order(product, bid_price, -vol))
                sell_capacity -= vol
                logger.print(f"ACO TAKE SELL {vol}@{bid_price}  (fair={fair:.2f})")
            else:
                break

        # ── Layer 2: Quote our own market, penny the best MM ─────────────────
        buy_price  = int(fair) - self.ACO_OFFSET
        sell_price = int(fair) + self.ACO_OFFSET

        if best_ask is not None and best_ask - 1 > fair:
            sell_price = best_ask - 1
        if best_bid is not None and best_bid + 1 < fair:
            buy_price = best_bid + 1

        if buy_capacity > 0:
            skew = max(0, pos - 15) // 5
            t1 = buy_capacity // 2
            t2 = buy_capacity - t1
            result.append(Order(product, buy_price - skew,     t1))
            result.append(Order(product, buy_price - skew - 1, t2))
        if sell_capacity > 0:
            skew = max(0, -pos - 15) // 5
            t1 = sell_capacity // 2
            t2 = sell_capacity - t1
            result.append(Order(product, sell_price + skew,     -t1))
            result.append(Order(product, sell_price + skew + 1, -t2))

        logger.print(f"ACO fair={fair:.2f}  pos={pos}  buy_cap={limit-pos}  sell_cap={limit+pos}")
        return result, td

    # ─────────────────────────────────────────────────────────
    #  IPR — original trend capture + market making
    # ─────────────────────────────────────────────────────────
    def _trade_ipr(self, state: TradingState, td: dict) -> tuple[list[Order], dict]:
        product = "INTARIAN_PEPPER_ROOT"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT[product]
        ts = state.timestamp
        result: list[Order] = []

        buy_capacity  = limit - pos
        sell_capacity = limit + pos

        ob_mid = None
        if od.sell_orders and od.buy_orders:
            best_ask_ipr = min(od.sell_orders.keys())
            best_bid_ipr = max(od.buy_orders.keys())
            ob_mid = (best_ask_ipr + best_bid_ipr) / 2

        last_ts = td.get('ipr_last_ts', -1)
        if (ts < last_ts or ts == 0) and ob_mid is not None:
            td['ipr_day_open'] = ob_mid
        td['ipr_last_ts'] = ts

        day_open = td.get('ipr_day_open', None)

        if day_open is not None:
            model_fair = day_open + ts * 0.001
            if ob_mid is not None:
                fair = 0.7 * model_fair + 0.3 * ob_mid
            else:
                fair = model_fair
        elif ob_mid is not None:
            fair = ob_mid
        else:
            return result, td

        fair_int_low  = math.floor(fair)
        fair_int_high = math.ceil(fair)

        logger.print(f"IPR ts={ts}  fair={fair:.2f}  pos={pos}  day_open={day_open}  ob_mid={ob_mid}")

        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price < fair and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol))
                buy_capacity -= vol
            else:
                break

        if buy_capacity > 0 and od.sell_orders:
            best_ask = min(od.sell_orders.keys())
            if buy_capacity >= 3:
                take_vol = min(-od.sell_orders[best_ask], buy_capacity)
                result.append(Order(product, best_ask, take_vol))
                buy_capacity -= take_vol
                logger.print(f"IPR TREND BUY {take_vol}@{best_ask}")

        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None

        buy_price  = fair_int_low  - self.IPR_OFFSET
        sell_price = fair_int_high + self.IPR_OFFSET

        if best_ask is not None and best_ask - 1 > fair:
            sell_price = best_ask - 1
        if best_bid is not None and best_bid + 1 < fair:
            buy_price = best_bid + 1

        if buy_capacity > 0:
            result.append(Order(product, buy_price, buy_capacity))

        if sell_capacity > 0 and pos >= limit - 1:
            result.append(Order(product, sell_price, -sell_capacity))

        return result, td