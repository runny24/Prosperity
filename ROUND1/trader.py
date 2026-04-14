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
    LIMIT = {"ASH_COATED_OSMIUM": 50, "INTARIAN_PEPPER_ROOT": 50}

    # ACO: hardcoded fair value (mean-reversion target)
    ACO_FAIR = 10000

    # Quote offsets (ticks from fair value).
    # ACO spread ≈ 16 ticks  → offset 3 is inside spread 98% of time
    # IPR spread ≈ 13 ticks  → offset 3 is inside spread 97% of time
    ACO_OFFSET = 3
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
                orders[product] = self._trade_aco(state)
            elif product == "INTARIAN_PEPPER_ROOT":
                orders[product], td = self._trade_ipr(state, td)

        trader_data_out = json.dumps(td)
        logger.flush(state, orders, conversions, trader_data_out)
        return orders, conversions, trader_data_out

    # ─────────────────────────────────────────────────────────
    #  ACO — mean-reversion market maker
    # ─────────────────────────────────────────────────────────
    def _trade_aco(self, state: TradingState) -> list[Order]:
        product = "ASH_COATED_OSMIUM"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT[product]
        result: list[Order] = []

        # ── Build sorted orderbooks (positive volumes) ───────────────────────
        buys  = dict(sorted(od.buy_orders.items(),  reverse=True))  # bid: high→low
        sells = dict(sorted(od.sell_orders.items()))                 # ask: low→high

        best_bid  = max(buys)  if buys  else None
        best_ask  = min(sells) if sells else None
        worst_bid = min(buys)  if buys  else None   # lowest visible bid  (wall)
        worst_ask = max(sells) if sells else None   # highest visible ask (wall)

        # wall_mid: midpoint of the outer book edges — more stable than best/ask mid
        # If one side is missing, fall back to hardcoded fair (don't trade passively)
        if worst_bid is None or worst_ask is None:
            wall_mid = self.ACO_FAIR
            one_sided = True
        else:
            wall_mid = (worst_bid + worst_ask) / 2
            one_sided = False

        buy_capacity  = limit - pos
        sell_capacity = limit + pos

        # ── Layer 1: Take orders with clear edge vs wall_mid ─────────────────
        # Take asks clearly below fair (≤ wall_mid - 1)
        for ask, vol in sells.items():
            avail = -vol  # sell_orders have negative convention
            if ask <= wall_mid - 1 and buy_capacity > 0:
                qty = min(avail, buy_capacity)
                result.append(Order(product, ask, qty))
                buy_capacity -= qty
                logger.print(f"ACO TAKE BUY  {qty}@{ask}  wall_mid={wall_mid:.1f}")
            elif ask <= wall_mid and pos < 0:
                # Short inventory: also take at fair to unwind position
                qty = min(avail, buy_capacity, -pos)
                if qty > 0:
                    result.append(Order(product, ask, qty))
                    buy_capacity -= qty
            else:
                break

        # Take bids clearly above fair (≥ wall_mid + 1)
        for bid, vol in buys.items():
            if bid >= wall_mid + 1 and sell_capacity > 0:
                qty = min(vol, sell_capacity)
                result.append(Order(product, bid, -qty))
                sell_capacity -= qty
                logger.print(f"ACO TAKE SELL {qty}@{bid}  wall_mid={wall_mid:.1f}")
            elif bid >= wall_mid and pos > 0:
                # Long inventory: also take at fair to unwind position
                qty = min(vol, sell_capacity, pos)
                if qty > 0:
                    result.append(Order(product, bid, -qty))
                    sell_capacity -= qty
            else:
                break

        # ── Layer 2: Make the market (skip entirely if one-sided book) ───────
        if one_sided:
            return result

        # Start at inner wall edges, then penny any meaningful market maker
        buy_price  = worst_bid + 1
        sell_price = worst_ask - 1

        # Penny the best bid if it has volume > 1 and is still below fair
        for bid, vol in buys.items():
            if vol > 1 and bid + 1 < wall_mid:
                buy_price = max(buy_price, bid + 1)
                break
            if bid < wall_mid:
                buy_price = max(buy_price, bid)
                break

        # Penny the best ask if it has volume > 1 and is still above fair
        for ask, vol in sells.items():
            avail = -vol
            if avail > 1 and ask - 1 > wall_mid:
                sell_price = min(sell_price, ask - 1)
                break
            if ask > wall_mid:
                sell_price = min(sell_price, ask)
                break

        # Post full remaining capacity
        if buy_capacity > 0:
            result.append(Order(product, buy_price,  buy_capacity))
        if sell_capacity > 0:
            result.append(Order(product, sell_price, -sell_capacity))

        logger.print(f"ACO MAKE  bid={buy_price} ask={sell_price}  pos={pos}  wall_mid={wall_mid:.1f}")
        return result

    # ─────────────────────────────────────────────────────────
    #  IPR — trend capture + market making
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

        # ── Estimate running fair value ──────────────────────────────────────
        # Strategy A: use order book mid (works when both sides exist)
        # Strategy B: use the linear model fair = day_open + ts * 0.001
        #             requires knowing day_open (stored in traderData)

        ob_mid = None
        if od.sell_orders and od.buy_orders:
            # Use WORST bid/ask midpoint (same as CMU for KELP/SQUID)
            # This is more stable than best bid/ask in wide-spread markets
            worst_ask = max(od.sell_orders.keys())
            worst_bid = min(od.buy_orders.keys())
            ob_mid = (worst_ask + worst_bid) / 2

        # Seed / refresh day_open when timestamp resets (new day)
        if ts == 0 and ob_mid is not None:
            td['ipr_day_open'] = ob_mid

        day_open = td.get('ipr_day_open', None)

        if day_open is not None:
            # linear model: price = day_open + ts * 0.001
            model_fair = day_open + ts * 0.001
            # blend: trust model 70%, book 30% (model is R²=0.9999)
            if ob_mid is not None:
                fair = 0.7 * model_fair + 0.3 * ob_mid
            else:
                fair = model_fair
        elif ob_mid is not None:
            fair = ob_mid
        else:
            # no information — skip this tick
            return result, td

        fair_int_low  = math.floor(fair)
        fair_int_high = math.ceil(fair)

        logger.print(f"IPR ts={ts}  fair={fair:.2f}  pos={pos}  "
                     f"day_open={day_open}  ob_mid={ob_mid}")

        # ── Layer 1: Take mispriced orders (cross below/above fair) ──────────
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price < fair and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol))
                buy_capacity -= vol
            else:
                break

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price > fair and sell_capacity > 0:
                vol = min(od.buy_orders[bid_price], sell_capacity)
                result.append(Order(product, bid_price, -vol))
                sell_capacity -= vol
            else:
                break

        # ── Layer 2: Trend capture — aggressively buy to stay max long ───────
        # The price rises +1000/day. Being long 50 the whole day = +50,000 PnL.
        # So we want to be at +50 whenever possible.
        # We'll take the best ask if we're not at limit yet.
        if buy_capacity > 0 and od.sell_orders:
            best_ask = min(od.sell_orders.keys())
            # Only take market if we're meaningfully below limit
            if buy_capacity >= 3:
                take_vol = min(-od.sell_orders[best_ask], buy_capacity)
                result.append(Order(product, best_ask, take_vol))
                buy_capacity -= take_vol
                logger.print(f"IPR TREND BUY {take_vol}@{best_ask}")

        # ── Layer 3: Quote our own market to earn spread ──────────────────────
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None

        buy_price  = fair_int_low  - self.IPR_OFFSET
        sell_price = fair_int_high + self.IPR_OFFSET

        if best_ask is not None and best_ask - 1 > fair:
            sell_price = best_ask - 1
        if best_bid is not None and best_bid + 1 < fair:
            buy_price = best_bid + 1

        # Since we want to stay long, be more aggressive on buys, relaxed on sells
        # Scale down sell aggressiveness — only sell if well above fair
        if buy_capacity > 0:
            result.append(Order(product, buy_price, buy_capacity))

        # Only post sell orders when very long (to collect spread while capped)
        if sell_capacity > 0 and pos > 30:
            result.append(Order(product, sell_price, -sell_capacity))

        return result, td
