"""
Round 3 v15 — Feature-flag testbed on top of v14 confirmed baseline (~10,594)
==============================================================================
Confirmed baseline changes (always on):
  • LOWER_5300_SELL_EDGE = True  (sell VEV_5300 when bid > 52 instead of > 53)
    → +104 PnL vs v13 in isolated test (438898.log)

Three new experimental flags (all False = baseline ~10,594):
  Toggle ONE at a time to isolate each improvement's contribution.

FEATURE_LOWER_5400_SELL_EDGE   [TEST FIRST — largest expected gain ~+500-800]
  VEV_5400 TAKE_EDGE_SELL: 1.0 → 0.5
  Effect: sell when bid ≥ 17 instead of bid ≥ 18.
  NPC bid=17 occurs on 22.9% of ticks vs 6.9% for bid≥18 (3.3x more fills).
  Could enable a 3rd complete buy-sell cycle on VEV_5400 each day.

FEATURE_LOWER_5300_BUY_EDGE   [TEST SECOND]
  VEV_5300 TAKE_EDGE_BUY: 3.0 → 2.0
  Effect: buy when ask < 48 instead of ask < 47.
  NPC ask<48 occurs on 16.6% of ticks vs 9.2% for ask<47 (~1.8x more buys).

FEATURE_SCALE_VEV_5100        [TEST THIRD — independent of above two]
  VEV_5100 SOFT_LIMIT: 30 → 60
  Effect: in 438898.log VEV_5100 sold exactly 30 (hard limit hit) with
  318 ticks where bid>176 was available. Limit was binding, not fill ceiling.
  Doubling the cap should nearly double VEV_5100 PnL.
"""

import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState
)

# ── EXPERIMENT FLAGS ──────────────────────────────────────────────────────────
# Baseline (all False) ≈ +10,594  (v14 with LOWER_5300_SELL_EDGE confirmed)
# Toggle ONE at a time to isolate each improvement's contribution.
# Recommended test order: 1st, 2nd, 3rd (as listed above).

FEATURE_LOWER_5400_SELL_EDGE = False  # VEV_5400 sell edge 1.0 → 0.5 (sell at bid≥17)
FEATURE_LOWER_5300_BUY_EDGE  = False  # VEV_5300 buy  edge 3.0 → 2.0 (buy  at ask<48)
FEATURE_SCALE_VEV_5100       = False  # VEV_5100 SOFT_LIMIT 30 → 60
# ─────────────────────────────────────────────────────────────────────────────


class Logger:
    def __init__(self): self.logs = ""; self.max_log_length = 3750
    def print(self, *objects, sep=" ", end="\n"): self.logs += sep.join(map(str, objects)) + end
    def flush(self, state, orders, conversions, trader_data):
        bl = len(self.to_json([self.cs(state, ""), self.co(orders), conversions, "", ""]))
        m = (self.max_log_length - bl) // 3
        print(self.to_json([self.cs(state, self.t(state.traderData, m)), self.co(orders), conversions, self.t(trader_data, m), self.t(self.logs, m)]))
        self.logs = ""
    def cs(self, s, td):
        return [s.timestamp, td, [[l.symbol,l.product,l.denomination] for l in s.listings.values()],
                {k:[v.buy_orders,v.sell_orders] for k,v in s.order_depths.items()},
                [[t.symbol,t.price,t.quantity,t.buyer,t.seller,t.timestamp] for a in s.own_trades.values() for t in a],
                [[t.symbol,t.price,t.quantity,t.buyer,t.seller,t.timestamp] for a in s.market_trades.values() for t in a],
                s.position, self.cobs(s.observations)]
    def cobs(self, obs):
        co = {}
        for p, o in obs.conversionObservations.items():
            co[p] = [o.bidPrice,o.askPrice,o.transportFees,o.exportTariff,o.importTariff,o.sunlight,o.humidity]
        return [obs.plainValueObservations, co]
    def co(self, orders): return [[o.symbol,o.price,o.quantity] for a in orders.values() for o in a]
    def to_json(self, v): return json.dumps(v, cls=ProsperityEncoder, separators=(",",":"))
    def t(self, v, m): return v if len(v) <= m else v[:m-3] + "..."

logger = Logger()


class Trader:

    # ── Position limits ────────────────────────────────────────────────────────
    LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK":       200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300, "VEV_4500": 300,
        "VEV_5000": 300, "VEV_5100": 300, "VEV_5200": 300,
        "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
        "VEV_6000": 300, "VEV_6500": 300,
    }

    # Soft limits — conservative caps for lower-alpha vouchers
    # FEATURE_SCALE_VEV_5100 doubles VEV_5100 from 30 → 60
    SOFT_LIMITS: Dict[str, int] = {
        "VEV_5000": 80,
        "VEV_5100": 60 if FEATURE_SCALE_VEV_5100 else 30,
        "VEV_5200": 50,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
    }

    VOUCHER_STRIKES: Dict[str, int] = {
        "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    }

    # Base time values — empirically calibrated.
    VOUCHER_TIME_VALUE: Dict[str, float] = {
        "VEV_5000": 5.0,
        "VEV_5100": 8.0,
        "VEV_5200": 35.0,
        "VEV_5300": 50.0,
        "VEV_5400": 16.0,
        "VEV_5500": 5.0,
    }

    DISABLED_VOUCHERS = {"VEV_4000", "VEV_4500", "VEV_6000", "VEV_6500"}

    # Take edges for BUY side (buy when ask < fair - edge)
    # FEATURE_LOWER_5300_BUY_EDGE: VEV_5300 buy edge 3.0 → 2.0 (buy at ask<48)
    TAKE_EDGE_BUY: Dict[str, float] = {
        "HYDROGEL_PACK":       8.0,
        "VELVETFRUIT_EXTRACT": 2.0,
        "VEV_5000": 6.0,
        "VEV_5100": 3.0,
        "VEV_5200": 3.0,
        "VEV_5300": 2.0 if FEATURE_LOWER_5300_BUY_EDGE else 3.0,
        "VEV_5400": 1.0,
        "VEV_5500": 0.5,
    }

    # Take edges for SELL side (sell when bid > fair + edge)
    # LOWER_5300_SELL_EDGE is now CONFIRMED baseline: VEV_5300 sell edge = 2.0
    # FEATURE_LOWER_5400_SELL_EDGE: VEV_5400 sell edge 1.0 → 0.5 (sell at bid≥17)
    TAKE_EDGE_SELL: Dict[str, float] = {
        "HYDROGEL_PACK":       8.0,
        "VELVETFRUIT_EXTRACT": 2.0,
        "VEV_5000": 6.0,
        "VEV_5100": 3.0,
        "VEV_5200": 3.0,
        "VEV_5300": 2.0,   # confirmed baseline (was 3.0 in v13)
        "VEV_5400": 0.5 if FEATURE_LOWER_5400_SELL_EDGE else 1.0,
        "VEV_5500": 0.5,
    }

    MAX_TAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK":       24,
        "VELVETFRUIT_EXTRACT": 50,
        "VEV_5000": 12,
        "VEV_5100": 8,
        "VEV_5200": 10,
        "VEV_5300": 100,
        "VEV_5400": 100,
        "VEV_5500": 80,
    }

    MAKE_EDGE: Dict[str, int] = {
        "HYDROGEL_PACK":       10,
        "VELVETFRUIT_EXTRACT":  3,
    }

    MAX_MAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK":       24,
        "VELVETFRUIT_EXTRACT": 50,
    }

    EMA_ALPHA: Dict[str, float] = {
        "HYDROGEL_PACK":       0.08,
        "VELVETFRUIT_EXTRACT": 0.12,
    }

    # ── Spread overlay (VEV_5300 / VEV_5400) ──────────────────────────────────
    SPREAD_PAIR_MAX_TAKE = 30
    SPREAD_POSITION_CAP  = 120
    SPREAD_HIGH_SELL     = 36.5
    SPREAD_LOW_BUY       = 24.0

    # ── VEV_5500 strict rule ───────────────────────────────────────────────────
    VEV_5500_BUY_MAX  = 5
    VEV_5500_SELL_MIN = 8

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        # ── Spot fair values ───────────────────────────────────────────────────
        velvet_fair = self._spot_fair("VELVETFRUIT_EXTRACT", state, td)

        if "HYDROGEL_PACK" in state.order_depths:
            hydrogel_fair = self._spot_fair("HYDROGEL_PACK", state, td)
            orders["HYDROGEL_PACK"] = self._trade_spot(
                "HYDROGEL_PACK", state, td, known_fair=hydrogel_fair
            )

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            orders["VELVETFRUIT_EXTRACT"] = self._trade_spot(
                "VELVETFRUIT_EXTRACT", state, td, known_fair=velvet_fair
            )

        if velvet_fair is not None:
            for product in state.order_depths:
                if product.startswith("VEV_"):
                    orders[product] = self._trade_voucher(product, velvet_fair, state)
            self._trade_5300_5400_spread(state, orders)

        tdo = json.dumps(td, separators=(",", ":"))
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ── Spot: volume-weighted EMA fair + take + passive quotes ────────────────
    def _trade_spot(
        self, product: str, state: TradingState, td: Dict,
        known_fair: Optional[float] = None
    ) -> List[Order]:
        fair = known_fair if known_fair is not None else self._spot_fair(product, state, td)
        if fair is None:
            return []
        od     = state.order_depths[product]
        orders: List[Order] = []
        self._take_mispriced(product, od, fair, state, orders)
        make_edge = self.MAKE_EDGE.get(product)
        if make_edge is not None:
            self._add_passive_quotes(product, od, fair, state, orders, make_edge)
        return orders

    # ── Voucher: intrinsic + time value ───────────────────────────────────────
    def _trade_voucher(
        self, product: str, underlying_fair: float, state: TradingState
    ) -> List[Order]:
        if product in self.DISABLED_VOUCHERS or product not in self.VOUCHER_STRIKES:
            return []
        if product == "VEV_5500":
            return self._trade_5500_strict(product, state)

        strike     = self.VOUCHER_STRIKES[product]
        time_value = self.VOUCHER_TIME_VALUE[product]
        fair       = max(underlying_fair - strike, 0.0) + time_value
        od         = state.order_depths[product]
        orders: List[Order] = []
        self._take_mispriced(product, od, fair, state, orders)
        return orders

    # ── VEV_5500 strict low-price rule ─────────────────────────────────────────
    def _trade_5500_strict(self, product: str, state: TradingState) -> List[Order]:
        od = state.order_depths[product]
        orders: List[Order] = []
        buy_cap, sell_cap = self._remaining_capacity(product, state, orders)
        max_take = self.MAX_TAKE_SIZE.get(product, 80)

        bought = 0
        for ask_price in sorted(od.sell_orders):
            if ask_price > self.VEV_5500_BUY_MAX or buy_cap <= 0 or bought >= max_take:
                break
            qty = min(-od.sell_orders[ask_price], buy_cap, max_take - bought)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_cap -= qty; bought += qty

        pos_after = state.position.get(product, 0) + bought
        if pos_after <= 0:
            return orders

        sold = 0
        for bid_price in sorted(od.buy_orders, reverse=True):
            if bid_price < self.VEV_5500_SELL_MIN or sell_cap <= 0 or sold >= max_take:
                break
            qty = min(od.buy_orders[bid_price], sell_cap, pos_after, max_take - sold)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_cap -= qty; sold += qty; pos_after -= qty

        return orders

    # ── VEV_5300 / VEV_5400 spread overlay ────────────────────────────────────
    def _trade_5300_5400_spread(
        self, state: TradingState, result: Dict[str, List[Order]]
    ) -> None:
        p_hi, p_lo = "VEV_5300", "VEV_5400"
        if p_hi not in state.order_depths or p_lo not in state.order_depths:
            return

        od_hi = state.order_depths[p_hi]
        od_lo = state.order_depths[p_lo]
        bid_hi, ask_hi = self._best_bid_ask(od_hi)
        bid_lo, ask_lo = self._best_bid_ask(od_lo)
        if None in (bid_hi, ask_hi, bid_lo, ask_lo):
            return

        sell_spread = bid_hi - ask_lo
        buy_spread  = ask_hi - bid_lo
        spread_pos  = state.position.get(p_hi, 0) - state.position.get(p_lo, 0)
        orders_hi   = result.setdefault(p_hi, [])
        orders_lo   = result.setdefault(p_lo, [])

        if sell_spread >= self.SPREAD_HIGH_SELL and spread_pos > -self.SPREAD_POSITION_CAP:
            qty = min(
                self.SPREAD_PAIR_MAX_TAKE,
                od_hi.buy_orders.get(bid_hi, 0),
                -od_lo.sell_orders.get(ask_lo, 0),
                self.SPREAD_POSITION_CAP + spread_pos,
                self._remaining_capacity(p_hi, state, orders_hi)[1],
                self._remaining_capacity(p_lo, state, orders_lo)[0],
            )
            if qty > 0:
                orders_hi.append(Order(p_hi, bid_hi, -qty))
                orders_lo.append(Order(p_lo, ask_lo,  qty))

        elif buy_spread <= self.SPREAD_LOW_BUY and spread_pos < self.SPREAD_POSITION_CAP:
            qty = min(
                self.SPREAD_PAIR_MAX_TAKE,
                -od_hi.sell_orders.get(ask_hi, 0),
                od_lo.buy_orders.get(bid_lo, 0),
                self.SPREAD_POSITION_CAP - spread_pos,
                self._remaining_capacity(p_hi, state, orders_hi)[0],
                self._remaining_capacity(p_lo, state, orders_lo)[1],
            )
            if qty > 0:
                orders_hi.append(Order(p_hi, ask_hi,  qty))
                orders_lo.append(Order(p_lo, bid_lo, -qty))

    # ── Spot fair: volume-weighted mid → EMA ──────────────────────────────────
    def _spot_fair(
        self, product: str, state: TradingState, td: Dict
    ) -> Optional[float]:
        if product not in state.order_depths:
            return None
        od = state.order_depths[product]
        best_bid, best_ask = self._best_bid_ask(od)

        raw_fair: Optional[float] = None
        if best_bid is not None and best_ask is not None:
            bid_vol = max(od.buy_orders.get(best_bid, 0), 0)
            ask_vol = max(-od.sell_orders.get(best_ask, 0), 0)
            raw_fair = (
                (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
                if bid_vol + ask_vol > 0
                else (best_bid + best_ask) / 2.0
            )
        elif best_bid is not None:
            raw_fair = float(best_bid)
        elif best_ask is not None:
            raw_fair = float(best_ask)

        if raw_fair is None:
            prev = td.get(product + "_ema")
            return float(prev) if prev is not None else None

        alpha = self.EMA_ALPHA.get(product, 0.10)
        prev  = td.get(product + "_ema")
        ema   = raw_fair if prev is None else alpha * raw_fair + (1.0 - alpha) * float(prev)
        td[product + "_ema"] = ema
        return ema

    # ── Take mispriced liquidity — asymmetric buy/sell edges ──────────────────
    def _take_mispriced(
        self, product: str, od: OrderDepth, fair: float,
        state: TradingState, orders: List[Order]
    ) -> None:
        buy_cap, sell_cap = self._remaining_capacity(product, state, orders)
        edge_buy  = self.TAKE_EDGE_BUY.get(product,  999_999.0)
        edge_sell = self.TAKE_EDGE_SELL.get(product, 999_999.0)
        max_take  = self.MAX_TAKE_SIZE.get(product, 0)
        if max_take == 0:
            return

        bought = 0
        for ask_price in sorted(od.sell_orders):
            if ask_price >= fair - edge_buy or buy_cap <= 0 or bought >= max_take:
                break
            qty = min(-od.sell_orders[ask_price], buy_cap, max_take - bought)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_cap -= qty; bought += qty

        sold = 0
        for bid_price in sorted(od.buy_orders, reverse=True):
            if bid_price <= fair + edge_sell or sell_cap <= 0 or sold >= max_take:
                break
            qty = min(od.buy_orders[bid_price], sell_cap, max_take - sold)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_cap -= qty; sold += qty

    # ── Passive quoting for spots ──────────────────────────────────────────────
    def _add_passive_quotes(
        self, product: str, od: OrderDepth, fair: float,
        state: TradingState, orders: List[Order], edge: int
    ) -> None:
        best_bid, best_ask = self._best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return
        buy_cap, sell_cap = self._remaining_capacity(product, state, orders)
        max_make = self.MAX_MAKE_SIZE.get(product, 0)
        if max_make <= 0:
            return

        buy_px  = int(math.floor(fair - edge))
        sell_px = int(math.ceil(fair  + edge))

        if best_bid + 1 < fair - 1:
            buy_px = max(buy_px, best_bid + 1)
        if best_ask - 1 > fair + 1:
            sell_px = min(sell_px, best_ask - 1)

        if buy_px  >= best_ask: buy_px  = best_ask - 1
        if sell_px <= best_bid: sell_px = best_bid + 1
        buy_px  = max(0, buy_px)
        sell_px = max(1, sell_px)

        pos   = state.position.get(product, 0)
        limit = self.LIMITS[product]
        buy_scale  = max(0.0, 1.0 - max( pos, 0) / limit)
        sell_scale = max(0.0, 1.0 - max(-pos, 0) / limit)

        buy_qty  = min(buy_cap,  max(1, int(max_make * buy_scale)))
        sell_qty = min(sell_cap, max(1, int(max_make * sell_scale)))

        if buy_qty  > 0: orders.append(Order(product, buy_px,   buy_qty))
        if sell_qty > 0: orders.append(Order(product, sell_px, -sell_qty))

    # ── Remaining capacity (respects soft limits) ─────────────────────────────
    def _remaining_capacity(
        self, product: str, state: TradingState, existing: List[Order]
    ) -> Tuple[int, int]:
        limit    = self.SOFT_LIMITS.get(product, self.LIMITS.get(product, 200))
        position = state.position.get(product, 0)
        pending_buy = pending_sell = 0
        for o in existing:
            if o.quantity > 0: pending_buy  += o.quantity
            else:              pending_sell += -o.quantity
        return max(0, limit - position - pending_buy), max(0, limit + position - pending_sell)

    # ── Best bid / ask ─────────────────────────────────────────────────────────
    def _best_bid_ask(
        self, od: OrderDepth
    ) -> Tuple[Optional[int], Optional[int]]:
        return (
            max(od.buy_orders)  if od.buy_orders  else None,
            min(od.sell_orders) if od.sell_orders else None,
        )
