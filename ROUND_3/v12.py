"""
Round 3 v12 — Fix HYDROGEL Anchor Bug (EMA instead of fixed 10000)
====================================================================
CHANGES FROM v11:

KEY FIX: HYDROGEL_PACK fair value switched from fixed anchor=10000 → slow EMA

ROOT CAUSE IDENTIFIED from 432353.log (v11 result = +8,311):
  - v11 non-HYDROGEL PnL: +9,504 (VEV logic working well)
  - v11 HYDROGEL PnL:     -1,193 (the one broken piece)

WHY FIXED ANCHOR=10000 FAILED:
  - Day 2 test HYDROGEL mean = 9,979 (NOT 10,000)
  - 60% of ticks traded below 9,990
  - Fixed anchor=10000 forces buy_px up to best_ask-1 ≈ 9,967 every tick
    when market mid is ~9,960 → paying ~8.65 ticks above market mid
  - Accumulated 200-contract LONG at avg 9,989 when market ended at 9,960
  - Net unrealized loss: 200 * (9960 - 9989) = -5,814
  - Sell side never fires: sell_px=best_ask=9,968, last in NPC queue

WHAT THE FRIEND DOES (and why it works):
  - Volume-weighted EMA, alpha=0.08 (very slow, tracks actual market)
  - After 100+ ticks at mid=9,979, EMA converges to ~9,979
  - buy_px = floor(9,979 - 10) = 9,969 → inside NPC spread, proper MM
  - sell_px = min(ceil(9,979 + 10), best_ask-1) = 9,967 → inside spread
  - Both sides get filled → flat inventory → no adverse selection

THE FIX (single change):
  - Remove HYDROGEL_ANCHOR constant
  - HYDROGEL now uses _spot_fair() like VEV: volume-weighted, EMA alpha=0.08
  - All other parameters (MAKE_EDGE=10, MAX_MAKE_SIZE=24, TAKE_EDGE=8) unchanged

Expected improvement: HYDROGEL goes from -1,193 → +500 to +1,000
Expected total: 8,311 + ~1,700 ≈ 10,000+ (matching friend's +10,151)

Unchanged from v11:
  - Correct position limits (HYDROGEL=200, options=300)
  - Friend's time_value model for vouchers
  - VEV_5300/5400 spread overlay
  - VEV_5500 strict low-price rule
  - Disabled VEV_4000/4500/6000/6500
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState
)


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

    # ── Position limits (corrected from v10) ───────────────────────────────────
    LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK":       200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300, "VEV_4500": 300,
        "VEV_5000": 300, "VEV_5100": 300, "VEV_5200": 300,
        "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
        "VEV_6000": 300, "VEV_6500": 300,
    }

    # Soft (conservative) limits for smaller-alpha vouchers
    SOFT_LIMITS: Dict[str, int] = {
        "VEV_5000": 80,
        "VEV_5100": 30,
        "VEV_5200": 50,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
    }

    VOUCHER_STRIKES: Dict[str, int] = {
        "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    }

    # Fair time value above intrinsic — empirically calibrated from backtests
    VOUCHER_TIME_VALUE: Dict[str, float] = {
        "VEV_5000": 5.0,
        "VEV_5100": 8.0,
        "VEV_5200": 35.0,
        "VEV_5300": 50.0,   # fair=50, take when ask<47; NPC ask avg=45.5 → fills ~50% of ticks
        "VEV_5400": 16.0,   # fair=16, take when ask<15
        "VEV_5500": 5.0,    # handled by strict low-price rule, not this model
    }

    # Vouchers to completely ignore (zero edge or consistent losses)
    DISABLED_VOUCHERS = {"VEV_4000", "VEV_4500", "VEV_6000", "VEV_6500"}

    # How far inside fair we must be before taking (prevents taking at bad prices)
    TAKE_EDGE: Dict[str, float] = {
        "HYDROGEL_PACK":       8.0,
        "VELVETFRUIT_EXTRACT": 2.0,
        "VEV_5000": 6.0,
        "VEV_5100": 3.0,
        "VEV_5200": 3.0,
        "VEV_5300": 3.0,    # buy when ask < 47 (fair=50, edge=3)
        "VEV_5400": 1.0,    # buy when ask < 15
        "VEV_5500": 0.5,
    }

    # Max units to take in a single tick per product
    MAX_TAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK":       24,
        "VELVETFRUIT_EXTRACT": 30,
        "VEV_5000": 12,
        "VEV_5100": 8,
        "VEV_5200": 10,
        "VEV_5300": 100,
        "VEV_5400": 100,
        "VEV_5500": 80,
    }

    # Passive quoting parameters for spots only
    MAKE_EDGE: Dict[str, int] = {
        "HYDROGEL_PACK":       10,
        "VELVETFRUIT_EXTRACT":  3,
    }

    MAX_MAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK":       24,   # ~12% of limit per tick
        "VELVETFRUIT_EXTRACT": 36,   # ~18% of limit per tick
    }

    # EMA decay for spot fair values (slower = less noise chasing)
    EMA_ALPHA: Dict[str, float] = {
        "HYDROGEL_PACK":       0.08,
        "VELVETFRUIT_EXTRACT": 0.12,
    }

    # ── Spread overlay parameters (VEV_5300 / VEV_5400) ───────────────────────
    SPREAD_PAIR_MAX_TAKE = 30          # max contracts per leg per tick
    SPREAD_POSITION_CAP  = 120         # max net spread position (5300 - 5400)
    SPREAD_HIGH_SELL     = 36.5        # sell 5300 / buy 5400 when bid_5300 - ask_5400 ≥ this
    SPREAD_LOW_BUY       = 24.0        # buy 5300 / sell 5400 when ask_5300 - bid_5400 ≤ this

    # ── VEV_5500 strict low-price rule ─────────────────────────────────────────
    VEV_5500_BUY_MAX   = 5             # buy only if ask ≤ 5
    VEV_5500_SELL_MIN  = 8             # sell only if bid ≥ 8

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        # ── Compute VEV spot fair value (volume-weighted EMA) ──────────────────
        velvet_fair = self._spot_fair("VELVETFRUIT_EXTRACT", state, td)

        # ── HYDROGEL_PACK ──────────────────────────────────────────────────────
        if "HYDROGEL_PACK" in state.order_depths:
            hydrogel_fair = self._spot_fair("HYDROGEL_PACK", state, td)
            orders["HYDROGEL_PACK"] = self._trade_spot(
                "HYDROGEL_PACK", state, td, known_fair=hydrogel_fair
            )

        # ── VELVETFRUIT_EXTRACT ────────────────────────────────────────────────
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            orders["VELVETFRUIT_EXTRACT"] = self._trade_spot(
                "VELVETFRUIT_EXTRACT", state, td, known_fair=velvet_fair
            )

        # ── Vouchers ───────────────────────────────────────────────────────────
        if velvet_fair is not None:
            for product in state.order_depths:
                if product.startswith("VEV_"):
                    orders[product] = self._trade_voucher(product, velvet_fair, state)

            # Spread overlay (modifies existing voucher orders in-place)
            self._trade_5300_5400_spread(state, orders)

        tdo = json.dumps(td, separators=(",", ":"))
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ── Spot (VEV underlying and HYDROGEL): volume-weighted EMA + take + make ───
    def _trade_spot(
        self, product: str, state: TradingState, td: Dict,
        known_fair: Optional[float] = None
    ) -> List[Order]:
        fair = known_fair if known_fair is not None else self._spot_fair(product, state, td)
        if fair is None:
            return []

        od = state.order_depths[product]
        orders: List[Order] = []
        self._take_mispriced(product, od, fair, state, orders)

        make_edge = self.MAKE_EDGE.get(product)
        if make_edge is not None:
            self._add_passive_quotes(product, od, fair, state, orders, make_edge)

        return orders

    # ── Voucher trading: intrinsic + calibrated time value ─────────────────────
    def _trade_voucher(
        self, product: str, underlying_fair: float, state: TradingState
    ) -> List[Order]:
        if product in self.DISABLED_VOUCHERS:
            return []
        if product not in self.VOUCHER_STRIKES:
            return []

        # VEV_5500 uses a separate strict low-price-only logic
        if product == "VEV_5500":
            return self._trade_5500_strict(product, state)

        strike     = self.VOUCHER_STRIKES[product]
        time_value = self.VOUCHER_TIME_VALUE[product]
        fair       = max(underlying_fair - strike, 0.0) + time_value

        od     = state.order_depths[product]
        orders: List[Order] = []
        self._take_mispriced(product, od, fair, state, orders)
        return orders

    # ── VEV_5500: only buy at ≤5, sell at ≥8 ──────────────────────────────────
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
                buy_cap -= qty
                bought  += qty

        pos_after_buys = state.position.get(product, 0) + bought
        if pos_after_buys <= 0:
            return orders   # nothing to sell

        sold = 0
        for bid_price in sorted(od.buy_orders, reverse=True):
            if bid_price < self.VEV_5500_SELL_MIN or sell_cap <= 0 or sold >= max_take:
                break
            qty = min(od.buy_orders[bid_price], sell_cap, pos_after_buys, max_take - sold)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_cap       -= qty
                sold           += qty
                pos_after_buys -= qty

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

        sell_spread = bid_hi - ask_lo  # executable spread when selling 5300 / buying 5400
        buy_spread  = ask_hi - bid_lo  # executable spread when buying  5300 / selling 5400

        spread_pos     = state.position.get(p_hi, 0) - state.position.get(p_lo, 0)
        orders_hi      = result.setdefault(p_hi, [])
        orders_lo      = result.setdefault(p_lo, [])

        if sell_spread >= self.SPREAD_HIGH_SELL and spread_pos > -self.SPREAD_POSITION_CAP:
            # Sell rich 5300, buy cheap 5400 as hedge
            available_hi = od_hi.buy_orders.get(bid_hi, 0)
            available_lo = -od_lo.sell_orders.get(ask_lo, 0)
            room         = self.SPREAD_POSITION_CAP + spread_pos
            buy_cap_lo   = self._remaining_capacity(p_lo, state, orders_lo)[0]
            sell_cap_hi  = self._remaining_capacity(p_hi, state, orders_hi)[1]
            qty = min(self.SPREAD_PAIR_MAX_TAKE, available_hi, available_lo,
                      room, sell_cap_hi, buy_cap_lo)
            if qty > 0:
                orders_hi.append(Order(p_hi, bid_hi, -qty))
                orders_lo.append(Order(p_lo, ask_lo,  qty))

        elif buy_spread <= self.SPREAD_LOW_BUY and spread_pos < self.SPREAD_POSITION_CAP:
            # Buy cheap 5300, sell rich 5400 as hedge
            available_hi = -od_hi.sell_orders.get(ask_hi, 0)
            available_lo = od_lo.buy_orders.get(bid_lo, 0)
            room         = self.SPREAD_POSITION_CAP - spread_pos
            buy_cap_hi   = self._remaining_capacity(p_hi, state, orders_hi)[0]
            sell_cap_lo  = self._remaining_capacity(p_lo, state, orders_lo)[1]
            qty = min(self.SPREAD_PAIR_MAX_TAKE, available_hi, available_lo,
                      room, buy_cap_hi, sell_cap_lo)
            if qty > 0:
                orders_hi.append(Order(p_hi, ask_hi,  qty))
                orders_lo.append(Order(p_lo, bid_lo, -qty))

    # ── Spot fair value: volume-weighted mid, then EMA ─────────────────────────
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
            if bid_vol + ask_vol > 0:
                # Volume-weighted: skews toward the side with more depth
                raw_fair = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
            else:
                raw_fair = (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            raw_fair = float(best_bid)
        elif best_ask is not None:
            raw_fair = float(best_ask)

        if raw_fair is None:
            prev = td.get(product + "_ema")
            return float(prev) if prev is not None else None

        alpha    = self.EMA_ALPHA.get(product, 0.10)
        prev     = td.get(product + "_ema")
        ema      = raw_fair if prev is None else alpha * raw_fair + (1.0 - alpha) * float(prev)
        td[product + "_ema"] = ema
        return ema

    # ── Take mispriced liquidity (buy if ask < fair - edge; sell if bid > fair + edge) ─
    def _take_mispriced(
        self, product: str, od: OrderDepth, fair: float,
        state: TradingState, orders: List[Order]
    ) -> None:
        buy_cap, sell_cap = self._remaining_capacity(product, state, orders)
        edge     = self.TAKE_EDGE.get(product, 999_999.0)
        max_take = self.MAX_TAKE_SIZE.get(product, 0)
        if max_take == 0:
            return

        bought = 0
        for ask_price in sorted(od.sell_orders):
            if ask_price >= fair - edge or buy_cap <= 0 or bought >= max_take:
                break
            qty = min(-od.sell_orders[ask_price], buy_cap, max_take - bought)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_cap -= qty
                bought  += qty

        sold = 0
        for bid_price in sorted(od.buy_orders, reverse=True):
            if bid_price <= fair + edge or sell_cap <= 0 or sold >= max_take:
                break
            qty = min(od.buy_orders[bid_price], sell_cap, max_take - sold)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_cap -= qty
                sold     += qty

    # ── Passive quoting for spot products ─────────────────────────────────────
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

        # Push inside the NPC book only when it meaningfully improves our position
        if best_bid + 1 < fair - 1:
            buy_px = max(buy_px, best_bid + 1)
        if best_ask - 1 > fair + 1:
            sell_px = min(sell_px, best_ask - 1)

        # Prevent crossing the book
        if buy_px  >= best_ask: buy_px  = best_ask - 1
        if sell_px <= best_bid: sell_px = best_bid + 1
        buy_px  = max(0, buy_px)
        sell_px = max(1, sell_px)

        # Scale size with how far we are from limit (inventory management)
        pos   = state.position.get(product, 0)
        limit = self.LIMITS[product]
        buy_scale  = max(0.0, 1.0 - max( pos, 0) / limit)
        sell_scale = max(0.0, 1.0 - max(-pos, 0) / limit)

        buy_qty  = min(buy_cap,  max(1, int(max_make * buy_scale)))
        sell_qty = min(sell_cap, max(1, int(max_make * sell_scale)))

        if buy_qty  > 0: orders.append(Order(product, buy_px,   buy_qty))
        if sell_qty > 0: orders.append(Order(product, sell_px, -sell_qty))

    # ── Remaining order capacity (respects soft limits where set) ─────────────
    def _remaining_capacity(
        self, product: str, state: TradingState, existing: List[Order]
    ) -> Tuple[int, int]:
        limit    = self.SOFT_LIMITS.get(product, self.LIMITS.get(product, 200))
        position = state.position.get(product, 0)
        pending_buy = pending_sell = 0
        for o in existing:
            if o.quantity > 0: pending_buy  += o.quantity
            else:              pending_sell += -o.quantity
        buy_cap  = limit - position - pending_buy
        sell_cap = limit + position - pending_sell
        return max(0, buy_cap), max(0, sell_cap)

    # ── Best bid / ask helpers ─────────────────────────────────────────────────
    def _best_bid_ask(
        self, od: OrderDepth
    ) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(od.buy_orders)  if od.buy_orders  else None
        best_ask = min(od.sell_orders) if od.sell_orders else None
        return best_bid, best_ask
