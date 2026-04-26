"""
Round 3 v16 — Four-strategy testbed on top of 445307 baseline (~13,459)
========================================================================
Base: teammate's 445307.py (~13,459 platform PnL) — the current best robust
submission. One confirmed improvement from our v15 is baked in as baseline:
  • VEV_5300 sell edge: 3.0 → 2.0  (sell when bid > 52 instead of > 53)
    → +104 PnL confirmed in isolated test (438898.log)
    445307 had this at 3.0 (symmetric), so this is a net improvement.

Baseline (all flags False) ≈ ~13,500–13,600 (445307 + LOWER_5300_SELL_EDGE).

Four independent strategies — toggle ONE at a time first, then combine winners.
Recommended test order: A → B → A+B → C → D → combine all winners.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY A — FEATURE_DEEP_ITM_VOUCHERS                         [expected +3–5k]
  Activate VEV_4000 and VEV_4500.
  These are deeply ITM (strike 4000/4500, VEV ~5262) so they move ~1:1 with the
  underlying. Fair = max(VEV_ema - strike, 0) + tiny_time_value. Average spreads
  are 21 pts (VEV_4000) and 16 pts (VEV_4500) — wide enough for an 8-pt take
  edge to be profitable. Teammate's cross-day backtesting confirms both as
  consistent positives across all historical days.
  Parameters:
    VEV_4000: time_value=2, take_edge=8, soft_limit=120, max_take=60/tick
    VEV_4500: time_value=3, take_edge=8, soft_limit=150, max_take=60/tick

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY B — FEATURE_NEUTRALIZE_5100_5200                      [expected +0.5–1k]
  Disable VEV_5100 and VEV_5200 entirely.
  Teammate's all-days backtesting: VEV_5100 = -880, VEV_5200 = -3,412.
  On the platform test they contribute only +25 and +86, so cost on test day is
  near zero while cross-day risk is real. Cleanest fix: add to DISABLED_VOUCHERS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY C — FEATURE_HYDROGEL_EXTREME                          [expected +2–4k]
  Fix the HYDROGEL extreme-price module (broken in 445307).
  In 445307 EXTREME_SELL = 10,040 — max observed bid was 10,023, so it NEVER
  fired. EXTREME_BUY = 9,940 fired 55 times but contributes nothing because the
  EMA market-making already captures that range.
  New calibration (price-level, no timestamps):
    EXTREME_SELL = 10,012: 20 ticks in test data had bid ≥ 10,019; setting at
      10,012 captures that early elevated period without any timestamp window.
    EXTREME_BUY  =  9,952: ~1 std-dev below sim mean of 9,979 (~29 std).
    Cap: 80 units, max 60/tick, 5-tick warmup to skip stale opening book.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY D — FEATURE_VEV5000_TUNING                            [expected +1–3k]
  VEV_5000 earns ~0 in 445307 despite 156 buy-eligible ticks (ask < 261).
  The bottleneck is the sell side: only 22 ticks with bid > 273 (edge=6.0).
  Two changes:
  1. Lower VEV_5000 sell edge: 6.0 → 3.0 (sell when bid > 270, opens 3–4×
     more sell opportunities, enabling proper cycling).
  2. Add extreme buy: aggressively buy when ask ≤ 252 (15 pts below fair ~267).
     Normal take already captures ask < 261. This adds extra size at the
     deepest discount ticks (observed min ask = 248). Separate 80-unit cap.
"""

import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState
)

# ── EXPERIMENT FLAGS ──────────────────────────────────────────────────────────
# Baseline (all False) ≈ ~13,500  (445307 + LOWER_5300_SELL_EDGE baked in)
# Test ONE at a time. Recommended order: A → B → A+B → C → D → combine winners.

FEATURE_DEEP_ITM_VOUCHERS    = False  # Strategy A: activate VEV_4000 + VEV_4500
FEATURE_NEUTRALIZE_5100_5200 = False  # Strategy B: disable VEV_5100 + VEV_5200
FEATURE_HYDROGEL_EXTREME     = True  # Strategy C: fix HYDROGEL extreme thresholds
FEATURE_VEV5000_TUNING       = True  # Strategy D: lower VEV_5000 sell edge + extreme buy
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

    # ── Hard position limits ───────────────────────────────────────────────────
    LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK":       200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300, "VEV_4500": 300,
        "VEV_5000": 300, "VEV_5100": 300, "VEV_5200": 300,
        "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
        "VEV_6000": 300, "VEV_6500": 300,
    }

    # Soft limits — caps for the fair-value take module
    # Strategy A adds VEV_4000/4500 entries when enabled
    SOFT_LIMITS: Dict[str, int] = {
        **( {"VEV_4000": 120, "VEV_4500": 150} if FEATURE_DEEP_ITM_VOUCHERS else {} ),
        "VEV_5000": 80,
        "VEV_5100": 30,
        "VEV_5200": 50,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
    }

    # Strikes for all vouchers (declared for all; DISABLED_VOUCHERS gates usage)
    VOUCHER_STRIKES: Dict[str, int] = {
        "VEV_4000": 4000, "VEV_4500": 4500,
        "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    }

    # Time values — calibrated for TTE=5 (Round 3) from market observations
    # VEV_4000/4500: deeply ITM, delta~1, almost no optionality → TV ≈ 0
    VOUCHER_TIME_VALUE: Dict[str, float] = {
        "VEV_4000":  2.0,
        "VEV_4500":  3.0,
        "VEV_5000":  5.0,
        "VEV_5100":  8.0,
        "VEV_5200": 35.0,
        "VEV_5300": 50.0,
        "VEV_5400": 16.0,
        "VEV_5500":  5.0,
    }

    # Disabled set — computed from flags at class definition time
    DISABLED_VOUCHERS = (
        {"VEV_6000", "VEV_6500"}
        | (set()                       if FEATURE_DEEP_ITM_VOUCHERS    else {"VEV_4000", "VEV_4500"})
        | ({"VEV_5100", "VEV_5200"}    if FEATURE_NEUTRALIZE_5100_5200 else set())
    )

    # ── Extreme price module (445307-style, with corrected thresholds) ─────────
    # Strategy C replaces 445307's broken thresholds with calibrated values.
    # All other products keep 445307's original thresholds (unchanged from base).
    EXTREME_BUY: Dict[str, int] = {
        "VELVETFRUIT_EXTRACT": 5226,   # 445307 original (never fired; kept for compat)
        "VEV_4000":            1226,   # 445307 original
        "VEV_4500":             726,   # 445307 original
        "VEV_5000":             233,   # 445307 original
        "VEV_5300":              36,   # 445307 original
        "VEV_5400":              11,   # 445307 original
        "VEV_5500":               4,   # 445307 original
        # HYDROGEL: injected by Strategy C; not in this dict when flag is False
        **( {"HYDROGEL_PACK": 9_952} if FEATURE_HYDROGEL_EXTREME else
            {"HYDROGEL_PACK": 9_940} ),   # 9940 = 445307 original (fires but trivial)
    }

    EXTREME_SELL: Dict[str, int] = {
        "VELVETFRUIT_EXTRACT": 5275,   # 445307 original
        "VEV_4000":            1275,   # 445307 original
        "VEV_4500":             775,   # 445307 original
        "VEV_5000":             278,   # 445307 original
        "VEV_5300":              56,   # 445307 original
        "VEV_5400":              21,   # 445307 original
        "VEV_5500":               9,   # 445307 original
        # HYDROGEL: 10,040 in 445307 (never fired); Strategy C lowers to 10,012
        **( {"HYDROGEL_PACK": 10_012} if FEATURE_HYDROGEL_EXTREME else
            {"HYDROGEL_PACK": 10_040} ),
    }

    # Per-product position caps for the extreme module (from 445307)
    EXTREME_CAPS: Dict[str, int] = {
        "HYDROGEL_PACK":        80,    # 445307 had 120; reduced to 80 (conservative)
        "VELVETFRUIT_EXTRACT": 120,
        "VEV_4000":            120,
        "VEV_4500":            150,
        "VEV_5000":            180,
        "VEV_5300":            220,
        "VEV_5400":            220,
        "VEV_5500":            220,
    }

    EXTREME_MAX_TAKE = 60    # max units per tick for the extreme module
    EXTREME_WARMUP   =  5    # ticks to skip at startup (avoid stale opening book)

    # ── Fair-value take edges (445307 base, with v15 improvement baked in) ────
    # Baked-in improvement: VEV_5300 sell edge 3.0 → 2.0  (+104 confirmed).
    # Strategy D: VEV_5000 sell edge 6.0 → 3.0 when enabled.
    TAKE_EDGE_BUY: Dict[str, float] = {
        "HYDROGEL_PACK":       8.0,
        "VELVETFRUIT_EXTRACT": 2.0,
        "VEV_4000":            8.0,   # half the 21-pt avg spread
        "VEV_4500":            8.0,   # half the 16-pt avg spread
        "VEV_5000":            6.0,
        "VEV_5100":            3.0,
        "VEV_5200":            3.0,
        "VEV_5300":            3.0,
        "VEV_5400":            1.0,
        "VEV_5500":            0.5,
    }

    TAKE_EDGE_SELL: Dict[str, float] = {
        "HYDROGEL_PACK":       8.0,
        "VELVETFRUIT_EXTRACT": 2.0,
        "VEV_4000":            8.0,
        "VEV_4500":            8.0,
        "VEV_5000": 3.0 if FEATURE_VEV5000_TUNING else 6.0,
        "VEV_5100":            3.0,
        "VEV_5200":            3.0,
        "VEV_5300":            2.0,   # ← baked-in improvement from v15 (+104)
        "VEV_5400":            1.0,
        "VEV_5500":            0.5,
    }

    MAX_TAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK":       24,
        "VELVETFRUIT_EXTRACT": 50,
        "VEV_4000":            60,
        "VEV_4500":            60,
        "VEV_5000":            12,
        "VEV_5100":             8,
        "VEV_5200":            10,
        "VEV_5300":           100,
        "VEV_5400":           100,
        "VEV_5500":            80,
    }

    MAKE_EDGE: Dict[str, int] = {
        "HYDROGEL_PACK":       10,
        "VELVETFRUIT_EXTRACT":  3,
    }

    # 445307 uses 18 for HYDROGEL (vs v15's 24) — kept from 445307
    MAX_MAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK":       18,
        "VELVETFRUIT_EXTRACT": 50,
    }

    EMA_ALPHA: Dict[str, float] = {
        "HYDROGEL_PACK":       0.08,
        "VELVETFRUIT_EXTRACT": 0.12,
    }

    # ── Spread overlay (VEV_5300 / VEV_5400) — unchanged from 445307 ──────────
    SPREAD_PAIR_MAX_TAKE = 30
    SPREAD_POSITION_CAP  = 120
    SPREAD_HIGH_SELL     = 36.5
    SPREAD_LOW_BUY       = 24.0

    # ── VEV_5500 strict rule ───────────────────────────────────────────────────
    VEV_5500_BUY_MAX  = 5
    VEV_5500_SELL_MIN = 8

    # ── Strategy D: VEV_5000 extreme-buy threshold ────────────────────────────
    VEV5000_EXTREME_BUY_THRESHOLD = 252   # 15 pts below fair ~267; min ask in test = 248
    VEV5000_EXTREME_CAP           =  80
    VEV5000_EXTREME_MAX_TAKE      =  30

    # ──────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        velvet_fair = self._spot_fair("VELVETFRUIT_EXTRACT", state, td)

        # ── HYDROGEL ───────────────────────────────────────────────────────────
        if "HYDROGEL_PACK" in state.order_depths:
            hydrogel_fair = self._spot_fair("HYDROGEL_PACK", state, td)
            orders["HYDROGEL_PACK"] = self._trade_spot(
                "HYDROGEL_PACK", state, td, known_fair=hydrogel_fair
            )
            self._trade_extreme("HYDROGEL_PACK", state, td, orders["HYDROGEL_PACK"])

        # ── VELVETFRUIT ────────────────────────────────────────────────────────
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            orders["VELVETFRUIT_EXTRACT"] = self._trade_spot(
                "VELVETFRUIT_EXTRACT", state, td, known_fair=velvet_fair
            )
            self._trade_extreme(
                "VELVETFRUIT_EXTRACT", state, td, orders["VELVETFRUIT_EXTRACT"]
            )

        # ── Vouchers ───────────────────────────────────────────────────────────
        if velvet_fair is not None:
            for product in state.order_depths:
                if product.startswith("VEV_"):
                    orders[product] = self._trade_voucher(product, velvet_fair, state)
                    self._trade_extreme(product, state, td, orders[product])

            # Strategy D: VEV_5000 additional extreme buy
            if FEATURE_VEV5000_TUNING and "VEV_5000" in state.order_depths:
                self._trade_vev5000_extreme(state, orders["VEV_5000"])

            self._trade_5300_5400_spread(state, orders)

        tdo = json.dumps(td, separators=(",", ":"))
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ── Spot: EMA fair + take + passive quotes ─────────────────────────────────
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

    # ── Extreme price module (445307-style with corrected thresholds) ──────────
    def _trade_extreme(
        self, product: str, state: TradingState,
        td: Dict, orders: List[Order]
    ) -> None:
        """
        Fires absolute-price extreme orders independently of fair-value module.
        Skips the first EXTREME_WARMUP ticks per product to avoid stale books.
        Uses EXTREME_CAPS as position cap (separate from SOFT_LIMITS).
        Strategy C corrects HYDROGEL thresholds; all other products unchanged
        from 445307 originals.
        """
        if product not in self.EXTREME_CAPS or product not in state.order_depths:
            return

        seen_key = product + "_xseen"
        seen = td.get(seen_key, 0) + 1
        td[seen_key] = seen
        if seen <= self.EXTREME_WARMUP:
            return

        od = state.order_depths[product]
        best_bid, best_ask = self._best_bid_ask(od)

        buy_cap, sell_cap = self._remaining_extreme_capacity(product, state, orders)

        # Sell at extreme high
        sell_thresh = self.EXTREME_SELL.get(product)
        if sell_thresh is not None and best_bid is not None and best_bid >= sell_thresh and sell_cap > 0:
            qty = min(sell_cap, self.EXTREME_MAX_TAKE, od.buy_orders.get(best_bid, 0))
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))

        # Buy at extreme low
        buy_thresh = self.EXTREME_BUY.get(product)
        if buy_thresh is not None and best_ask is not None and best_ask <= buy_thresh and buy_cap > 0:
            qty = min(buy_cap, self.EXTREME_MAX_TAKE, -od.sell_orders.get(best_ask, 0))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))

    # ── Strategy D: VEV_5000 deep-discount extreme buy ────────────────────────
    def _trade_vev5000_extreme(
        self, state: TradingState, orders: List[Order]
    ) -> None:
        """
        Fires when VEV_5000 ask ≤ 252 (15 pts below fair ~267, trough region).
        Normal take already captures ask < 261 (6-pt edge). This adds extra
        size at genuinely extreme discounts using a separate 80-unit cap.
        """
        od = state.order_depths.get("VEV_5000")
        if od is None:
            return
        best_ask = min(od.sell_orders) if od.sell_orders else None
        if best_ask is None or best_ask > self.VEV5000_EXTREME_BUY_THRESHOLD:
            return

        buy_cap, _ = self._remaining_extreme_capacity_with_cap(
            "VEV_5000", state, orders, self.VEV5000_EXTREME_CAP
        )
        qty = min(buy_cap, self.VEV5000_EXTREME_MAX_TAKE, -od.sell_orders.get(best_ask, 0))
        if qty > 0:
            orders.append(Order("VEV_5000", best_ask, qty))

    # ── VEV_5500 strict low-price rule ────────────────────────────────────────
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

    # ── Remaining capacity (respects SOFT_LIMITS) ─────────────────────────────
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

    # ── Remaining capacity for extreme module (uses EXTREME_CAPS) ─────────────
    def _remaining_extreme_capacity(
        self, product: str, state: TradingState, existing: List[Order]
    ) -> Tuple[int, int]:
        cap      = self.EXTREME_CAPS.get(product, self.LIMITS.get(product, 200))
        position = state.position.get(product, 0)
        pending_buy = pending_sell = 0
        for o in existing:
            if o.quantity > 0: pending_buy  += o.quantity
            else:              pending_sell += -o.quantity
        return max(0, cap - position - pending_buy), max(0, cap + position - pending_sell)

    # ── Remaining capacity with explicit cap (for Strategy D) ─────────────────
    def _remaining_extreme_capacity_with_cap(
        self, product: str, state: TradingState,
        existing: List[Order], cap: int
    ) -> Tuple[int, int]:
        position = state.position.get(product, 0)
        pending_buy = pending_sell = 0
        for o in existing:
            if o.quantity > 0: pending_buy  += o.quantity
            else:              pending_sell += -o.quantity
        return max(0, cap - position - pending_buy), max(0, cap + position - pending_sell)

    # ── Best bid / ask ─────────────────────────────────────────────────────────
    def _best_bid_ask(
        self, od: OrderDepth
    ) -> Tuple[Optional[int], Optional[int]]:
        return (
            max(od.buy_orders)  if od.buy_orders  else None,
            min(od.sell_orders) if od.sell_orders else None,
        )
