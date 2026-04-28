"""
Round 3 v23 — VEV_6000/6500 conservative sell-only rule
=========================================================
v17 1k-window PnL: 44,264  |  v22 = current best (30k calibration + HYDROGEL SELL compromise)

Changes from v22:
  1. VEV_6000/VEV_6500 added to EXTREME_SELL at threshold=2 (sell if bid ≥ 2).
     Far-OTM options (strikes 6000/6500 vs velvetfruit ≈ 5248-5262); intrinsic = 0.
     Any bid ≥ 2 is essentially free money — the market is pricing in optionality
     that doesn't exist given TTE=5 and underlying ~250-300 pts below strike.
  2. VEV_6000/VEV_6500 added to EXTREME_CAPS at 50.
     Worst-case loss if velvetfruit spikes to 6100: short 50 × ~100pts = 5,000.
     No buy thresholds — never accumulate far-OTM calls.
  Products remain in DISABLED_VOUCHERS (TV-take still disabled).

v22 single change from v21: HYDROGEL_PACK EXTREME_SELL  10022 → 10015

Why this is the only useful compromise
────────────────────────────────────────
The 1k simulation used prices at the HIGH end of the 30k distribution
(velvetfruit mean 5262 vs 5248 broad). This created two categories of conflict:

RECOVERABLE (threshold satisfying 1k≥5% AND 30k≤20% exists):
  HYDROGEL SELL: 10022(0.5%/10.3%) → 10015(5.9%/15.2%)  ← THIS CHANGE
  All other adjustments were already at their optimal compromise in v21.

STRUCTURAL CONFLICT (no middle threshold works — any threshold firing ≥5%
on the 1k high-price window fires ≥50% on the broader 30k dataset):
  VELVETFRUIT BUY: v17's 5255 fired 12.7% on 1k but 58.0% on 30k.
    Any threshold that fires in the 1k high-price window buys at 30k normal prices.
    Keep v21=5233 (10.4% on 30k).
  VEV_5000 BUY: same issue (258 was above 30k mean). Keep v21=240.
  VEV_5300 BUY: 47 was right at 30k mean. Keep v21=39.
  VEV_4000/4500 BUY: even v17 fired only 2-3% on 1k — not impactful.

Expected impact of the HYDROGEL SELL change:
  1k window: HYDROGEL gain ~5,546→~12,000-13,000 (+6-7k recovery)
  30k:       HYDROGEL fires 15.2% vs v21's 10.3% → small, still extreme prices

v21 calibration summary (unchanged, all at ~10% on 30k):
  HYDROGEL BUY: 9948→9956  |  VELVETFRUIT BUY: 5255→5233
  VEV_4000: sell 1269→1261 buy 1255→1241  |  VEV_4500: sell 769→763 buy 755→738
  VEV_5000 BUY: 258→240  |  VEV_5100 BUY: 165→153  |  VEV_5200 BUY: 93→85
  VEV_5300 BUY: 47→39  |  VEV_5400: sell 18→19 buy 14→12
  VEV_5500 SELL_MIN=8 confirmed fires 18.4% on 30k (was 0% on 1k — quiet day)

Key design decisions (unchanged from v17)
──────────────────────────────────────────
• VEV_5100/5200: disabled for TV-take, extreme-only with cap=30.
• VEV_5000: asymmetric take edges BUY=6, SELL=3.
• TV-take / extreme conflict resolved via DISABLED_VOUCHERS.
• VEV_4000/4500: fixed absolute thresholds (dynamic v18 cost -4,789).
• No timestamp logic. No feature flags.
"""

import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)


# ── Logger (unchanged from 445307 / v16) ─────────────────────────────────────
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

    # ── Hard position limits ──────────────────────────────────────────────────
    LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK":       200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300, "VEV_4500": 300,
        "VEV_5000": 300, "VEV_5100": 300, "VEV_5200": 300,
        "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
        "VEV_6000": 300, "VEV_6500": 300,
    }

    # Soft limits — cap for the TV-take module (not the extreme module)
    SOFT_LIMITS: Dict[str, int] = {
        "VEV_5000": 150,   # TV-take + extreme both use this; extreme further bound by EXTREME_CAPS
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
        # VEV_4000/4500/5100/5200: in DISABLED_VOUCHERS (no TV-take), extreme module uses LIMITS
    }

    # Voucher strikes
    VOUCHER_STRIKES: Dict[str, int] = {
        "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    }

    # Empirically calibrated time values (TTE=5, Round 3)
    VOUCHER_TIME_VALUE: Dict[str, float] = {
        "VEV_5000":  5.0,
        "VEV_5100": 12.0,
        "VEV_5200": 38.5,
        "VEV_5300": 50.0,
        "VEV_5400": 16.0,
        "VEV_5500":  5.0,
    }

    # Products excluded from TV-take.
    # VEV_4000/4500: deeply ITM, delta≈1, time value≈0 → extreme module only.
    # VEV_5100/5200: TV-take fair ≈ extreme_sell threshold → modules conflict
    #                (simultaneous buy/sell on same tick), causing large losses.
    #                Handled by extreme module only, with conservative caps.
    # VEV_6000/6500: far OTM, no liquid market.
    DISABLED_VOUCHERS = {"VEV_4000", "VEV_4500", "VEV_5100", "VEV_5200",
                         "VEV_6000", "VEV_6500"}

    # ── TV-take edges (asymmetric for VEV_5000) ───────────────────────────────
    # VEV_5000 sell edge 6→3: opens 3× more sell opportunities for cycling.
    # VEV_5300 sell edge 3→2: +104 PnL confirmed in isolated test (445307 base).
    TAKE_EDGE_BUY: Dict[str, float] = {
        "HYDROGEL_PACK":       8.0,
        "VELVETFRUIT_EXTRACT": 2.0,
        "VEV_5000":            6.0,   # buy when ask < fair - 6 ≈ 261
        "VEV_5100":            3.0,   # not used (DISABLED), kept for reference
        "VEV_5200":            3.0,   # not used (DISABLED), kept for reference
        "VEV_5300":            3.0,
        "VEV_5400":            1.0,
        "VEV_5500":            0.5,
    }

    TAKE_EDGE_SELL: Dict[str, float] = {
        "HYDROGEL_PACK":       8.0,
        "VELVETFRUIT_EXTRACT": 2.0,
        "VEV_5000":            3.0,   # sell when bid > fair + 3 ≈ 270 (was 6 → 273)
        "VEV_5100":            3.0,   # not used (DISABLED)
        "VEV_5200":            3.0,   # not used (DISABLED)
        "VEV_5300":            2.0,   # ← baked-in improvement (+104 confirmed)
        "VEV_5400":            1.0,
        "VEV_5500":            0.5,
    }

    MAX_TAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK":       30,
        "VELVETFRUIT_EXTRACT": 50,
        "VEV_5000":            12,
        "VEV_5100":            25,   # not used (DISABLED)
        "VEV_5200":            30,   # not used (DISABLED)
        "VEV_5300":           100,
        "VEV_5400":           100,
        "VEV_5500":            80,
    }

    # More aggressive passive quoting (50k calibration)
    MAKE_EDGE: Dict[str, int] = {
        "HYDROGEL_PACK":        5,   # was 10 in v16 / 445307
        "VELVETFRUIT_EXTRACT":  2,   # was 3 in 445307
    }

    MAX_MAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK":       30,
        "VELVETFRUIT_EXTRACT": 50,
    }

    EMA_ALPHA: Dict[str, float] = {
        "HYDROGEL_PACK":       0.08,
        "VELVETFRUIT_EXTRACT": 0.12,
    }

    # ── Extreme-price module ──────────────────────────────────────────────────
    # Thresholds calibrated from test-data price distributions.
    # Trigger rates: HYDROGEL ~10%, VELVETFRUIT ~10-12%, most VEV options 7-17%.
    # NOT calibrated to specific timestamps — pure price-level mean-reversion.
    #
    # VEV_4000/4500: fixed absolute thresholds calibrated to VEV≈5262.
    #   delta≈1 means fair moves 1:1 with VELVETFRUIT. If VEV drifts ±50pts
    #   max drift loss is ~-2,500 per product (~-5,000 total) with cap=200.
    #   Expected value of dynamic-threshold protection: negative (v18 cost -4,789).
    # VEV_5100/5200 notes: thresholds are only ±5-10 pts from fair value.
    #   Capped at 30 to limit worst-case downside to ~300 PnL per product.
    EXTREME_SELL: Dict[str, int] = {
        "HYDROGEL_PACK":       10_015,   # v22 compromise: 5.9% on 1k, 15.2% on 30k (v21 was 10022→0.5% on 1k)
        "VELVETFRUIT_EXTRACT":  5_269,   # 30k confirmed: 10.3% ✓
        "VEV_4000":             1_261,   # 30k recal: 10.2% (was 1269 → 2.4%; VEV drifted lower)
        "VEV_4500":               763,   # 30k recal: 10.9% (was 769 → 3.9%)
        "VEV_5000":               272,   # 30k confirmed: 9.5% ✓
        "VEV_5100":               181,   # 30k recal: 11.1% (was 180 → 12.9%; minor)
        "VEV_5200":               106,   # 30k confirmed: 11.4% ✓
        "VEV_5300":                53,   # 30k confirmed: 12.4% ✓
        "VEV_5400":                19,   # 30k recal: 15.6% (was 18 → 23.9%)
        "VEV_6000":                 2,   # v23: sell-only; strike 6000 vs VEV≈5248 → intrinsic=0
        "VEV_6500":                 2,   # v23: sell-only; strike 6500 vs VEV≈5248 → intrinsic=0
    }

    EXTREME_BUY: Dict[str, int] = {
        "HYDROGEL_PACK":        9_956,   # 30k recal: 10.3% (was 9948 → 5.7%; too cold)
        "VELVETFRUIT_EXTRACT":  5_233,   # 30k recal: 10.4% (was 5255 → 58.0%; VEV drifted down)
        "VEV_4000":             1_241,   # 30k recal: 10.7% (was 1255 → 40.5%; buying at mean!)
        "VEV_4500":               738,   # 30k recal: 10.1% (was 755 → 46.4%; buying at mean!)
        "VEV_5000":               240,   # 30k recal: 10.3% (was 258 → 52.6%; fair = 248)
        "VEV_5100":               153,   # 30k recal: 10.7% (was 165 → 39.3%)
        "VEV_5200":                85,   # 30k recal: 11.6% (was 93 → 34.6%)
        "VEV_5300":                39,   # 30k recal: 10.3% (was 47 → 44.6%; buying at mean!)
        "VEV_5400":                12,   # 30k recal: 13.5% (was 14 → 25.9%)
        "VEV_5500":                 5,   # 30k confirmed: 19.6% ✓ (BUY_MAX enforces this)
    }

    # Per-product position caps for the extreme module.
    # VEV_5100/5200 deliberately small (see note above).
    # VEV_6000/6500 capped at 50: sell-only, limits worst-case spike exposure to ~5k.
    EXTREME_CAPS: Dict[str, int] = {
        "HYDROGEL_PACK":       180,
        "VELVETFRUIT_EXTRACT": 180,
        "VEV_4000":            200,
        "VEV_4500":            200,
        "VEV_5000":            200,
        "VEV_5100":             30,   # conservative: fragile thresholds
        "VEV_5200":             30,   # conservative: fragile thresholds
        "VEV_5300":            250,
        "VEV_5400":            250,
        "VEV_5500":            250,
        "VEV_6000":             50,   # v23: sell-only cap; max loss ≈5k if underlying spikes
        "VEV_6500":             50,   # v23: sell-only cap; max loss ≈5k if underlying spikes
    }

    EXTREME_MAX_TAKE = 60   # max units per tick from extreme module

    # ── VEV_5300 / VEV_5400 spread overlay ───────────────────────────────────
    # 50k loosening: SPREAD_HIGH_SELL 36.5→35.5, SPREAD_POSITION_CAP 120→150.
    SPREAD_PAIR_MAX_TAKE = 30
    SPREAD_POSITION_CAP  = 150   # was 120 in 445307
    SPREAD_HIGH_SELL     = 35.5  # was 36.5 in 445307
    SPREAD_LOW_BUY       = 24.0

    # ── VEV_5500 strict rule ──────────────────────────────────────────────────
    # Market data analysis (461960.log, 1000 ticks):
    #   bid distribution: min=4, p25=5, p50=6, p75=6, max=7
    #   SELL_MIN=8 → fires 0/1000 ticks (dead threshold, max bid only 7)
    #   SELL_MIN=6 → fires 663/1000 (66.3%) ← correct value
    VEV_5500_BUY_MAX  = 5
    VEV_5500_SELL_MIN = 8   # 30k confirmed: fires 18.4% (1k subset was unusually quiet, max bid=7)

    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        velvet_fair = self._spot_fair("VELVETFRUIT_EXTRACT", state, td)

        # ── HYDROGEL ──────────────────────────────────────────────────────────
        if "HYDROGEL_PACK" in state.order_depths:
            hyd_fair = self._spot_fair("HYDROGEL_PACK", state, td)
            orders["HYDROGEL_PACK"] = self._trade_spot(
                "HYDROGEL_PACK", state, td, known_fair=hyd_fair)
            self._trade_extreme("HYDROGEL_PACK", state, orders["HYDROGEL_PACK"])

        # ── VELVETFRUIT ───────────────────────────────────────────────────────
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            orders["VELVETFRUIT_EXTRACT"] = self._trade_spot(
                "VELVETFRUIT_EXTRACT", state, td, known_fair=velvet_fair)
            self._trade_extreme(
                "VELVETFRUIT_EXTRACT", state, orders["VELVETFRUIT_EXTRACT"])

        # ── Vouchers ──────────────────────────────────────────────────────────
        if velvet_fair is not None:
            for product in state.order_depths:
                if product.startswith("VEV_"):
                    orders[product] = self._trade_voucher(
                        product, velvet_fair, state)
                    self._trade_extreme(product, state, orders[product])

            self._trade_5300_5400_spread(state, orders)

        tdo = json.dumps(td, separators=(",", ":"))
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ── Spot: EMA fair + TV-take + passive quotes ─────────────────────────────
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

    # ── Voucher: intrinsic + fixed time value → TV-take ──────────────────────
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

    # ── Extreme-price module ──────────────────────────────────────────────────
    def _trade_extreme(
        self, product: str, state: TradingState, orders: List[Order]
    ) -> None:
        """
        Fires absolute-price extreme orders independently of the TV-take module.
        Positions are bounded by EXTREME_CAPS (separate from SOFT_LIMITS).
        All products use fixed absolute thresholds from EXTREME_SELL/BUY.
        """
        cap = self.EXTREME_CAPS.get(product)
        if cap is None or product not in state.order_depths:
            return

        od = state.order_depths[product]
        best_bid, best_ask = self._best_bid_ask(od)
        pos = state.position.get(product, 0)

        pending_buy  = sum( o.quantity for o in orders if o.quantity > 0)
        pending_sell = sum(-o.quantity for o in orders if o.quantity < 0)

        # Hard-cap remaining buys / sells from extreme module
        extreme_buy_room  = max(0, cap - pos - pending_buy)
        extreme_sell_room = max(0, cap + pos - pending_sell)

        sell_thresh = self.EXTREME_SELL.get(product)
        buy_thresh  = self.EXTREME_BUY.get(product)

        # Sell at extreme high
        if (sell_thresh is not None and best_bid is not None
                and best_bid >= sell_thresh and extreme_sell_room > 0):
            qty = min(extreme_sell_room, self.EXTREME_MAX_TAKE,
                      od.buy_orders.get(best_bid, 0))
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))

        # Buy at extreme low
        if (buy_thresh is not None and best_ask is not None
                and best_ask <= buy_thresh and extreme_buy_room > 0):
            qty = min(extreme_buy_room, self.EXTREME_MAX_TAKE,
                      -od.sell_orders.get(best_ask, 0))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))

    # ── VEV_5500 strict rule ──────────────────────────────────────────────────
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

    # ── VEV_5300 / VEV_5400 spread overlay ───────────────────────────────────
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

    # ── EMA fair value (volume-weighted mid) ──────────────────────────────────
    def _spot_fair(
        self, product: str, state: TradingState, td: Dict
    ) -> Optional[float]:
        if product not in state.order_depths:
            return None
        od = state.order_depths[product]
        best_bid, best_ask = self._best_bid_ask(od)

        raw: Optional[float] = None
        if best_bid is not None and best_ask is not None:
            bid_vol = max(od.buy_orders.get(best_bid, 0), 0)
            ask_vol = max(-od.sell_orders.get(best_ask, 0), 0)
            raw = (
                (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
                if bid_vol + ask_vol > 0
                else (best_bid + best_ask) / 2.0
            )
        elif best_bid is not None:
            raw = float(best_bid)
        elif best_ask is not None:
            raw = float(best_ask)

        if raw is None:
            prev = td.get(product + "_ema")
            return float(prev) if prev is not None else None

        alpha = self.EMA_ALPHA.get(product, 0.10)
        prev  = td.get(product + "_ema")
        ema   = raw if prev is None else alpha * raw + (1.0 - alpha) * float(prev)
        td[product + "_ema"] = ema
        return ema

    # ── Asymmetric take: buy cheap, sell rich ─────────────────────────────────
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

    # ── Passive quoting (spots only) ─────────────────────────────────────────
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

    # ── Remaining capacity (uses SOFT_LIMITS, falls back to LIMITS) ───────────
    def _remaining_capacity(
        self, product: str, state: TradingState, existing: List[Order]
    ) -> Tuple[int, int]:
        limit    = self.SOFT_LIMITS.get(product, self.LIMITS.get(product, 200))
        position = state.position.get(product, 0)
        pending_buy  = sum( o.quantity for o in existing if o.quantity > 0)
        pending_sell = sum(-o.quantity for o in existing if o.quantity < 0)
        return (max(0, limit - position - pending_buy),
                max(0, limit + position - pending_sell))

    # ── Best bid / ask ────────────────────────────────────────────────────────
    def _best_bid_ask(
        self, od: OrderDepth
    ) -> Tuple[Optional[int], Optional[int]]:
        return (
            max(od.buy_orders)  if od.buy_orders  else None,
            min(od.sell_orders) if od.sell_orders else None,
        )
