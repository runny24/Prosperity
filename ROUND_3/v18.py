"""
Round 3 v18 — VEV_4000/4500 dynamic extreme thresholds
=========================================================
v17 test PnL: 44,264 (HYDROGEL 18,392 · VELVETFRUIT 7,535 · VEV_5000 5,825 ·
              VEV_5300 3,415 · VEV_4500 2,646 · VEV_4000 2,144 · VEV_5400 1,898
              VEV_5100 1,083 · VEV_5200 883 · VEV_5500 445)

Change from v17: VEV_4000 / VEV_4500 extreme thresholds made dynamic.

Problem with v17 fixed thresholds for VEV_4000/4500
────────────────────────────────────────────────────
VEV_4000/4500 have delta≈1 — fair moves 1:1 with VELVETFRUIT_EXTRACT.
v17 hardcoded EXTREME_SELL[VEV_4000]=1269, calibrated to VEV≈5262 (fair≈1264).
If VEV drifts +50 pts on competition day (5262→5312):
  • option fair rises to 1314
  • threshold stays at 1269
  • extreme SELL fires every tick (bid≈1312 ≥ 1269), selling ~12 pts below fair
  • cap of 200 → up to ~-2,500 loss per product (~-5,000 total)
Stress test confirmed: fires 1000/1000 ticks in +50pt scenario.

Fix: compute thresholds at runtime as fair_ema ± fixed_offset.
  sell_thresh = max(velvet_ema - strike, 0) + tv + sell_offset
  buy_thresh  = max(velvet_ema - strike, 0) + tv - buy_offset
At VEV=5262 (calibration) these reproduce the exact v17 values:
  VEV_4000: sell = 1264+5 = 1269 ✓   buy = 1264-9 = 1255 ✓
  VEV_4500: sell =  765+4 =  769 ✓   buy =  765-10 =  755 ✓
At any VEV level: trigger rate stays ~20%, no below-fair accumulation.

Key design decisions (unchanged from v17)
──────────────────────────────────────────
• VEV_5100/5200: disabled for TV-take, extreme-only with cap=30.
• VEV_5000: asymmetric take edges BUY=6, SELL=3.
• TV-take / extreme conflict resolved via DISABLED_VOUCHERS.
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
    # VEV_5100/5200 notes: thresholds are only ±5-10 pts from fair value.
    #   A ±20pt VEV shift would cause near-100% trigger rate on these.
    #   Capped at 30 to limit worst-case downside to ~300 PnL per product.
    #   (50k.py used cap=200 and earned +7.8k/+6.4k, but was on competition
    #    day data where VEV happened to be at the calibrated level.)
    # VEV_4000/4500 use DYNAMIC thresholds (see ITM_VOUCHER_CONFIG below).
    # All other products use fixed absolute thresholds.
    EXTREME_SELL: Dict[str, int] = {
        "HYDROGEL_PACK":       10_012,   # C: was 10,040 (never fired) → 10% trigger
        "VELVETFRUIT_EXTRACT":  5_269,   # 50k calibration: ~10% trigger
        # VEV_4000/4500: computed dynamically from velvet_ema (see ITM_VOUCHER_CONFIG)
        "VEV_5000":               272,   # 50k calibration: ~7.9% (was 278 → 1%)
        "VEV_5100":               180,   # fragile — capped at 30
        "VEV_5200":               106,   # fragile — capped at 30
        "VEV_5300":                53,   # 50k calibration: ~12.6% (was 56 → 3%)
        "VEV_5400":                18,   # 50k calibration: ~6.9%  (was 21 → 0%)
    }

    EXTREME_BUY: Dict[str, int] = {
        "HYDROGEL_PACK":        9_948,   # C: was 9,940 → 10% trigger (was trivial)
        "VELVETFRUIT_EXTRACT":  5_255,   # 50k calibration: ~12.7% (was 5226 → 0%)
        # VEV_4000/4500: computed dynamically from velvet_ema (see ITM_VOUCHER_CONFIG)
        "VEV_5000":               258,   # D+50k: 15 pts below fair ~267; ~9.4% trigger
        "VEV_5100":               165,   # 50k (tightened): trough bottoms at 157-165
        "VEV_5200":                93,   # 50k (tightened): trough bottoms at 87-91
        "VEV_5300":                47,   # 50k calibration: ~16.6% (was 36 → 0%)
        "VEV_5400":                14,   # 50k calibration: ~8.9%  (was 11 → 0%)
        "VEV_5500":                 5,   # unchanged
    }

    # Dynamic threshold config for deeply-ITM vouchers (delta≈1).
    # sell_thresh = max(velvet_ema - strike, 0) + tv + sell_offset
    # buy_thresh  = max(velvet_ema - strike, 0) + tv - buy_offset
    # At VEV=5262 these reproduce the exact same values as v17's fixed thresholds.
    ITM_VOUCHER_CONFIG: Dict[str, Dict] = {
        "VEV_4000": {"strike": 4000, "tv": 2, "sell_offset": 5, "buy_offset": 9},
        "VEV_4500": {"strike": 4500, "tv": 3, "sell_offset": 4, "buy_offset": 10},
    }

    # Per-product position caps for the extreme module.
    # VEV_5100/5200 deliberately small (see note above).
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
    }

    EXTREME_MAX_TAKE = 60   # max units per tick from extreme module

    # ── VEV_5300 / VEV_5400 spread overlay ───────────────────────────────────
    # 50k loosening: SPREAD_HIGH_SELL 36.5→35.5, SPREAD_POSITION_CAP 120→150.
    SPREAD_PAIR_MAX_TAKE = 30
    SPREAD_POSITION_CAP  = 150   # was 120 in 445307
    SPREAD_HIGH_SELL     = 35.5  # was 36.5 in 445307
    SPREAD_LOW_BUY       = 24.0

    # ── VEV_5500 strict rule ──────────────────────────────────────────────────
    VEV_5500_BUY_MAX  = 5
    VEV_5500_SELL_MIN = 8

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
                    # Pass velvet_fair so ITM vouchers can use dynamic thresholds
                    self._trade_extreme(product, state, orders[product], velvet_fair)

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
        self, product: str, state: TradingState, orders: List[Order],
        velvet_fair: Optional[float] = None
    ) -> None:
        """
        Fires absolute-price extreme orders independently of the TV-take module.
        Positions are bounded by EXTREME_CAPS (separate from SOFT_LIMITS).

        For deeply-ITM vouchers (VEV_4000/4500), thresholds are computed
        dynamically from velvet_ema so they scale correctly if VEV drifts.
        All other products use fixed absolute thresholds from EXTREME_SELL/BUY.
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

        # Resolve thresholds: dynamic for ITM vouchers, fixed for everything else
        itm_cfg = self.ITM_VOUCHER_CONFIG.get(product)
        if itm_cfg is not None and velvet_fair is not None:
            option_fair = max(velvet_fair - itm_cfg["strike"], 0.0) + itm_cfg["tv"]
            sell_thresh: Optional[float] = option_fair + itm_cfg["sell_offset"]
            buy_thresh:  Optional[float] = option_fair - itm_cfg["buy_offset"]
        else:
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
