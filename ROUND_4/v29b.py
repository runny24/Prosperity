"""
Round 4 v29 — Recalibrate BUY thresholds to official p20; HYDROGEL SELL p90→p80; MAX_TAKE 60→100
==================================================================================================
Base: v28 (536,621 on R4 30k local backtest; 47,038 on official R4 day 3).

Changes from v28
────────────────
All changes correct the systematic local→official data mismatch. Official prices
are consistently higher than local data; BUY thresholds calibrated on local data
were firing almost never on official. These are calibration fixes, not overfitting.

1. HYDROGEL EXTREME_SELL: 10,050 → 10,037  (official p80; fires ~20% vs ~10%)
   HYDROGEL EXTREME_BUY:   9,956 → 10,026  (official p20; was firing almost never)
   HYDROGEL EXTREME_CAPS:    180 →    200   (allow full position limit)
   BUY=9,956 never triggered on official (mean=10,033). At 10,026 we capture
   genuine cheap ticks and complete the buy-low/sell-high cycle.
   v_over vs v28 gap: +5,412 driven by this fix.

2. VEV_4500 EXTREME_BUY:   738 →    759   (official p20; local ~21pts lower)
   VEV_5000 EXTREME_BUY:   240 →    256   (official p20; local ~16pts lower)
   VEV_5100 EXTREME_BUY:   153 →    164   (official p20; local ~11pts lower)
   VEV_5200 EXTREME_BUY:    85 →     89   (official p20; local ~4pts lower)
   Systematic mismatch: local prices lower → thresholds too low → rarely fire
   on official data. Raising to official p20 gives ~20% fire rate on official.
   v_over gaps: 5000=+4,995 | 5100=+4,830 | 5200=+2,473 | 4500=+1,267.

3. EXTREME_MAX_TAKE: 60 → 100  (moderate; still well below all position limits)
   More units per clean extreme signal, compounding the BUY threshold fix.

Retained from v28:
   VEV_5100/5200 caps=200  |  VEV_5500 BUY_MAX=4  |  SELL_MIN=7  |  Mark01  |  TV unchanged

Architecture (unchanged from v23)
───────────────────────────────────
• EMA fair value (alpha=0.08 HYDROGEL, 0.12 VELVETFRUIT)
• TV-take: buy when ask < fair − edge, sell when bid > fair + edge
• Extreme module: absolute-price mean-reversion, independent of TV
• VEV_5500 strict rule: buy ≤ BUY_MAX=4, sell ≥ SELL_MIN=7
• VEV_5300/5400 spread overlay
• DISABLED_VOUCHERS: VEV_4000/4500/5100/5200/6000/6500 (extreme-only)
"""

import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)


# ── Logger (unchanged from v23) ───────────────────────────────────────────────
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

    # Soft limits — cap for the TV-take module
    SOFT_LIMITS: Dict[str, int] = {
        "VEV_5000": 150,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
    }

    # Voucher strikes
    VOUCHER_STRIKES: Dict[str, int] = {
        "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    }

    # Time values: unchanged from R3_final (confirmed correct for R4 order book)
    # R4 order book mids: VEV_5300=46.9, VEV_5400=15.7, VEV_5000≈253 (TV≈6.3)
    VOUCHER_TIME_VALUE: Dict[str, float] = {
        "VEV_5000":  5.0,
        "VEV_5100": 12.0,
        "VEV_5200": 38.5,
        "VEV_5300": 50.0,
        "VEV_5400": 16.0,
        "VEV_5500":  5.0,
    }

    # Products excluded from TV-take (extreme module only)
    DISABLED_VOUCHERS = {"VEV_4000", "VEV_4500", "VEV_5100", "VEV_5200",
                         "VEV_6000", "VEV_6500"}

    # ── TV-take edges ─────────────────────────────────────────────────────────
    TAKE_EDGE_BUY: Dict[str, float] = {
        "HYDROGEL_PACK":       8.0,
        "VELVETFRUIT_EXTRACT": 2.0,
        "VEV_5000":            6.0,
        "VEV_5100":            3.0,   # disabled, kept for reference
        "VEV_5200":            3.0,   # disabled, kept for reference
        "VEV_5300":            3.0,
        "VEV_5400":            1.0,
        "VEV_5500":            0.5,
    }

    TAKE_EDGE_SELL: Dict[str, float] = {
        "HYDROGEL_PACK":       8.0,
        "VELVETFRUIT_EXTRACT": 2.0,
        "VEV_5000":            3.0,
        "VEV_5100":            3.0,   # disabled
        "VEV_5200":            3.0,   # disabled
        "VEV_5300":            2.0,
        "VEV_5400":            1.0,
        "VEV_5500":            0.5,
    }

    MAX_TAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK":       30,
        "VELVETFRUIT_EXTRACT": 50,
        "VEV_5000":            12,
        "VEV_5100":            25,   # disabled
        "VEV_5200":            30,   # disabled
        "VEV_5300":           100,
        "VEV_5400":           100,
        "VEV_5500":            80,
    }

    MAKE_EDGE: Dict[str, int] = {
        "HYDROGEL_PACK":        5,
        "VELVETFRUIT_EXTRACT":  2,
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
    EXTREME_SELL: Dict[str, int] = {
        "HYDROGEL_PACK":       10_037,   # v29: official p80 (was 10050≈p90; fires ~20% vs ~10%)

        "VELVETFRUIT_EXTRACT":  5_269,
        "VEV_4000":             1_261,
        "VEV_4500":               763,
        "VEV_5000":               272,
        "VEV_5100":               181,
        "VEV_5200":               106,
        "VEV_5300":                53,
        "VEV_5400":                19,
        "VEV_6000":                 2,   # sell-only; intrinsic=0 at VF≈5239
        "VEV_6500":                 2,   # sell-only
    }

    EXTREME_BUY: Dict[str, int] = {
        "HYDROGEL_PACK":       10_026,   # v29: official p20 (was 9956, never fired officially)
        "VELVETFRUIT_EXTRACT":  5_233,
        "VEV_4000":             1_241,
        "VEV_4500":               738,   # v29b: reverted from 759 (-34k local, only +1.3k official upside)
        "VEV_5000":               256,   # v29: official p20 (was 240, local data ~16pts lower)
        "VEV_5100":               164,   # v29: official p20 (was 153, local data ~11pts lower)
        "VEV_5200":                89,   # v29: official p20 (was 85, local data ~4pts lower)
        "VEV_5300":                39,
        "VEV_5400":                12,
        "VEV_5500":                 5,
    }

    EXTREME_CAPS: Dict[str, int] = {
        "HYDROGEL_PACK":       200,   # v29: raised from 180 (full position limit)
        "VELVETFRUIT_EXTRACT": 180,
        "VEV_4000":            200,
        "VEV_4500":            200,
        "VEV_5000":            200,
        "VEV_5100":            200,   # v28: raised from 150 (at limit, captures more cycles)
        "VEV_5200":            200,   # v28: raised from 150
        "VEV_5300":            250,
        "VEV_5400":            250,
        "VEV_5500":            250,
        "VEV_6000":             50,   # sell-only cap
        "VEV_6500":             50,   # sell-only cap
    }

    EXTREME_MAX_TAKE = 100   # v29: raised from 60 (moderate step-up)

    # ── VEV_5300 / VEV_5400 spread overlay ───────────────────────────────────
    SPREAD_PAIR_MAX_TAKE = 30
    SPREAD_POSITION_CAP  = 150
    SPREAD_HIGH_SELL     = 35.5
    SPREAD_LOW_BUY       = 24.0

    # ── VEV_5500 strict rule ──────────────────────────────────────────────────
    # SELL_MIN lowered 8 → 7 (confirmed by SLURM sweep on R4 30k data).
    # VEV_5500 PnL: +1,269 (SELL_MIN=8) → +2,560 (SELL_MIN=7), total +484,764.
    VEV_5500_BUY_MAX  = 4   # v27: lowered from 5 (official mean=4.5; ≤5 was firing 74% → ≤4 targets p10)
    VEV_5500_SELL_MIN = 7   # sweep-confirmed optimum (v28 sweep: 4=454k, 5=534k, 6=535.9k, 7=536.6k, 8=535.6k)

    # ── Mark 01 counterparty overlay ─────────────────────────────────────────
    # Mark 01 is a passive limit buyer of OTM VEVs (never sells).
    # When seen as buyer in market_trades, his bids were hit last tick
    # and are likely live again this tick.
    # We hit best_bid if it clears a conservative floor (avoids selling at noise).
    # Floor is set well below EXTREME_SELL to avoid double-counting.
    # MARK01_MAX_SELL caps our additional exposure per tick per product.
    MARK01_IDS = {"Mark 01"}
    MARK01_MAX_SELL = 20   # additional units per tick when signal fires
    # Floors set AT or near EXTREME_SELL thresholds so the overlay adds SIZE
    # at already-extreme prices rather than creating new sub-threshold shorts.
    # VEV_5500 is the exception: floor=6 expands one tick below SELL_MIN=7.
    MARK01_MIN_BID: Dict[str, int] = {
        "VEV_5300":  53,   # matches EXTREME_SELL=53; adds size at extreme bids only
        "VEV_5400":  19,   # matches EXTREME_SELL=19; same rationale
        "VEV_5500":   6,   # one below SELL_MIN=7; expands sell range when Mark 01 active
        "VEV_6000":   2,   # matches EXTREME_SELL=2 (sell-only product)
        "VEV_6500":   2,   # matches EXTREME_SELL=2
    }

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

        # ── Mark 01 overlay (runs after all other modules) ────────────────────
        self._trade_mark01_overlay(state, orders)

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
        cap = self.EXTREME_CAPS.get(product)
        if cap is None or product not in state.order_depths:
            return

        od = state.order_depths[product]
        best_bid, best_ask = self._best_bid_ask(od)
        pos = state.position.get(product, 0)

        pending_buy  = sum( o.quantity for o in orders if o.quantity > 0)
        pending_sell = sum(-o.quantity for o in orders if o.quantity < 0)

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

    # ── Mark 01 counterparty overlay ─────────────────────────────────────────
    def _mark01_active(self, product: str, state: TradingState) -> bool:
        """Returns True if Mark 01 appeared as a buyer last tick for this product."""
        return any(
            t.buyer in self.MARK01_IDS
            for t in state.market_trades.get(product, [])
        )

    def _trade_mark01_overlay(
        self, state: TradingState, orders: Dict[str, List[Order]]
    ) -> None:
        """
        When Mark 01 is an active passive buyer, hit his best bid at or above
        MARK01_MIN_BID[product]. This is additive to TV-take and extreme modules.
        Respects position limits via _remaining_capacity.
        """
        for product, floor in self.MARK01_MIN_BID.items():
            if not self._mark01_active(product, state):
                continue
            if product not in state.order_depths:
                continue

            od = state.order_depths[product]
            best_bid, _ = self._best_bid_ask(od)
            if best_bid is None or best_bid < floor:
                continue

            existing = orders.get(product, [])
            _, sell_cap = self._remaining_capacity(product, state, existing)
            avail = od.buy_orders.get(best_bid, 0)
            qty = min(sell_cap, self.MARK01_MAX_SELL, avail)

            if qty > 0:
                product_orders = orders.setdefault(product, [])
                product_orders.append(Order(product, best_bid, -qty))

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

    # ── EMA fair value ────────────────────────────────────────────────────────
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

    # ── Asymmetric take ───────────────────────────────────────────────────────
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

    # ── Passive quoting ───────────────────────────────────────────────────────
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

    # ── Remaining capacity ────────────────────────────────────────────────────
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
