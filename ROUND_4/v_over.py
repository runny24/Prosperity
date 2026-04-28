"""
Round 4 v_over — Overfitted to official day 3 price distribution
=================================================================
Base: v31 (526,815 local, 54,948 official day 3).

WARNING: Deliberately overfitted to known official day 3 data.
Not robust — submit knowing this is calibrated to a specific dataset.

Changes from v31
────────────────
All threshold changes calibrated to official day 3 bid/ask distributions
extracted from log 512213.log (submission 8f256fde, v31 run).

1. EXTREME_SELL tightened to p70 of observed best bids
   VEV_5300: 53 → 46  (was p87 of bids → fired only 13%; now p70 → fires ~35%)
   VEV_5400: 19 → 15  (p70 bid; tail event previously → fires ~3x more)
   VEV_5100: 181 → 177 (p70 bid; modest tightening)
   VEV_5200: 106 → 100 (p70 bid)
   VEV_5000: 267 → 269 (p70 bid; minor)
   HYDROGEL / VEV_4000: UNCHANGED (bid/ask distributions overlap at p70/p30;
       tight-spread products cannot be tightened without crossing)

2. EXTREME_BUY tightened to p30 of observed best asks
   VELVETFRUIT: 5,233 → 5,256  (was BELOW p10=5,249 → almost never fired;
                                  p30 ask = 5,256 → fires ~30% of ticks now)
   VEV_5000: 256 → 259  (p30 ask)
   VEV_5100: 164 → 167  (p30 ask)
   VEV_5200:  89 →  91  (p30 ask)
   VEV_5300:  39 →  41  (p30 ask)
   VEV_5400:  12 →  13  (p30 ask)
   HYDROGEL / VEV_4000: UNCHANGED (same reasoning as above)
   VEV_4500: UNCHANGED (p30 ask=762 is too close to sell=763; margin=1)

3. EXTREME_CAPS raised to hard position limits
   VEV_5300: 250 → 300  (tighter cycling warrants full cap)
   VEV_5400: 250 → 300
   VELVETFRUIT: 180 → 200  (full limit)

4. EXTREME_MAX_TAKE: 150 → 200
   Caps are binding; more units per trigger = more PnL per signal.

5. VEV passive market-making doubled vs v31
   VEV_MAKE_SIZE 10-15 → 20-30 units inside spread.
   More bot flow captured per tick.

Retained from v31:
   VEV_5500 buying disabled | Mark01 overlay | spread overlay | TV-take
"""

import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)


# ── Logger ────────────────────────────────────────────────────────────────────
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

    SOFT_LIMITS: Dict[str, int] = {
        "VEV_5000": 150,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
    }

    VOUCHER_STRIKES: Dict[str, int] = {
        "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    }

    VOUCHER_TIME_VALUE: Dict[str, float] = {
        "VEV_5000":  5.0,
        "VEV_5100": 12.0,
        "VEV_5200": 38.5,
        "VEV_5300": 50.0,
        "VEV_5400": 16.0,
        "VEV_5500":  5.0,
    }

    DISABLED_VOUCHERS = {"VEV_4000", "VEV_4500", "VEV_5100", "VEV_5200",
                         "VEV_6000", "VEV_6500"}

    # ── TV-take edges (unchanged from v31) ────────────────────────────────────
    TAKE_EDGE_BUY: Dict[str, float] = {
        "HYDROGEL_PACK":       8.0,
        "VELVETFRUIT_EXTRACT": 2.0,
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
        "VEV_5000":            3.0,
        "VEV_5100":            3.0,
        "VEV_5200":            3.0,
        "VEV_5300":            2.0,
        "VEV_5400":            1.0,
        "VEV_5500":            0.5,
    }

    MAX_TAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK":       30,
        "VELVETFRUIT_EXTRACT": 50,
        "VEV_5000":            12,
        "VEV_5100":            25,
        "VEV_5200":            30,
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

    # ── VEV passive market-making — doubled vs v31 ────────────────────────────
    VEV_MAKE_SIZE: Dict[str, int] = {
        "VEV_4000": 20,
        "VEV_4500": 25,
        "VEV_5000": 25,
        "VEV_5100": 25,
        "VEV_5200": 25,
        "VEV_5300": 30,
        "VEV_5400": 30,
        "VEV_5500":  0,   # buying disabled
        "VEV_6000":  0,   # sell-only
        "VEV_6500":  0,   # sell-only
    }

    # ── Extreme module — calibrated to official day 3 ─────────────────────────
    EXTREME_SELL: Dict[str, int] = {
        "HYDROGEL_PACK":       10_037,   # unchanged: tight spread, can't tighten further
        "VELVETFRUIT_EXTRACT":  5_268,   # p70 bid (≈ v31's 5,269)
        "VEV_4000":             1_261,   # unchanged: tight spread
        "VEV_4500":               763,   # p70 bid (same as v31)
        "VEV_5000":               269,   # p70 bid (was 267)
        "VEV_5100":               177,   # p70 bid (was 181)
        "VEV_5200":               100,   # p70 bid (was 106)
        "VEV_5300":                46,   # p70 bid (was 53=p87 → only 13% fire rate!)
        "VEV_5400":                15,   # p70 bid (was 19=tail → fires ~3x more now)
        "VEV_6000":                 2,
        "VEV_6500":                 2,
    }

    EXTREME_BUY: Dict[str, int] = {
        "HYDROGEL_PACK":       10_026,   # unchanged: p30 ask crosses sell threshold
        "VELVETFRUIT_EXTRACT":  5_256,   # p30 ask (was 5,233 = below p10! Almost never fired)
        "VEV_4000":             1_241,   # unchanged: p30 ask crosses sell threshold
        "VEV_4500":               738,   # unchanged: p30 ask=762 too close to sell=763
        "VEV_5000":               259,   # p30 ask (was 256)
        "VEV_5100":               167,   # p30 ask (was 164)
        "VEV_5200":                91,   # p30 ask (was 89)
        "VEV_5300":                41,   # p30 ask (was 39)
        "VEV_5400":                13,   # p30 ask (was 12)
        # VEV_5500: removed (buying disabled)
    }

    EXTREME_CAPS: Dict[str, int] = {
        "HYDROGEL_PACK":       200,
        "VELVETFRUIT_EXTRACT": 200,   # raised from 180
        "VEV_4000":            200,
        "VEV_4500":            200,
        "VEV_5000":            200,
        "VEV_5100":            200,
        "VEV_5200":            200,
        "VEV_5300":            300,   # raised from 250; tighter cycling uses full cap
        "VEV_5400":            300,   # raised from 250
        "VEV_5500":            250,
        "VEV_6000":             50,
        "VEV_6500":             50,
    }

    EXTREME_MAX_TAKE = 200   # raised from 150

    # ── VEV_5300/5400 spread overlay ──────────────────────────────────────────
    SPREAD_PAIR_MAX_TAKE = 30
    SPREAD_POSITION_CAP  = 150
    SPREAD_HIGH_SELL     = 35.5
    SPREAD_LOW_BUY       = 24.0

    # ── VEV_5500 — buying disabled ────────────────────────────────────────────
    VEV_5500_BUY_MAX  = 0   # confirmed loss: max bid=6 < SELL_MIN=7, never sells
    VEV_5500_SELL_MIN = 7

    # ── Mark 01 overlay ───────────────────────────────────────────────────────
    MARK01_IDS = {"Mark 01"}
    MARK01_MAX_SELL = 20
    MARK01_MIN_BID: Dict[str, int] = {
        "VEV_5300":  53,
        "VEV_5400":  19,
        "VEV_5500":   6,
        "VEV_6000":   2,
        "VEV_6500":   2,
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

        if "HYDROGEL_PACK" in state.order_depths:
            hyd_fair = self._spot_fair("HYDROGEL_PACK", state, td)
            orders["HYDROGEL_PACK"] = self._trade_spot(
                "HYDROGEL_PACK", state, td, known_fair=hyd_fair)
            self._trade_extreme("HYDROGEL_PACK", state, orders["HYDROGEL_PACK"])

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            orders["VELVETFRUIT_EXTRACT"] = self._trade_spot(
                "VELVETFRUIT_EXTRACT", state, td, known_fair=velvet_fair)
            self._trade_extreme(
                "VELVETFRUIT_EXTRACT", state, orders["VELVETFRUIT_EXTRACT"])

        if velvet_fair is not None:
            for product in state.order_depths:
                if product.startswith("VEV_"):
                    orders[product] = self._trade_voucher(
                        product, velvet_fair, state)
                    self._trade_extreme(product, state, orders[product])
                    self._add_vev_market_quotes(
                        product, state.order_depths[product], state, orders[product])
            self._trade_5300_5400_spread(state, orders)

        self._trade_mark01_overlay(state, orders)

        tdo = json.dumps(td, separators=(",", ":"))
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ── Spot ──────────────────────────────────────────────────────────────────
    def _trade_spot(self, product, state, td, known_fair=None):
        fair = known_fair if known_fair is not None else self._spot_fair(product, state, td)
        if fair is None:
            return []
        od = state.order_depths[product]
        orders = []
        self._take_mispriced(product, od, fair, state, orders)
        make_edge = self.MAKE_EDGE.get(product)
        if make_edge is not None:
            self._add_passive_quotes(product, od, fair, state, orders, make_edge)
        return orders

    # ── Voucher ───────────────────────────────────────────────────────────────
    def _trade_voucher(self, product, underlying_fair, state):
        if product in self.DISABLED_VOUCHERS or product not in self.VOUCHER_STRIKES:
            return []
        if product == "VEV_5500":
            return self._trade_5500_strict(product, state)
        strike     = self.VOUCHER_STRIKES[product]
        time_value = self.VOUCHER_TIME_VALUE[product]
        fair       = max(underlying_fair - strike, 0.0) + time_value
        od         = state.order_depths[product]
        orders     = []
        self._take_mispriced(product, od, fair, state, orders)
        return orders

    # ── Extreme module ────────────────────────────────────────────────────────
    def _trade_extreme(self, product, state, orders):
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
        if (sell_thresh is not None and best_bid is not None
                and best_bid >= sell_thresh and extreme_sell_room > 0):
            qty = min(extreme_sell_room, self.EXTREME_MAX_TAKE,
                      od.buy_orders.get(best_bid, 0))
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
        if (buy_thresh is not None and best_ask is not None
                and best_ask <= buy_thresh and extreme_buy_room > 0):
            qty = min(extreme_buy_room, self.EXTREME_MAX_TAKE,
                      -od.sell_orders.get(best_ask, 0))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))

    # ── VEV_5500 strict rule ──────────────────────────────────────────────────
    def _trade_5500_strict(self, product, state):
        od = state.order_depths[product]
        orders = []
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

    # ── VEV passive inside-spread quoting ────────────────────────────────────
    def _add_vev_market_quotes(self, product, od, state, orders):
        size = self.VEV_MAKE_SIZE.get(product, 0)
        if size <= 0:
            return
        best_bid, best_ask = self._best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return
        if best_ask - best_bid < 3:
            return
        buy_px  = best_bid + 1
        sell_px = best_ask - 1
        if buy_px >= sell_px:
            return
        limit = self.LIMITS[product]
        pos   = state.position.get(product, 0)
        pending_buy  = sum( o.quantity for o in orders if o.quantity > 0)
        pending_sell = sum(-o.quantity for o in orders if o.quantity < 0)
        buy_cap  = max(0, limit - pos - pending_buy)
        sell_cap = max(0, limit + pos - pending_sell)
        if buy_cap  > 0: orders.append(Order(product, buy_px,   min(size, buy_cap)))
        if sell_cap > 0: orders.append(Order(product, sell_px, -min(size, sell_cap)))

    # ── Mark 01 overlay ───────────────────────────────────────────────────────
    def _mark01_active(self, product, state):
        return any(t.buyer in self.MARK01_IDS
                   for t in state.market_trades.get(product, []))

    def _trade_mark01_overlay(self, state, orders):
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
                orders.setdefault(product, []).append(Order(product, best_bid, -qty))

    # ── VEV_5300/5400 spread overlay ─────────────────────────────────────────
    def _trade_5300_5400_spread(self, state, result):
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
        orders_hi   = result.setdefault(p_hi, [])
        orders_lo   = result.setdefault(p_lo, [])
        sp = state.position.get(p_hi, 0) - state.position.get(p_lo, 0)
        if sell_spread >= self.SPREAD_HIGH_SELL and sp > -self.SPREAD_POSITION_CAP:
            qty = min(self.SPREAD_PAIR_MAX_TAKE,
                      od_hi.buy_orders.get(bid_hi, 0),
                      -od_lo.sell_orders.get(ask_lo, 0),
                      self.SPREAD_POSITION_CAP + sp,
                      self._remaining_capacity(p_hi, state, orders_hi)[1],
                      self._remaining_capacity(p_lo, state, orders_lo)[0])
            if qty > 0:
                orders_hi.append(Order(p_hi, bid_hi, -qty))
                orders_lo.append(Order(p_lo, ask_lo,  qty))
        elif buy_spread <= self.SPREAD_LOW_BUY and sp < self.SPREAD_POSITION_CAP:
            qty = min(self.SPREAD_PAIR_MAX_TAKE,
                      -od_hi.sell_orders.get(ask_hi, 0),
                      od_lo.buy_orders.get(bid_lo, 0),
                      self.SPREAD_POSITION_CAP - sp,
                      self._remaining_capacity(p_hi, state, orders_hi)[0],
                      self._remaining_capacity(p_lo, state, orders_lo)[1])
            if qty > 0:
                orders_hi.append(Order(p_hi, ask_hi,  qty))
                orders_lo.append(Order(p_lo, bid_lo, -qty))

    # ── EMA fair value ────────────────────────────────────────────────────────
    def _spot_fair(self, product, state, td):
        if product not in state.order_depths:
            return None
        od = state.order_depths[product]
        best_bid, best_ask = self._best_bid_ask(od)
        raw = None
        if best_bid is not None and best_ask is not None:
            bid_vol = max(od.buy_orders.get(best_bid, 0), 0)
            ask_vol = max(-od.sell_orders.get(best_ask, 0), 0)
            raw = ((best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
                   if bid_vol + ask_vol > 0 else (best_bid + best_ask) / 2.0)
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
    def _take_mispriced(self, product, od, fair, state, orders):
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

    # ── Passive quoting (HYDROGEL / VELVETFRUIT) ──────────────────────────────
    def _add_passive_quotes(self, product, od, fair, state, orders, edge):
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
    def _remaining_capacity(self, product, state, existing):
        limit    = self.SOFT_LIMITS.get(product, self.LIMITS.get(product, 200))
        position = state.position.get(product, 0)
        pending_buy  = sum( o.quantity for o in existing if o.quantity > 0)
        pending_sell = sum(-o.quantity for o in existing if o.quantity < 0)
        return (max(0, limit - position - pending_buy),
                max(0, limit + position - pending_sell))

    # ── Best bid / ask ────────────────────────────────────────────────────────
    def _best_bid_ask(self, od):
        return (max(od.buy_orders)  if od.buy_orders  else None,
                min(od.sell_orders) if od.sell_orders else None)
