"""
Round 3 v3 — Market-Mid Fair for ALL Products
==========================================
Fixes from v1:
1. VEV limit: 200 (was 400 → ALL orders rejected on every tick!)
2. HP fair: dynamic mid-based (was hardcoded 10000 → lost 2354)
3. TICKS_PER_DAY: 1000 (sim uses 1000 ticks, not 10000)
4. Options: keep limit 200 (no errors seen)

Strategy:
  HYDROGEL_PACK: penny-inside MM around EMA mid
  VEV: penny-inside MM around mid (now actually works)
  Options: penny-inside around MARKET MID (BS proved wrong, overpriced ATM/OTM)
"""

import json, math
from typing import Any
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


def _norm_cdf(x):
    if x < -8: return 0.0
    if x > 8: return 1.0
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x_abs = abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * math.exp(-x_abs*x_abs/2)
    return 0.5 * (1.0 + sign * y)


def bs_call(S, K, T, vol):
    if T <= 0.0001: return max(0.0, S - K)
    if S <= 0 or K <= 0 or vol <= 0: return max(0.0, S - K)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * vol * vol * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


class Trader:

    LIMIT = {
        "HYDROGEL_PACK": 80,
        "VELVETFRUIT_EXTRACT": 200,   # FIXED: was 400
        "VEV_4000": 200, "VEV_4500": 200,
        "VEV_5000": 200, "VEV_5100": 200, "VEV_5200": 200,
        "VEV_5300": 200, "VEV_5400": 200, "VEV_5500": 200,
        "VEV_6000": 200, "VEV_6500": 200,
    }

    STRIKES = {
        "VEV_4000": 4000, "VEV_4500": 4500,
        "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
        "VEV_6000": 6000, "VEV_6500": 6500,
    }

    VOL = 0.036
    TOTAL_TICKS = 3000   # 3 days × 1000 ticks (FIXED: was 30000)
    TICKS_PER_DAY = 1000  # FIXED: was 10000

    def run(self, state: TradingState):
        orders = {}; conversions = 0
        try: td = json.loads(state.traderData) if state.traderData else {}
        except: td = {}

        ts = state.timestamp
        day = td.get('day', 0)
        prev_ts = td.get('prev_ts', -1)
        if ts < prev_ts: day += 1; td['day'] = day
        td['prev_ts'] = ts

        tick_in_sim = day * self.TICKS_PER_DAY + ts // 100
        T = max(0.0001, (self.TOTAL_TICKS - tick_in_sim) / self.TOTAL_TICKS)

        vev_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vev_od = state.order_depths["VELVETFRUIT_EXTRACT"]
            if vev_od.buy_orders and vev_od.sell_orders:
                vev_mid = (max(vev_od.buy_orders.keys()) + min(vev_od.sell_orders.keys())) / 2.0

        for product in state.order_depths:
            if product == "HYDROGEL_PACK":
                orders[product] = self._trade_hydrogel(state, td)
            elif product == "VELVETFRUIT_EXTRACT":
                orders[product] = self._trade_vev(state, td)
            elif product in self.STRIKES:
                orders[product] = self._trade_option(state, td, product, vev_mid, T)

        tdo = json.dumps(td); logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ═══════════════════════════════════════════════════════════════
    #  HYDROGEL_PACK — Dynamic mid-based MM
    # ═══════════════════════════════════════════════════════════════
    def _trade_hydrogel(self, state, td):
        product = "HYDROGEL_PACK"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT[product]
        result = []

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        if best_bid is None or best_ask is None:
            return result

        ob_mid = (best_bid + best_ask) / 2.0
        FAIR = round(ob_mid)

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # Take
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price < FAIR and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol))
                buy_capacity -= vol
            else: break

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price > FAIR and sell_capacity > 0:
                vol = min(od.buy_orders[bid_price], sell_capacity)
                result.append(Order(product, bid_price, -vol))
                sell_capacity -= vol
            else: break

        # Passive penny-inside
        buy_price = best_bid + 1 if best_bid + 1 < FAIR else FAIR - 1
        sell_price = best_ask - 1 if best_ask - 1 > FAIR else FAIR + 1

        if buy_capacity > 0:
            result.append(Order(product, buy_price, buy_capacity))
        if sell_capacity > 0:
            result.append(Order(product, sell_price, -sell_capacity))

        return result

    # ═══════════════════════════════════════════════════════════════
    #  VELVETFRUIT_EXTRACT — Underlying MM
    # ═══════════════════════════════════════════════════════════════
    def _trade_vev(self, state, td):
        product = "VELVETFRUIT_EXTRACT"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT[product]
        result = []

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        if best_bid is None or best_ask is None:
            return result

        mid = (best_bid + best_ask) / 2.0
        FAIR = round(mid)

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # Take
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price < FAIR and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol))
                buy_capacity -= vol
            else: break

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price > FAIR and sell_capacity > 0:
                vol = min(od.buy_orders[bid_price], sell_capacity)
                result.append(Order(product, bid_price, -vol))
                sell_capacity -= vol
            else: break

        # Passive penny-inside
        buy_price = best_bid + 1 if best_bid + 1 < FAIR else FAIR - 1
        sell_price = best_ask - 1 if best_ask - 1 > FAIR else FAIR + 1

        if buy_capacity > 0:
            result.append(Order(product, buy_price, buy_capacity))
        if sell_capacity > 0:
            result.append(Order(product, sell_price, -sell_capacity))

        return result

    # ═══════════════════════════════════════════════════════════════
    #  VEV_XXXX — Options MM
    # ═══════════════════════════════════════════════════════════════
    def _trade_option(self, state, td, product, vev_mid, T):
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT.get(product, 200)
        result = []
        strike = self.STRIKES[product]

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        if best_bid is None or best_ask is None:
            return result

        market_mid = (best_bid + best_ask) / 2.0

        # Use market mid as fair (BS proved wrong — overprices ATM/OTM by 5-30pt)
        fair = market_mid
        fair_rounded = max(0, round(fair))

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # Take: buy below fair, sell above fair
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price < fair and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol))
                buy_capacity -= vol
            else: break

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price > fair and sell_capacity > 0:
                vol = min(od.buy_orders[bid_price], sell_capacity)
                result.append(Order(product, bid_price, -vol))
                sell_capacity -= vol
            else: break

        # Passive quotes
        spread = best_ask - best_bid
        if spread <= 2:
            buy_price = max(0, best_bid)
            sell_price = best_ask
        else:
            buy_price = max(0, fair_rounded - 1)
            sell_price = max(1, fair_rounded + 1)
            if best_bid + 1 < fair:
                buy_price = max(0, best_bid + 1)
            if best_ask - 1 > fair:
                sell_price = best_ask - 1

        if buy_capacity > 0 and buy_price >= 0:
            result.append(Order(product, buy_price, buy_capacity))
        if sell_capacity > 0 and sell_price > 0:
            result.append(Order(product, sell_price, -sell_capacity))

        return result