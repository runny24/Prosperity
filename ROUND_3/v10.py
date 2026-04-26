"""
Round 3 v10 — BS(Realized Vol) Fair Value + Aggressive OTM Shorting
=====================================================================
KEY INSIGHT FROM DATA ANALYSIS:
  VEV realized vol = 0.0373 per sim-time-unit (consistent across all 3 days).
  On Day 2, OTM options (5300/5400/5500) are priced WAY above BS(RV) fair value
  and settle at 0 (VEV never exceeds ~5300). Immediate short → hold to expiry
  = virtually guaranteed profit.

  Settlement analysis at Day 2 open (VEV=5267.5, final=5295.5):
    VEV_5300: bid=52, BS_fair=31, settles=0  → short 200 → +10,400
    VEV_5400: bid=16, BS_fair=7,  settles=0  → short 200 → +3,200
    VEV_5500: bid=6,  BS_fair=1,  settles=0  → short 200 → +1,200
    Total from OTM shorts alone: 14,800

  Deep ITM options (4000/4500): BS_fair ≈ intrinsic; SKIP (market-priced fairly)
  Near-ITM (5000/5100/5200): delta risk — if VEV moves up, shorts lose; SKIP

STRATEGY:
  1. AGGRESSIVELY SHORT OTM calls (5300, 5400, 5500):
     - At every tick, sell at NPC bid whenever sell_capacity > 0 and bid > BS_fair
     - Fill up to position limit as fast as possible, then hold to expiry
     - DO NOT cycle (spread cost > theta benefit from cycling)
  2. Deep ITM (4000/4500): use intrinsic fair = EMA(vev) - strike for tight MM
  3. Near-ITM/ATM (5000-5200): use BS(RV) as fair value for tight MM
  4. HYDROGEL: fixed anchor=10000 (mean-reverting), offset=3
  5. VEV: tight offset=1, EMA fair, inventory skew
  6. Skip VEV_6000/6500 (spread=1, no room; bid=0 → zero edge)

REALIZED VOL: 0.0373 per sim-unit (T = ticks_remaining / 30000)
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
    t_val = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5*t_val + a4)*t_val) + a3)*t_val + a2)*t_val + a1)*t_val * math.exp(-x_abs*x_abs/2)
    return 0.5 * (1.0 + sign * y)


def bs_call(S, K, T, vol):
    """Black-Scholes call price. Returns intrinsic if T or vol is negligible."""
    intrinsic = max(0.0, S - K)
    if T <= 0.0001 or vol <= 0.001 or S <= 0:
        return intrinsic
    try:
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * vol * vol * T) / (vol * sqrt_T)
        d2 = d1 - vol * sqrt_T
        return S * _norm_cdf(d1) - K * _norm_cdf(d2)
    except:
        return intrinsic


class Trader:

    LIMIT = {
        "HYDROGEL_PACK": 80,
        "VELVETFRUIT_EXTRACT": 200,
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

    DEEP_ITM = {"VEV_4000", "VEV_4500"}

    # Options where we SHORT aggressively (OTM, minimal delta risk, large theta edge)
    OTM_SHORT = {"VEV_5300", "VEV_5400", "VEV_5500"}

    # Skip (zero bid, no edge)
    SKIP = {"VEV_6000", "VEV_6500"}

    # Realized vol of VEV (0.0373 per sim time unit where T = ticks_remaining / 30000)
    REALIZED_VOL = 0.0373

    # Simulation constants
    TOTAL_TICKS = 30000
    TICKS_PER_DAY = 10000

    # HYDROGEL mean-reverts around 10000
    HYDROGEL_ANCHOR = 10000.0

    # Quoting offsets for the MM layer
    OFFSET = {
        "HYDROGEL_PACK": 3,
        "VELVETFRUIT_EXTRACT": 1,
        "VEV_4000": 4,
        "VEV_4500": 4,
        "VEV_5000": 2,
        "VEV_5100": 1,
        "VEV_5200": 1,
        "VEV_5300": 1,
        "VEV_5400": 1,
        "VEV_5500": 1,
    }

    # Secondary level offsets (multi-level quoting for wide products)
    OFFSET2 = {
        "HYDROGEL_PACK": 6,
        "VEV_4000": 8,
        "VEV_4500": 7,
        "VEV_5000": 3,
    }

    PRIMARY_FRAC = 0.7

    # Inventory skew
    SKEW = {
        "HYDROGEL_PACK": 0.10,
        "VELVETFRUIT_EXTRACT": 0.012,
        "VEV_4000": 0.05,
        "VEV_4500": 0.04,
        "VEV_5000": 0.015,
        "VEV_5100": 0.01,
        "VEV_5200": 0.008,
        "VEV_5300": 0.006,
        "VEV_5400": 0.006,
        "VEV_5500": 0.006,
    }

    EMA_ALPHA = 0.3

    def run(self, state: TradingState):
        orders = {}
        conversions = 0
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except:
            td = {}

        ema = td.get("ema", {})
        day = td.get("day", 0)
        prev_ts = td.get("prev_ts", -1)
        if state.timestamp < prev_ts:
            day += 1
            td["day"] = day
        td["prev_ts"] = state.timestamp

        # Time to expiry in sim units
        tick_in_sim = day * self.TICKS_PER_DAY + state.timestamp // 100
        T = max(0.0001, (self.TOTAL_TICKS - tick_in_sim) / self.TOTAL_TICKS)

        # ── VEV underlying: EMA-smoothed fair value ──────────────
        vev_fair = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vev_od = state.order_depths["VELVETFRUIT_EXTRACT"]
            if vev_od.buy_orders and vev_od.sell_orders:
                vev_mid = (max(vev_od.buy_orders.keys()) + min(vev_od.sell_orders.keys())) / 2.0
                prev = ema.get("VELVETFRUIT_EXTRACT")
                vev_fair = self.EMA_ALPHA * vev_mid + (1 - self.EMA_ALPHA) * prev if prev else vev_mid
                ema["VELVETFRUIT_EXTRACT"] = vev_fair

        for product in state.order_depths:
            od = state.order_depths[product]

            if product in self.SKIP:
                orders[product] = []
                continue

            if not od.buy_orders or not od.sell_orders:
                orders[product] = []
                continue

            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid

            pos = state.position.get(product, 0)
            limit = self.LIMIT.get(product, 200)
            buy_cap = limit - pos
            sell_cap = limit + pos
            result = []

            # ════════════════════════════════════════════════════
            #  OTM_SHORT products: aggressively sell at NPC bid
            #  These options settle at 0 (VEV never reaches strikes)
            #  BS(RV) fair << market bid → free edge from theta decay
            # ════════════════════════════════════════════════════
            if product in self.OTM_SHORT and vev_fair is not None:
                K = self.STRIKES[product]
                fair = bs_call(vev_fair, K, T, self.REALIZED_VOL)

                # TAKE: sell at bid if bid > fair (take all available)
                for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                    if bid_price > fair and sell_cap > 0:
                        vol = min(od.buy_orders[bid_price], sell_cap)
                        result.append(Order(product, bid_price, -vol))
                        sell_cap -= vol
                    else:
                        break

                # TAKE: buy at ask if ask < fair (rare for OTM, but handle it)
                for ask_price in sorted(od.sell_orders.keys()):
                    if ask_price < fair and buy_cap > 0:
                        vol = min(-od.sell_orders[ask_price], buy_cap)
                        result.append(Order(product, ask_price, vol))
                        buy_cap -= vol
                    else:
                        break

                # MAKE: quote around BS(RV) fair with tight offset + skew
                offset = self.OFFSET.get(product, 1)
                skew = pos * self.SKEW.get(product, 0.006)
                buy_px = round(fair - offset - skew)
                sell_px = round(fair + offset - skew)
                if buy_px >= sell_px:
                    center = round(fair - skew)
                    buy_px = center - 1
                    sell_px = center + 1
                buy_px = max(buy_px, best_bid)
                sell_px = min(sell_px, best_ask)
                buy_px = min(buy_px, best_ask - 1)
                sell_px = max(sell_px, best_bid + 1)
                buy_px = max(0, buy_px)
                sell_px = max(1, sell_px)

                if buy_cap > 0:
                    result.append(Order(product, buy_px, buy_cap))
                if sell_cap > 0:
                    result.append(Order(product, sell_px, -sell_cap))

                orders[product] = result
                continue

            # ════════════════════════════════════════════════════
            #  All other products: compute fair value, then MM
            # ════════════════════════════════════════════════════
            if product == "HYDROGEL_PACK":
                fair = self.HYDROGEL_ANCHOR

            elif product in self.DEEP_ITM and vev_fair is not None:
                # Intrinsic fair for deep ITM options
                raw_fair = vev_fair - self.STRIKES[product]
                prev = ema.get(product)
                fair = self.EMA_ALPHA * raw_fair + (1 - self.EMA_ALPHA) * prev if prev else raw_fair
                ema[product] = fair

            elif product in self.STRIKES and vev_fair is not None:
                # BS(realized vol) fair for ATM options
                K = self.STRIKES[product]
                intrinsic = max(0.0, vev_fair - K)
                raw_fair = bs_call(vev_fair, K, T, self.REALIZED_VOL)
                raw_fair = max(raw_fair, intrinsic)
                prev = ema.get(product)
                fair = self.EMA_ALPHA * raw_fair + (1 - self.EMA_ALPHA) * prev if prev else raw_fair
                ema[product] = fair

            elif product == "VELVETFRUIT_EXTRACT":
                fair = vev_fair if vev_fair is not None else mid

            else:
                raw_fair = mid
                prev = ema.get(product)
                fair = self.EMA_ALPHA * raw_fair + (1 - self.EMA_ALPHA) * prev if prev else raw_fair
                ema[product] = fair

            offset = self.OFFSET.get(product, 2)
            skew = pos * self.SKEW.get(product, 0.01)

            # ── TAKE: sweep mispriced resting orders ────────────
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price < fair and buy_cap > 0:
                    vol = min(-od.sell_orders[ask_price], buy_cap)
                    result.append(Order(product, ask_price, vol))
                    buy_cap -= vol
                else:
                    break

            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price > fair and sell_cap > 0:
                    vol = min(od.buy_orders[bid_price], sell_cap)
                    result.append(Order(product, bid_price, -vol))
                    sell_cap -= vol
                else:
                    break

            # ── MAKE: tight quotes around fair ──────────────────
            buy_px = round(fair - offset - skew)
            sell_px = round(fair + offset - skew)

            if buy_px >= sell_px:
                center = round(fair - skew)
                buy_px = center - 1
                sell_px = center + 1

            buy_px = max(buy_px, best_bid)
            sell_px = min(sell_px, best_ask)
            buy_px = min(buy_px, best_ask - 1)
            sell_px = max(sell_px, best_bid + 1)
            buy_px = max(0, buy_px)
            sell_px = max(1, sell_px)

            # ── Multi-level for wide products ────────────────────
            if product in self.OFFSET2 and spread >= 6:
                off2 = self.OFFSET2[product]
                buy_px2 = round(fair - off2 - skew)
                sell_px2 = round(fair + off2 - skew)
                buy_px2 = max(best_bid, max(0, buy_px2))
                sell_px2 = min(best_ask, max(1, sell_px2))
                buy_px2 = min(buy_px2, buy_px - 1)
                sell_px2 = max(sell_px2, sell_px + 1)

                frac = self.PRIMARY_FRAC
                buy_q1 = max(1, int(buy_cap * frac))
                buy_q2 = buy_cap - buy_q1
                sell_q1 = max(1, int(sell_cap * frac))
                sell_q2 = sell_cap - sell_q1

                if buy_cap > 0:
                    result.append(Order(product, buy_px, buy_q1))
                    if buy_q2 > 0 and buy_px2 >= 0 and buy_px2 < buy_px:
                        result.append(Order(product, buy_px2, buy_q2))
                    elif buy_q2 > 0:
                        result.append(Order(product, buy_px, buy_q2))
                if sell_cap > 0:
                    result.append(Order(product, sell_px, -sell_q1))
                    if sell_q2 > 0 and sell_px2 > sell_px:
                        result.append(Order(product, sell_px2, -sell_q2))
                    elif sell_q2 > 0:
                        result.append(Order(product, sell_px, -sell_q2))
            else:
                if buy_cap > 0:
                    result.append(Order(product, buy_px, buy_cap))
                if sell_cap > 0:
                    result.append(Order(product, sell_px, -sell_cap))

            orders[product] = result

        td["ema"] = ema
        tdo = json.dumps(td)
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo
