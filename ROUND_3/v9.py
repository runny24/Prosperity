"""
Round 3 v9 — Best-of-All: Fixed Anchor + Tight Quoting + Intrinsic ITM
=======================================================================
IMPROVEMENTS OVER 396164 (v8):

1. HYDROGEL_PACK: FIXED anchor=10000 (not EMA of market mid)
   - 396164 EMA drifted with price → adverse selection → -1,324
   - HYDROGEL mean-reverts around 10000; EMA chases noise
   - Fixed anchor=10000 restores ~+610 from Elias baseline

2. Deep ITM (VEV_4000, VEV_4500): Intrinsic fair = vev_EMA - strike
   - 396164 used EMA of market mid → +91/+76
   - Elias used intrinsic → +278/+243
   - Restoring intrinsic fair should recover ~+350 additional PnL

3. KEEP everything that works in 396164:
   - VEV: tight offset=1, EMA fair, inventory skew → +2,374
   - ATM/OTM (5000-5500): tight offset=1, EMA fair, inventory skew → +692/+769/+574
   - Multi-level quoting for wide-spread products
   - Skip VEV_6000 / VEV_6500

Expected improvement: ~3,508 - (-1,324-91-76) + (610+278+243) ≈ 4,500+
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
    SKIP = {"VEV_6000", "VEV_6500"}

    # HYDROGEL mean-reverts around 10000; use fixed anchor, not EMA
    HYDROGEL_ANCHOR = 10000

    # Quoting offsets from fair value (~25-50% of NPC half-spread)
    OFFSET = {
        "HYDROGEL_PACK": 3,         # NPC half-spread≈8; anchor=10000 so tighter is safer
        "VELVETFRUIT_EXTRACT": 1,   # NPC half-spread≈2.5
        "VEV_4000": 4,              # NPC half-spread≈10.5; intrinsic fair so offset=4
        "VEV_4500": 4,              # NPC half-spread≈8; intrinsic fair so offset=4
        "VEV_5000": 2,              # NPC half-spread≈3
        "VEV_5100": 1,              # NPC half-spread≈2.25
        "VEV_5200": 1,              # NPC half-spread≈1.5
        "VEV_5300": 1,              # NPC half-spread≈1
        "VEV_5400": 1,              # NPC half-spread≈0.75
        "VEV_5500": 1,              # NPC half-spread≈0.5
    }

    # Secondary level offsets for wide-spread products
    OFFSET2 = {
        "HYDROGEL_PACK": 6,
        "VEV_4000": 8,
        "VEV_4500": 7,
        "VEV_5000": 3,
    }

    # Fraction of capacity at the tight primary level
    PRIMARY_FRAC = 0.7

    # Inventory skew: shifts both quotes toward unwinding position
    SKEW = {
        "HYDROGEL_PACK": 0.10,       # full 80 → 8 tick skew
        "VELVETFRUIT_EXTRACT": 0.012, # full 200 → 2.4 tick skew
        "VEV_4000": 0.05,            # full 200 → 10 tick skew
        "VEV_4500": 0.04,            # full 200 → 8 tick skew
        "VEV_5000": 0.015,           # full 200 → 3 tick skew
        "VEV_5100": 0.01,
        "VEV_5200": 0.008,
        "VEV_5300": 0.006,
        "VEV_5400": 0.006,
        "VEV_5500": 0.006,
    }

    EMA_ALPHA = 0.3  # higher = more responsive to new data

    def run(self, state: TradingState):
        orders = {}
        conversions = 0
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except:
            td = {}

        ema = td.get("ema", {})

        # ── VEV underlying: EMA-smoothed fair value ──
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

            # ── Fair value ──────────────────────────────────────────
            if product == "HYDROGEL_PACK":
                # FIXED anchor — HYDROGEL mean-reverts around 10000
                # Do NOT use EMA here; it chases noise and causes adverse selection
                fair = float(self.HYDROGEL_ANCHOR)

            elif product in self.DEEP_ITM and vev_fair is not None:
                # Intrinsic fair for deep ITM options: massively better than market mid
                raw_fair = vev_fair - self.STRIKES[product]
                prev = ema.get(product)
                fair = self.EMA_ALPHA * raw_fair + (1 - self.EMA_ALPHA) * prev if prev else raw_fair
                ema[product] = fair

            elif product in self.STRIKES and vev_fair is not None:
                # ATM/OTM options: EMA of market mid
                # (intrinsic would be zero or very small for OTM, so use market mid)
                intrinsic = max(0.0, vev_fair - self.STRIKES[product])
                raw_fair = max(mid, intrinsic)
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

            pos = state.position.get(product, 0)
            limit = self.LIMIT.get(product, 200)
            buy_cap = limit - pos
            sell_cap = limit + pos

            result = []
            offset = self.OFFSET.get(product, 2)
            skew = pos * self.SKEW.get(product, 0.01)

            # ── TAKE: sweep mispriced resting orders ────────────────
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

            # ── MAKE: tight quotes around fair ──────────────────────
            buy_px = round(fair - offset - skew)
            sell_px = round(fair + offset - skew)

            # Floor: ensure minimum spread of 2 ticks
            if buy_px >= sell_px:
                center = round(fair - skew)
                buy_px = center - 1
                sell_px = center + 1

            # Never post outside the NPC book (we'd never get filled)
            buy_px = max(buy_px, best_bid)
            sell_px = min(sell_px, best_ask)

            # Never cross the book aggressively (that's done in TAKE)
            buy_px = min(buy_px, best_ask - 1)
            sell_px = max(sell_px, best_bid + 1)

            # Sanity: non-negative
            buy_px = max(0, buy_px)
            sell_px = max(1, sell_px)

            # ── Multi-level quoting for wide-spread products ─────────
            if product in self.OFFSET2 and spread >= 6:
                off2 = self.OFFSET2[product]
                buy_px2 = round(fair - off2 - skew)
                sell_px2 = round(fair + off2 - skew)

                # Clamp secondary level inside book
                buy_px2 = max(best_bid, max(0, buy_px2))
                sell_px2 = min(best_ask, max(1, sell_px2))
                # Secondary must be strictly worse than primary
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
                # Single level: full remaining capacity
                if buy_cap > 0:
                    result.append(Order(product, buy_px, buy_cap))
                if sell_cap > 0:
                    result.append(Order(product, sell_px, -sell_cap))

            orders[product] = result

        td["ema"] = ema
        tdo = json.dumps(td)
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo
