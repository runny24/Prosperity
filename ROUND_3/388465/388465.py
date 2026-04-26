"""
Round 3 v4 — Intrinsic Fair + Join-Book OTM
=============================================
Changes from v3:
1. Deep ITM (4000, 4500): fair = VEV - strike (intrinsic value)
   Catches 7.5% of ticks where option is mispriced vs underlying
2. OTM (5300-5500, 6000, 6500): join book at bid/ask for more fills
   Tight 1-2pt spreads make penny-inside useless; joining captures NPC crosses
3. ATM (5000-5200): market-mid + take mispricings vs intrinsic
4. HP, VEV: unchanged (working well)
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

    # Deep ITM: delta ≈ 1, use intrinsic
    DEEP_ITM = {"VEV_4000", "VEV_4500"}
    # OTM: tight spreads, join book
    OTM = {"VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"}

    def run(self, state: TradingState):
        orders = {}; conversions = 0
        try: td = json.loads(state.traderData) if state.traderData else {}
        except: td = {}

        # Get VEV mid for intrinsic value calculation
        vev_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vev_od = state.order_depths["VELVETFRUIT_EXTRACT"]
            if vev_od.buy_orders and vev_od.sell_orders:
                vev_mid = (max(vev_od.buy_orders.keys()) + min(vev_od.sell_orders.keys())) / 2.0

        for product in state.order_depths:
            if product == "HYDROGEL_PACK":
                orders[product] = self._trade_mm(state, product, None)
            elif product == "VELVETFRUIT_EXTRACT":
                orders[product] = self._trade_mm(state, product, None)
            elif product in self.DEEP_ITM:
                # Deep ITM: use intrinsic = VEV - strike
                intrinsic = (vev_mid - self.STRIKES[product]) if vev_mid else None
                orders[product] = self._trade_mm(state, product, intrinsic)
            elif product in self.OTM:
                orders[product] = self._trade_otm(state, product, vev_mid)
            elif product in self.STRIKES:
                # ATM: use market mid, but take vs intrinsic if mispriced
                intrinsic = max(0, vev_mid - self.STRIKES[product]) if vev_mid else None
                orders[product] = self._trade_atm(state, product, intrinsic)

        tdo = json.dumps(td); logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ═══════════════════════════════════════════════════════════════
    #  Generic penny-inside MM (HP, VEV, deep ITM)
    # ═══════════════════════════════════════════════════════════════
    def _trade_mm(self, state, product, external_fair):
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT.get(product, 200)
        result = []

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        if best_bid is None or best_ask is None:
            return result

        market_mid = (best_bid + best_ask) / 2.0
        # Use external fair if provided (e.g., intrinsic for deep ITM)
        fair = external_fair if external_fair is not None else market_mid
        fair = round(fair)

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # Take
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

        # Passive penny-inside
        buy_price = best_bid + 1 if best_bid + 1 < fair else fair - 1
        sell_price = best_ask - 1 if best_ask - 1 > fair else fair + 1

        if buy_capacity > 0:
            result.append(Order(product, buy_price, buy_capacity))
        if sell_capacity > 0:
            result.append(Order(product, sell_price, -sell_capacity))

        return result

    # ═══════════════════════════════════════════════════════════════
    #  ATM options: market-mid MM + take intrinsic mispricings
    # ═══════════════════════════════════════════════════════════════
    def _trade_atm(self, state, product, intrinsic):
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT.get(product, 200)
        result = []

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        if best_bid is None or best_ask is None:
            return result

        market_mid = (best_bid + best_ask) / 2.0
        fair = round(market_mid)

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # Take vs market mid
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

        # Also take if mispriced vs intrinsic (when intrinsic > market ask → cheap option)
        if intrinsic is not None:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price < intrinsic - 1 and buy_capacity > 0:
                    vol = min(-od.sell_orders[ask_price], buy_capacity)
                    result.append(Order(product, ask_price, vol))
                    buy_capacity -= vol
                else: break

            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price > intrinsic + 1 and sell_capacity > 0:
                    vol = min(od.buy_orders[bid_price], sell_capacity)
                    result.append(Order(product, bid_price, -vol))
                    sell_capacity -= vol
                else: break

        # Passive penny-inside
        buy_price = best_bid + 1 if best_bid + 1 < fair else fair - 1
        sell_price = best_ask - 1 if best_ask - 1 > fair else fair + 1

        if buy_capacity > 0 and buy_price >= 0:
            result.append(Order(product, buy_price, buy_capacity))
        if sell_capacity > 0 and sell_price > 0:
            result.append(Order(product, sell_price, -sell_capacity))

        return result

    # ═══════════════════════════════════════════════════════════════
    #  OTM options: join book at bid/ask (tight spreads)
    # ═══════════════════════════════════════════════════════════════
    def _trade_otm(self, state, product, vev_mid):
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
        spread = best_ask - best_bid

        # Intrinsic value (usually 0 for OTM)
        intrinsic = max(0, vev_mid - strike) if vev_mid else 0

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # Take: buy below intrinsic (shouldn't happen often for OTM)
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price < intrinsic and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol))
                buy_capacity -= vol
            else: break

        # Take: sell above market mid (captures any NPC aggression)
        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price > market_mid and sell_capacity > 0:
                vol = min(od.buy_orders[bid_price], sell_capacity)
                result.append(Order(product, bid_price, -vol))
                sell_capacity -= vol
            else: break

        # Passive: JOIN the book (post at best bid and best ask)
        # For tight spreads, this captures NPC crosses without needing edge
        buy_price = max(0, best_bid)
        sell_price = best_ask

        # If spread > 2, try penny-inside instead
        if spread > 2:
            buy_price = max(0, best_bid + 1)
            sell_price = best_ask - 1

        if buy_capacity > 0 and buy_price >= 0:
            result.append(Order(product, buy_price, buy_capacity))
        if sell_capacity > 0 and sell_price > 0:
            result.append(Order(product, sell_price, -sell_capacity))

        return result