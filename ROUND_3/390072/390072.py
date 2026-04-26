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
        "HYDROGEL_PACK": 80, "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 200, "VEV_4500": 200, "VEV_5000": 200, "VEV_5100": 200, "VEV_5200": 200,
        "VEV_5300": 200, "VEV_5400": 200, "VEV_5500": 200, "VEV_6000": 200, "VEV_6500": 200,
    }
    STRIKES = {
        "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000, "VEV_6500": 6500,
    }

    def get_delta(self, symbol, underlying_price):
        if symbol not in self.STRIKES: return 0
        strike = self.STRIKES[symbol]
        # Simple linear delta approximation for calls: 0 at -200, 1 at +200 from strike
        diff = underlying_price - strike
        delta = 0.5 + (diff / 400.0)
        return max(0, min(1, delta))

    def run(self, state: TradingState):
        orders = {}; conversions = 0
        try: td = json.loads(state.traderData) if state.traderData else {}
        except: td = {}

        vev_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            od = state.order_depths["VELVETFRUIT_EXTRACT"]
            if od.buy_orders and od.sell_orders:
                vev_mid = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0

        total_option_delta = 0
        # 1. Trade Options first to see current positions
        for product in state.order_depths:
            if product in self.STRIKES:
                intrinsic = max(0, vev_mid - self.STRIKES[product]) if vev_mid else None
                # Use slightly wider spreads for ATM to mitigate losses
                orders[product] = self._trade_option(state, product, intrinsic, vev_mid)
                
                # Calculate aggregate delta exposure
                current_pos = state.position.get(product, 0)
                if vev_mid:
                    total_option_delta += current_pos * self.get_delta(product, vev_mid)

        # 2. Trade Hydrogel (Make/Take mixture)
        if "HYDROGEL_PACK" in state.order_depths:
            orders["HYDROGEL_PACK"] = self._trade_hydrogel(state)

        # 3. Trade Underlying (VELVETFRUIT_EXTRACT) + Delta Hedge
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            # We want Underlying_Pos = -Total_Option_Delta to be delta neutral
            hedge_target = -round(total_option_delta)
            orders["VELVETFRUIT_EXTRACT"] = self._trade_underlying_with_hedge(state, hedge_target)

        tdo = json.dumps(td); logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    def _trade_hydrogel(self, state):
        product = "HYDROGEL_PACK"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT[product]
        result = []

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        fair = (best_bid + best_ask) / 2.0

        buy_cap = limit - pos
        sell_cap = limit + pos

        # TAKE: Better than fair
        for price, vol in sorted(od.sell_orders.items()):
            if price < fair - 1 and buy_cap > 0:
                v = min(-vol, buy_cap)
                result.append(Order(product, price, v)); buy_cap -= v
        for price, vol in sorted(od.buy_orders.items(), reverse=True):
            if price > fair + 1 and sell_cap > 0:
                v = min(vol, sell_cap)
                result.append(Order(product, price, -v)); sell_cap -= v

        # MAKE: Join book
        if buy_cap > 0: result.append(Order(product, best_bid, buy_cap))
        if sell_cap > 0: result.append(Order(product, best_ask, -sell_cap))
        return result

    def _trade_option(self, state, product, intrinsic, vev_mid):
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT[product]
        result = []
        
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        if not best_bid or not best_ask: return []
        
        fair = (best_bid + best_ask) / 2.0
        # If intrinsic is significantly higher/lower, use it to pull the fair
        if intrinsic is not None:
            fair = (fair + intrinsic) / 2.0

        # Conservative Take
        buy_cap = limit - pos
        sell_cap = limit + pos
        
        for price, vol in sorted(od.sell_orders.items()):
            if price < fair - 1 and buy_cap > 0:
                v = min(-vol, buy_cap)
                result.append(Order(product, price, v)); buy_cap -= v
        for price, vol in sorted(od.buy_orders.items(), reverse=True):
            if price > fair + 1 and sell_cap > 0:
                v = min(vol, sell_cap)
                result.append(Order(product, price, -v)); sell_cap -= v

        # Passive: penny inside if spread permits
        if best_ask - best_bid > 2:
            if buy_cap > 0: result.append(Order(product, best_bid + 1, buy_cap))
            if sell_cap > 0: result.append(Order(product, best_ask - 1, -sell_cap))
        else: # Join
            if buy_cap > 0: result.append(Order(product, best_bid, buy_cap))
            if sell_cap > 0: result.append(Order(product, best_ask, -sell_cap))
        return result

    def _trade_underlying_with_hedge(self, state, hedge_target):
        product = "VELVETFRUIT_EXTRACT"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT[product]
        result = []

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        fair = (best_bid + best_ask) / 2.0
        
        # Clamp hedge target to limits
        hedge_target = max(-limit, min(limit, hedge_target))
        
        # First: fulfill hedge requirement via taking
        diff = hedge_target - pos
        if diff > 0: # Need to buy
            for price, vol in sorted(od.sell_orders.items()):
                if diff <= 0: break
                v = min(-vol, diff)
                result.append(Order(product, price, v)); diff -= v; pos += v
        elif diff < 0: # Need to sell
            for price, vol in sorted(od.buy_orders.items(), reverse=True):
                if diff >= 0: break
                v = min(vol, abs(diff))
                result.append(Order(product, price, -v)); diff += v; pos -= v

        # Second: MM for the remaining capacity
        buy_cap = limit - pos
        sell_cap = limit + pos
        if buy_cap > 0: result.append(Order(product, best_bid, buy_cap))
        if sell_cap > 0: result.append(Order(product, best_ask, -sell_cap))
        return result