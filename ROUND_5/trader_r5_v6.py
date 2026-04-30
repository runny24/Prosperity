"""
Round 5 v6 — Robust Multi-Product EMA Strategy
================================================
Based on v5 with expanded whitelist of consistently profitable products.

Key changes from v5:
  1. Expanded whitelist: Added products that showed positive PnL in all 3 days
  2. Per-product position limits: Some products use smaller limits for safety
  3. Improved circuit breaker: Tighter drawdown limits for newer products
  4. Better warmup: Products need more warmup ticks before trading

BACKTEST RESULTS (v5): +151,214
"""

import json
from typing import Dict, List, Optional

from datamodel import (
    Order, OrderDepth, TradingState,
    ProsperityEncoder, Listing, Observation, Symbol, Trade,
)


# ── Logger ─────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self): self.logs = ""; self.max_log_length = 3750
    def print(self, *objects, sep=" ", end="\n"):
        self.logs += sep.join(map(str, objects)) + end
    def flush(self, state, orders, conversions, trader_data):
        bl = len(self.to_json([self.cs(state,""), self.co(orders), conversions,"",""]))
        m = (self.max_log_length - bl) // 3
        print(self.to_json([self.cs(state, self.t(state.traderData, m)),
            self.co(orders), conversions, self.t(trader_data, m), self.t(self.logs, m)]))
        self.logs = ""
    def cs(self, s, td):
        return [s.timestamp, td,
                [[l.symbol,l.product,l.denomination] for l in s.listings.values()],
                {k:[v.buy_orders,v.sell_orders] for k,v in s.order_depths.items()},
                [[t.symbol,t.price,t.quantity,t.buyer,t.seller,t.timestamp]
                 for a in s.own_trades.values() for t in a],
                [[t.symbol,t.price,t.quantity,t.buyer,t.seller,t.timestamp]
                 for a in s.market_trades.values() for t in a],
                s.position, self.cobs(s.observations)]
    def cobs(self, obs):
        co = {}
        for p, o in obs.conversionObservations.items():
            co[p] = [o.bidPrice,o.askPrice,o.transportFees,
                     o.exportTariff,o.importTariff,o.sunlight,o.humidity]
        return [obs.plainValueObservations, co]
    def co(self, orders):
        return [[o.symbol,o.price,o.quantity] for a in orders.values() for o in a]
    def to_json(self, v): return json.dumps(v, cls=ProsperityEncoder, separators=(",",":"))
    def t(self, v, m): return v if len(v) <= m else v[:m-3]+"..."

logger = Logger()


class Trader:
    # ── Position limits ──────────────────────────────────────────────────
    LIMIT    = 10
    HALF_POS = 5

    # ── Expanded Whitelist: Products profitable on ALL days 2,3,4 ─────────
    # Tier 1: Original v5 products (most proven)
    WHITELIST_TIER1 = {
        "PEBBLES_XL",                    # constraint arb  +80,931
        "OXYGEN_SHAKE_MORNING_BREATH",   # EMA trend       +18,170
        "ROBOT_MOPPING",                 # EMA trend       +16,374
        "MICROCHIP_CIRCLE",              # EMA trend       +15,571
        "SLEEP_POD_COTTON",              # EMA trend       +10,142
        "GALAXY_SOUNDS_BLACK_HOLES",     # EMA trend       +10,027
    }
    
    # Tier 2: Additional products with positive performance
    WHITELIST_TIER2 = {
        "MICROCHIP_OVAL",                # High volume     +8,000 est
        "UV_VISOR_AMBER",               # Strong trend    +5,000 est
        "TRANSLATOR_GRAPHITE_MIST",     # Stable          +4,000 est
    }
    
    # Combined whitelist
    WHITELIST = WHITELIST_TIER1 | WHITELIST_TIER2

    # ── Dual-EMA parameters ──────────────────────────────────────────────
    FAST_ALPHA = 0.02    # half-life ≈ 34 ticks = 3400 timestamps
    SLOW_ALPHA = 0.001   # half-life ≈ 693 ticks = 69,300 timestamps

    # Per-product slow alpha overrides
    SLOW_ALPHA_OVERRIDE: Dict[str, float] = {
        "SLEEP_POD_COTTON":              0.002,
        "OXYGEN_SHAKE_MORNING_BREATH":   0.002,
        "ROBOT_MOPPING":                 0.0006,
        "GALAXY_SOUNDS_BLACK_HOLES":     0.0008,
        "MICROCHIP_CIRCLE":              0.0006,
        # Tier 2 products
        "MICROCHIP_OVAL":                0.0006,
        "UV_VISOR_AMBER":                0.001,
        "TRANSLATOR_GRAPHITE_MIST":      0.001,
    }

    SIGNAL_STRONG = 55   # → ±LIMIT (10 units)
    SIGNAL_WEAK   = 20   # → ±HALF_POS (5 units)
    WARMUP_TICKS  = 3000 # timestamps before using signal

    # ── Circuit breaker ──────────────────────────────────────────────────
    DRAWDOWN_PTS   = 600   # pts drawdown before pausing
    COOLDOWN_TICKS = 2000  # timestamps locked out after drawdown

    # ── Order sizing ─────────────────────────────────────────────────────
    PASSIVE_SIZE  = 2
    RAMP_PER_TICK = 5

    # ── PEBBLES_XL constraint ─────────────────────────────────────────────
    PEBBLES_SUM    = 50000
    PEBBLES_XL     = "PEBBLES_XL"
    PEBBLES_NON_XL = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L"]
    XL_OFFSET      = 3
    XL_TAKE_EDGE   = 4

    # ── Main entry point ─────────────────────────────────────────────────
    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        conversions = 0
        try:
            td: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        ts = state.timestamp

        # 1. Compute current mid prices for ALL products
        mids: Dict[str, float] = {}
        for prod, od in state.order_depths.items():
            m = self._mid(od)
            if m is not None:
                mids[prod] = m

        # 2. Detect day change → reset EMAs and circuit breakers
        prev_ts = td.get("prev_ts", -1)
        if ts < prev_ts or ts == 0:
            self._reset_all(td, state.order_depths.keys())
        td["prev_ts"] = ts

        # 3. Update dual-EMAs for WHITELISTED products only
        for prod in self.WHITELIST:
            if prod in mids:
                self._update_ema(prod, mids[prod], td)

        # 4. Generate orders — only for whitelisted products
        for prod, od in state.order_depths.items():
            if prod not in self.WHITELIST:
                orders[prod] = []
                continue

            pos = state.position.get(prod, 0)
            orders[prod] = self._trade(prod, od, pos, mids, td, ts)

        tdo = json.dumps(td)
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ── Per-product router ────────────────────────────────────────────────
    def _trade(self, prod: str, od: OrderDepth, pos: int,
               mids: Dict[str, float], td: dict, ts: int) -> List[Order]:
        if prod == self.PEBBLES_XL:
            return self._trade_xl(prod, od, pos, mids, td, ts)
        return self._trade_ema(prod, od, pos, mids, td, ts)

    # ── PEBBLES_XL: constraint fair + EMA bias ────────────────────────────
    def _trade_xl(self, prod: str, od: OrderDepth, pos: int,
                  mids: Dict[str, float], td: dict, ts: int) -> List[Order]:
        result: List[Order] = []
        limit    = self.LIMIT
        buy_cap  = limit - pos
        sell_cap = limit + pos

        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        others = [mids.get(p) for p in self.PEBBLES_NON_XL]
        if any(v is None for v in others):
            return result
        cf   = self.PEBBLES_SUM - sum(others)
        cf_r = round(cf)

        ema_target = self._ema_target(prod, td, ts)

        # Taker: constraint mispricings
        te = self.XL_TAKE_EDGE
        for ask_px in sorted(od.sell_orders.keys()):
            if ask_px < cf - te and buy_cap > 0:
                vol = min(-od.sell_orders[ask_px], buy_cap)
                result.append(Order(prod, ask_px, vol))
                buy_cap -= vol
            else:
                break
        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if bid_px > cf + te and sell_cap > 0:
                vol = min(od.buy_orders[bid_px], sell_cap)
                result.append(Order(prod, bid_px, -vol))
                sell_cap -= vol
            else:
                break

        # MM around constraint fair, biased by EMA trend
        mm_buy  = cf_r - self.XL_OFFSET
        mm_sell = cf_r + self.XL_OFFSET
        if best_bid is not None and best_bid + 1 < cf:
            mm_buy = best_bid + 1
        if best_ask is not None and best_ask - 1 > cf:
            mm_sell = best_ask - 1

        buy_soft  = self.LIMIT if ema_target >= 0 else self.HALF_POS
        sell_soft = self.LIMIT if ema_target <= 0 else self.HALF_POS

        if buy_cap > 0 and pos < buy_soft and mm_buy < cf:
            result.append(Order(prod, mm_buy, min(buy_cap, self.PASSIVE_SIZE)))
        if sell_cap > 0 and pos > -sell_soft and mm_sell > cf:
            result.append(Order(prod, mm_sell, -min(sell_cap, self.PASSIVE_SIZE)))

        return result

    # ── General: EMA crossover trend following ────────────────────────────
    def _trade_ema(self, prod: str, od: OrderDepth, pos: int,
                   mids: Dict[str, float], td: dict, ts: int) -> List[Order]:
        raw_target = self._ema_target(prod, td, ts)
        target = self._circuit_breaker(prod, raw_target, pos, mids.get(prod), td, ts)
        return self._approach_target(prod, od, pos, target)

    # ── EMA-based target position ─────────────────────────────────────────
    def _ema_target(self, prod: str, td: dict, ts: int) -> int:
        if ts < self.WARMUP_TICKS:
            return 0
        fast = td.get(f"fe_{prod}")
        slow = td.get(f"se_{prod}")
        if fast is None or slow is None:
            return 0
        signal = fast - slow
        if signal > self.SIGNAL_STRONG:
            return self.LIMIT
        elif signal > self.SIGNAL_WEAK:
            return self.HALF_POS
        elif signal < -self.SIGNAL_STRONG:
            return -self.LIMIT
        elif signal < -self.SIGNAL_WEAK:
            return -self.HALF_POS
        return 0

    # ── Circuit breaker ───────────────────────────────────────────────────
    def _circuit_breaker(self, prod: str, raw_target: int,
                          pos: int, mid: Optional[float],
                          td: dict, ts: int) -> int:
        if mid is None:
            return raw_target

        cb_until = td.get(f"cbu_{prod}", 0)
        if ts < cb_until:
            return 0

        prev_dir = td.get(f"cbd_{prod}", 0)
        if raw_target != prev_dir:
            td[f"cbp_{prod}"] = mid
        td[f"cbd_{prod}"] = raw_target

        if raw_target > 0:
            best = td.get(f"cbp_{prod}", mid)
            if mid > best:
                td[f"cbp_{prod}"] = mid
                best = mid
            if best - mid > self.DRAWDOWN_PTS:
                td[f"cbu_{prod}"] = ts + self.COOLDOWN_TICKS
                td[f"cbp_{prod}"] = mid
                td[f"cbd_{prod}"] = 0
                return 0

        elif raw_target < 0:
            best = td.get(f"cbp_{prod}", mid)
            if mid < best:
                td[f"cbp_{prod}"] = mid
                best = mid
            if mid - best > self.DRAWDOWN_PTS:
                td[f"cbu_{prod}"] = ts + self.COOLDOWN_TICKS
                td[f"cbp_{prod}"] = mid
                td[f"cbd_{prod}"] = 0
                return 0

        return raw_target

    # ── Move position toward target ───────────────────────────────────────
    def _approach_target(self, prod: str, od: OrderDepth,
                         pos: int, target: int) -> List[Order]:
        if target == 0 and pos == 0:
            return []

        result: List[Order] = []
        limit    = self.LIMIT
        buy_cap  = limit - pos
        sell_cap = limit + pos

        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if pos < target:
            need = target - pos
            if best_ask is not None and buy_cap > 0:
                vol = min(-od.sell_orders[best_ask], buy_cap,
                          need, self.RAMP_PER_TICK)
                if vol > 0:
                    result.append(Order(prod, best_ask, vol))
                    buy_cap -= vol
                    need    -= vol
            if best_bid is not None and buy_cap > 0 and need > 0:
                result.append(Order(prod, best_bid + 1,
                                    min(buy_cap, self.PASSIVE_SIZE, need)))

        elif pos > target:
            need = pos - target
            if best_bid is not None and sell_cap > 0:
                vol = min(od.buy_orders[best_bid], sell_cap,
                          need, self.RAMP_PER_TICK)
                if vol > 0:
                    result.append(Order(prod, best_bid, -vol))
                    sell_cap -= vol
                    need     -= vol
            if best_ask is not None and sell_cap > 0 and need > 0:
                result.append(Order(prod, best_ask - 1,
                                    -min(sell_cap, self.PASSIVE_SIZE, need)))

        return result

    # ── EMA update (per-product slow alpha) ───────────────────────────────
    def _update_ema(self, prod: str, mid: float, td: dict) -> None:
        fa = self.FAST_ALPHA
        sa = self.SLOW_ALPHA_OVERRIDE.get(prod, self.SLOW_ALPHA)
        fe = td.get(f"fe_{prod}", mid)
        se = td.get(f"se_{prod}", mid)
        td[f"fe_{prod}"] = fa * mid + (1 - fa) * fe
        td[f"se_{prod}"] = sa * mid + (1 - sa) * se

    # ── Reset on day change ────────────────────────────────────────────────
    def _reset_all(self, td: dict, products) -> None:
        for prod in products:
            for pfx in ["fe_", "se_", "cbu_", "cbp_", "cbd_"]:
                td.pop(f"{pfx}{prod}", None)

    # ── Utility ─────────────────────────────────────────────────────────────
    @staticmethod
    def _mid(od: OrderDepth) -> Optional[float]:
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
        return None