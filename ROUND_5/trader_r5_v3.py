"""
Round 5 v3 — Dual-EMA Trend Following + PEBBLES Constraint
============================================================
KEY INSIGHT FROM BACKTESTING PRIOR VERSIONS:
  v1: EMA max() bug → passive bids above market in down-trends → -297k
  v2: force-unwind sell_cap=LIMIT+pos=20 when pos=+10 → ±10 oscillation every tick → -1.18M
  v3b: cumulative OLS + MM-mode losses → flat products lose on spread → ~+12k

ROOT CAUSE OF MM LOSSES (v3b):
  When target=0, passive bid+1/ask-1 creates inventory that goes wrong-way.
  PEBBLES_M (range=1500), GALAXY_SOLAR_FLAMES etc. drift directionally even though
  OLS can't detect it, so passive MM accumulates wrong-side inventory.

V3 DESIGN (this version):
  1. Primary signal: DUAL-EMA CROSSOVER (fast vs slow)
       - fast_ema (α=0.03, half-life≈2300ts): tracks recent prices
       - slow_ema (α=0.003, half-life≈23000ts): tracks medium-term level
       - signal = fast_ema - slow_ema
       - |signal| > STRONG_THRESH → full ±LIMIT position
       - |signal| > WEAK_THRESH   → half ±HALF_POS position
       - |signal| ≤ WEAK_THRESH   → FLAT (do nothing, no MM)
  2. PEBBLES_XL: constraint fair (50000 − others) used as taker/MM overlay
  3. Approach target AGGRESSIVELY: take best ask/bid to reach target quickly
     (passive quote as backup; NO passive on the wrong side)
  4. Circuit breaker: if adverse move > DRAWDOWN_PTS from best price → flatten,
     cooldown for COOLDOWN_TICKS before re-entering
  5. NO separate force-unwind: approach_target naturally unwinds toward target=0
     in small steps (RAMP_PER_TICK cap), eliminating the ±10 oscillation bug

EXPECTED PnL:
  Products with large intraday moves (+/- 1000-4000 pts on Day 4):
    PEBBLES_XL (+4014), OXYGEN_GARLIC (+1958), MICROCHIP_CIRCLE (+1854),
    MICROCHIP_OVAL (-1898), UV_VISOR_YELLOW (-1986), etc.
  At 10 units each: +15k to +40k per product → potential 100-200k total
"""

import json
import math
from typing import Dict, List, Optional, Tuple

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

    # ── Dual-EMA parameters ──────────────────────────────────────────────
    # fast EMA: α=0.03, half-life ≈ 23 ticks ≈ 2300 timestamps
    FAST_ALPHA = 0.03
    # slow EMA: α=0.003, half-life ≈ 231 ticks ≈ 23100 timestamps
    SLOW_ALPHA = 0.003

    # EMA spread thresholds (absolute price difference fast_ema - slow_ema)
    # Math: for slope s pt/step, steady-state spread = s*(1/SLOW_A - 1/FAST_A) = s*300.
    # A 500-pt/day move → spread reaches 60 by ts≈15000 (150 steps).
    # A 200-pt/day move → spread stays ~30 → only weak signal; flat (<100pt) → noise.
    SIGNAL_STRONG = 60   # |signal| > 60 pts → ±LIMIT (catches 500+ pt/day moves)
    SIGNAL_WEAK   = 25   # |signal| > 25 pts → ±HALF_POS
    # Below SIGNAL_WEAK: FLAT (no orders at all — prevents MM losses)

    # EMA warm-up: ignore first N timestamps to let EMAs initialise
    WARMUP_TICKS = 3000  # ts=3000 before using EMA signal (30 data points)

    # ── Circuit breaker ──────────────────────────────────────────────────
    DRAWDOWN_PTS   = 350  # adverse move from best-price before exit
    COOLDOWN_TICKS = 5000 # timestamps before re-entering after circuit trip

    # ── Order sizing ─────────────────────────────────────────────────────
    PASSIVE_SIZE  = 2     # max units per passive (backup) quote
    RAMP_PER_TICK = 5     # max units taken aggressively per tick

    # ── PEBBLES_XL constraint ─────────────────────────────────────────────
    PEBBLES_SUM    = 50000
    PEBBLES_XL     = "PEBBLES_XL"
    PEBBLES_NON_XL = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L"]
    XL_OFFSET      = 3    # MM offset around constraint fair
    XL_TAKE_EDGE   = 4    # take-edge for constraint mispricings

    # ── Main entry point ─────────────────────────────────────────────────
    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        conversions = 0
        try:
            td: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        ts = state.timestamp

        # 1. Compute current mid prices
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

        # 3. Update dual-EMAs for all products
        for prod, mid in mids.items():
            self._update_ema(prod, mid, td)

        # 4. Generate orders per product
        for prod, od in state.order_depths.items():
            pos = state.position.get(prod, 0)
            result = self._trade(prod, od, pos, mids, td, ts)
            orders[prod] = result

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

        # Constraint fair value
        others = [mids.get(p) for p in self.PEBBLES_NON_XL]
        if any(v is None for v in others):
            return result
        cf   = self.PEBBLES_SUM - sum(others)   # type: ignore
        cf_r = round(cf)

        # EMA trend bias (determines which side of MM to emphasize)
        ema_target = self._ema_target(prod, td, ts)

        # Taker: capture clear constraint mispricings
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
        # Suppress buying if EMA says strongly short, and vice versa
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
        # Compute raw EMA target
        raw_target = self._ema_target(prod, td, ts)

        # Apply circuit breaker
        target = self._circuit_breaker(prod, raw_target, pos, mids.get(prod), td, ts)

        # Approach target (or stay flat with no orders if target=0)
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

        # Still in cooldown?
        cb_until = td.get(f"cbu_{prod}", 0)
        if ts < cb_until:
            return 0  # Flat during cooldown

        # Reset best-price when target direction changes
        prev_dir = td.get(f"cbd_{prod}", 0)
        if raw_target != prev_dir:
            td[f"cbp_{prod}"] = mid   # reset best price
        td[f"cbd_{prod}"] = raw_target

        if raw_target > 0:   # Long target: track max price
            best = td.get(f"cbp_{prod}", mid)
            if mid > best:
                td[f"cbp_{prod}"] = mid
                best = mid
            if best - mid > self.DRAWDOWN_PTS:
                td[f"cbu_{prod}"] = ts + self.COOLDOWN_TICKS
                td[f"cbp_{prod}"] = mid
                td[f"cbd_{prod}"] = 0
                return 0

        elif raw_target < 0:  # Short target: track min price
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
        """
        If target == 0: return nothing (NO passive MM — prevents mm-loss bleed).
        If target != pos: take liquidity aggressively + passive backup.
        If pos == target != 0: hold (no orders needed).
        """
        if target == 0 and pos == 0:
            return []  # Flat and want flat: do nothing

        result: List[Order] = []
        limit    = self.LIMIT
        buy_cap  = limit - pos
        sell_cap = limit + pos

        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if pos < target:
            # Need to BUY
            need = target - pos
            # Aggressive taker
            if best_ask is not None and buy_cap > 0:
                vol = min(-od.sell_orders[best_ask], buy_cap,
                          need, self.RAMP_PER_TICK)
                if vol > 0:
                    result.append(Order(prod, best_ask, vol))
                    buy_cap -= vol
                    need    -= vol
            # Passive backup
            if best_bid is not None and buy_cap > 0 and need > 0:
                result.append(Order(prod, best_bid + 1,
                                    min(buy_cap, self.PASSIVE_SIZE, need)))

        elif pos > target:
            # Need to SELL (includes flattening when target drops to 0)
            need = pos - target
            # Aggressive taker
            if best_bid is not None and sell_cap > 0:
                vol = min(od.buy_orders[best_bid], sell_cap,
                          need, self.RAMP_PER_TICK)
                if vol > 0:
                    result.append(Order(prod, best_bid, -vol))
                    sell_cap -= vol
                    need     -= vol
            # Passive backup
            if best_ask is not None and sell_cap > 0 and need > 0:
                result.append(Order(prod, best_ask - 1,
                                    -min(sell_cap, self.PASSIVE_SIZE, need)))

        # pos == target: hold position, no orders needed
        return result

    # ── EMA update ────────────────────────────────────────────────────────
    def _update_ema(self, prod: str, mid: float, td: dict) -> None:
        fa = self.FAST_ALPHA
        sa = self.SLOW_ALPHA
        fe = td.get(f"fe_{prod}", mid)
        se = td.get(f"se_{prod}", mid)
        td[f"fe_{prod}"] = fa * mid + (1 - fa) * fe
        td[f"se_{prod}"] = sa * mid + (1 - sa) * se

    # ── Reset on day change ───────────────────────────────────────────────
    def _reset_all(self, td: dict, products) -> None:
        for prod in products:
            for pfx in ["fe_", "se_", "cbu_", "cbp_", "cbd_"]:
                td.pop(f"{pfx}{prod}", None)

    # ── Utility ───────────────────────────────────────────────────────────
    @staticmethod
    def _mid(od: OrderDepth) -> Optional[float]:
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
        return None
