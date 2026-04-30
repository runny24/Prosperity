"""
Round 5 v5
==========
v4 plus one more controlled increase in standalone snack alpha.

The submitted v4 log reached 34.4k and all three snack names remained positive.
Local full-day sweeps improved again when the snack limit was raised from 8 to
10: the worst full-day replay rose above 63k and drawdown improved versus v4.

Shape:
- original baseline EMA roster
- original PEBBLES_XL constraint arb
- no broad microstructure gating on EMA products
- microstructure-only trades for SNACKPACK_PISTACHIO, SNACKPACK_STRAWBERRY,
  and SNACKPACK_VANILLA
- snack limit increased from 8 to 10
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

    # ── Whitelist ────────────────────────────────────────────────────────
    WHITELIST = {
        # PEBBLES_XL: constraint arb + EMA (structural, not trend-dependent)
        "PEBBLES_XL",
        # EMA trend products — α ≥ 0.0006 (adapts within ~170 ticks, robust online)
        "OXYGEN_SHAKE_MORNING_BREATH",   # α=0.002  | online +799
        "ROBOT_MOPPING",                 # α=0.0006 | online +2012
        "MICROCHIP_CIRCLE",              # α=0.0006 | online -730  (borderline, keep)
        "SLEEP_POD_COTTON",              # α=0.002  | online +5130
        "GALAXY_SOUNDS_BLACK_HOLES",     # α=0.0008 | online -104  (borderline, keep)
        "GALAXY_SOUNDS_PLANETARY_RINGS", # α=0.0008 | online +5526
        "MICROCHIP_OVAL",                # α=0.0008 | online -503  (borderline, keep)
        "ROBOT_IRONING",                 # α=0.0006 | online +4040
        "PANEL_1X4",                     # α=0.0006 | online +1829
        # α=0.0004: UV_VISOR_AMBER confirmed positive on 4/4 days (3 local + 1 online)
        "UV_VISOR_AMBER",                # α=0.0004 | online +2979
        # α=0.0002: TRANSLATOR positive online despite slow alpha — genuine signal
        "TRANSLATOR_GRAPHITE_MIST",      # α=0.0002 | online +2091
    }

    MICRO_ONLY = {
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_VANILLA",
    }

    TRADED_PRODUCTS = WHITELIST | MICRO_ONLY

    EMA_NEEDS_MICRO_CONFIRM = set()

    # ── Dual-EMA parameters ──────────────────────────────────────────────
    FAST_ALPHA = 0.02
    SLOW_ALPHA = 0.001

    # Per-product slow alpha overrides
    SLOW_ALPHA_OVERRIDE: Dict[str, float] = {
        "SLEEP_POD_COTTON":              0.002,
        "OXYGEN_SHAKE_MORNING_BREATH":   0.002,
        "ROBOT_MOPPING":                 0.0006,
        "GALAXY_SOUNDS_BLACK_HOLES":     0.0008,
        "MICROCHIP_CIRCLE":              0.0006,
        "GALAXY_SOUNDS_PLANETARY_RINGS": 0.0008,
        "MICROCHIP_OVAL":                0.0008,
        "ROBOT_IRONING":                 0.0006,
        "PANEL_1X4":                     0.0006,
        "UV_VISOR_AMBER":                0.0004,
        "TRANSLATOR_GRAPHITE_MIST":      0.0002,
    }

    SIGNAL_STRONG = 55
    SIGNAL_WEAK   = 20
    WARMUP_TICKS  = 3000

    # ── Circuit breaker ──────────────────────────────────────────────────
    DRAWDOWN_PTS   = 600
    COOLDOWN_TICKS = 2000

    # ── Order sizing ─────────────────────────────────────────────────────
    PASSIVE_SIZE  = 2
    RAMP_PER_TICK = 5

    # ── Microstructure overlay ───────────────────────────────────────────
    MICRO_CONFIRM = 0.04
    MICRO_BLOCK   = 0.12
    MICRO_ENTRY   = 0.18
    MICRO_TAKE    = 0.30
    SNACK_LIMIT   = 10

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

        # 4. Generate orders — only for traded products
        for prod, od in state.order_depths.items():
            if prod not in self.TRADED_PRODUCTS:
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
        if prod in self.MICRO_ONLY:
            return self._trade_micro_only(prod, od, pos)
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
        micro_score = self._micro_score(od)

        if (buy_cap > 0 and pos < buy_soft and mm_buy < cf
                and (micro_score is None or micro_score > -self.MICRO_BLOCK)):
            result.append(Order(prod, mm_buy, min(buy_cap, self.PASSIVE_SIZE)))
        if (sell_cap > 0 and pos > -sell_soft and mm_sell > cf
                and (micro_score is None or micro_score < self.MICRO_BLOCK)):
            result.append(Order(prod, mm_sell, -min(sell_cap, self.PASSIVE_SIZE)))

        return result

    # ── General: EMA crossover trend following ────────────────────────────
    def _trade_ema(self, prod: str, od: OrderDepth, pos: int,
                   mids: Dict[str, float], td: dict, ts: int) -> List[Order]:
        raw_target = self._ema_target(prod, td, ts)
        target = self._circuit_breaker(prod, raw_target, pos, mids.get(prod), td, ts)
        return self._approach_target(prod, od, pos, target)

    # ── SNACKPACK: short-horizon microstructure alpha only ────────────────
    def _trade_micro_only(self, prod: str, od: OrderDepth, pos: int) -> List[Order]:
        score = self._micro_score(od)
        if score is None:
            return []

        if score >= self.MICRO_ENTRY:
            target = self.SNACK_LIMIT
        elif score <= -self.MICRO_ENTRY:
            target = -self.SNACK_LIMIT
        elif abs(score) <= self.MICRO_CONFIRM:
            target = 0
        else:
            target = pos

        allow_take = abs(score) >= self.MICRO_TAKE
        return self._approach_target(prod, od, pos, target, allow_take=allow_take)

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
                         pos: int, target: int,
                         allow_take: bool = True) -> List[Order]:
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
            if allow_take and best_ask is not None and buy_cap > 0:
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
            if allow_take and best_bid is not None and sell_cap > 0:
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

    # ── Microstructure helpers ────────────────────────────────────────────
    def _micro_adjusted_target(self, prod: str, od: OrderDepth,
                               pos: int, target: int) -> int:
        score = self._micro_score(od)
        if target == 0 or score is None:
            return target

        need_confirm = prod in self.EMA_NEEDS_MICRO_CONFIRM
        if target > 0:
            if need_confirm and score < self.MICRO_CONFIRM:
                return 0
            if score < -self.MICRO_BLOCK:
                return 0 if pos <= 0 else min(pos, self.HALF_POS)
            if score < 0:
                return min(target, self.HALF_POS)
        else:
            if need_confirm and score > -self.MICRO_CONFIRM:
                return 0
            if score > self.MICRO_BLOCK:
                return 0 if pos >= 0 else max(pos, -self.HALF_POS)
            if score > 0:
                return max(target, -self.HALF_POS)
        return target

    def _micro_allows_take(self, od: OrderDepth, desired_change: int) -> bool:
        score = self._micro_score(od)
        if score is None or desired_change == 0:
            return True
        if desired_change > 0:
            return score >= self.MICRO_CONFIRM
        return score <= -self.MICRO_CONFIRM

    def _micro_score(self, od: OrderDepth) -> Optional[float]:
        f = self._book_features(od)
        if f is None:
            return None
        # Notebook ICs: L1 imbalance and microprice are same-direction;
        # L3 imbalance is consistently opposite-direction.
        return (
            f["imbalance_1"]
            - f["imbalance_3"]
            + 0.25 * f["micro_edge_ticks"]
        )

    @staticmethod
    def _book_features(od: OrderDepth) -> Optional[Dict[str, float]]:
        if not od.buy_orders or not od.sell_orders:
            return None

        bid_prices = sorted(od.buy_orders.keys(), reverse=True)
        ask_prices = sorted(od.sell_orders.keys())
        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        bid_vol_1 = max(0, od.buy_orders[best_bid])
        ask_vol_1 = max(0, -od.sell_orders[best_ask])
        depth_1 = bid_vol_1 + ask_vol_1
        if depth_1 <= 0:
            return None

        bid_depth_3 = sum(max(0, od.buy_orders[p]) for p in bid_prices[:3])
        ask_depth_3 = sum(max(0, -od.sell_orders[p]) for p in ask_prices[:3])
        depth_3 = bid_depth_3 + ask_depth_3
        if depth_3 <= 0:
            return None

        mid = (best_bid + best_ask) / 2.0
        micro = (best_ask * bid_vol_1 + best_bid * ask_vol_1) / depth_1
        return {
            "mid": mid,
            "spread": best_ask - best_bid,
            "imbalance_1": (bid_vol_1 - ask_vol_1) / depth_1,
            "imbalance_3": (bid_depth_3 - ask_depth_3) / depth_3,
            "micro_edge_ticks": micro - mid,
        }

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
