"""
Round 5 v2 — Fix stale-EMA blowup
====================================
ROOT CAUSE OF v1 FAILURE (-297k loss):
  1. max() BUG: mm_buy = max(EMA-offset, best_bid+1) posts our bid ABOVE the
     actual market when EMA lags. In trending products (price −2000/day),
     EMA is always high → our bid gets filled at stale prices → we accumulate
     a long position while price keeps falling. MTM loss = 10 × drift.
  2. Full-capacity passive: posting all 10 units in one level means a single
     sell order can jam us to the limit immediately.
  3. EMA take misfired: ask < EMA − 8 fired in DOWN-trends (EMA stale-high),
     buying into falling markets.
  4. No unwind: once at limit, no force-exit → stayed stuck forever.

WHAT WORKED (keep unchanged):
  - PEBBLES_XL constraint: +3,521 (only product with positive PnL)
  - SNACKPACK CHOC/VAN pair: near-zero loss (correct fair value keeps pos ≤ 3)

V2 CHANGES:
  1. Passive quotes = ALWAYS best_bid+1 / best_ask-1 (zero EMA dependency)
  2. Soft limit = 5: gate passive on crowded side; still allow taker up to 10
  3. Quote size = 2 units max (not entire remaining cap)
  4. Force-unwind: if |pos| ≥ 8, cross spread to exit (take opposite side)
  5. Higher take_edge (14 default) — only take on extreme dislocations
  6. PEBBLES_XL: keep constraint-based fair → quote at fair±3 (worked in v1)
  7. SNACKPACK pairs: keep pair-sum EMA fair → quote at fair±3 (worked in v1)
  8. All others: book-based passive + high-threshold taker (EMA still used
     for taking but only at large deviations)
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
    # ── Hard position limit ───────────────────────────────────────────────
    LIMIT = 10

    # ── Inventory gates ───────────────────────────────────────────────────
    # Passive quotes suppressed on crowded side when |pos| ≥ SOFT_LIMIT
    SOFT_LIMIT = 5
    # Force-cross the spread to unwind when |pos| ≥ UNWIND_LIMIT
    UNWIND_LIMIT = 8
    # Max units per single passive quote order
    PASSIVE_SIZE = 2

    # ── EMA (used for TAKER only on generic products) ─────────────────────
    EMA_ALPHA      = 0.10   # standard EMA for individual products
    EMA_ALPHA_PAIR = 0.05   # slower: SNACKPACK pair-sum (more stable)

    # ── Taking thresholds ─────────────────────────────────────────────────
    # Generic products: only take on large dislocations (EMA still lags →
    # keep threshold wide enough that we don't fire on trend-driven moves)
    TAKE_EDGE_GENERIC = 14

    # Constraint / pair products: fair is precise so threshold can be tight
    TAKE_EDGE_XL   = 4    # PEBBLES_XL (constraint accurate to ±3)
    TAKE_EDGE_PAIR = 6    # SNACKPACK pairs (pair-sum std ≈ 76 over all ticks)

    # ── PEBBLES_XL constraint ─────────────────────────────────────────────
    PEBBLES_SUM    = 50000
    PEBBLES_XL     = "PEBBLES_XL"
    PEBBLES_NON_XL = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L"]

    # MM offset for constraint/pair products (tight because fair is accurate)
    MM_OFFSET_PRECISE = 3

    # ── SNACKPACK pairs ───────────────────────────────────────────────────
    # (product → (partner, historical_sum_mean))
    SNACK_PAIRS: Dict[str, Tuple[str, float]] = {
        "SNACKPACK_CHOCOLATE": ("SNACKPACK_VANILLA",   19941.0),
        "SNACKPACK_VANILLA":   ("SNACKPACK_CHOCOLATE", 19941.0),
        "SNACKPACK_RASPBERRY": ("SNACKPACK_PISTACHIO", 19574.0),
        "SNACKPACK_PISTACHIO": ("SNACKPACK_RASPBERRY", 19574.0),
    }

    # ── Main entry point ─────────────────────────────────────────────────
    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        conversions = 0
        try:
            td: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        # 1. Collect mid prices and update EMAs
        mids: Dict[str, float] = {}
        for prod, od in state.order_depths.items():
            m = self._mid(od)
            if m is not None:
                mids[prod] = m
                k = f"e_{prod}"
                td[k] = self.EMA_ALPHA * m + (1 - self.EMA_ALPHA) * td.get(k, m)

        # 2. Update SNACKPACK pair-sum EMAs
        for prod, (partner, init_sum) in self.SNACK_PAIRS.items():
            if prod < partner and prod in mids and partner in mids:
                current_sum = mids[prod] + mids[partner]
                k = f"psum_{prod}_{partner}"
                td[k] = (self.EMA_ALPHA_PAIR * current_sum
                          + (1 - self.EMA_ALPHA_PAIR) * td.get(k, current_sum))

        # 3. Generate orders for each product
        for prod, od in state.order_depths.items():
            pos = state.position.get(prod, 0)
            result = self._trade(prod, od, pos, mids, td)
            orders[prod] = result

        tdo = json.dumps(td)
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ── Per-product order logic ───────────────────────────────────────────
    def _trade(self, prod: str, od: OrderDepth, pos: int,
               mids: Dict[str, float], td: dict) -> List[Order]:
        result: List[Order] = []

        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        limit    = self.LIMIT
        buy_cap  = limit - pos
        sell_cap = limit + pos

        # ── FORCE UNWIND: position too extreme ────────────────────────────
        # Cross the spread aggressively to reduce inventory.
        # This caps maximum loss from a stuck position.
        if pos >= self.UNWIND_LIMIT and best_bid is not None and sell_cap > 0:
            # Sell immediately at best_bid
            qty = min(od.buy_orders[best_bid], sell_cap)
            result.append(Order(prod, best_bid, -qty))
            sell_cap -= qty
        elif pos <= -self.UNWIND_LIMIT and best_ask is not None and buy_cap > 0:
            qty = min(-od.sell_orders[best_ask], buy_cap)
            result.append(Order(prod, best_ask, qty))
            buy_cap -= qty

        # Recompute after unwind
        buy_cap  = limit - pos - sum(o.quantity for o in result if o.quantity > 0)
        sell_cap = limit + pos + sum(o.quantity for o in result if o.quantity < 0)

        # ── Decide strategy based on product type ─────────────────────────
        if prod == self.PEBBLES_XL:
            self._trade_xl(prod, od, pos, buy_cap, sell_cap, mids, result)

        elif prod in self.SNACK_PAIRS:
            self._trade_pair(prod, od, pos, buy_cap, sell_cap, mids, td, result)

        else:
            self._trade_generic(prod, od, pos, buy_cap, sell_cap, mids, td, result)

        return result

    # ── PEBBLES_XL: constraint-based fair value ────────────────────────────
    def _trade_xl(self, prod, od, pos, buy_cap, sell_cap, mids, result):
        others = [mids.get(p) for p in self.PEBBLES_NON_XL]
        if any(v is None for v in others):
            return
        fair = self.PEBBLES_SUM - sum(others)
        fair_r = round(fair)
        te = self.TAKE_EDGE_XL
        offset = self.MM_OFFSET_PRECISE

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        # Taker: cross spread when clearly mispriced vs constraint
        for ask_px in sorted(od.sell_orders.keys()):
            if ask_px < fair - te and buy_cap > 0 and pos < self.SOFT_LIMIT:
                vol = min(-od.sell_orders[ask_px], buy_cap)
                result.append(Order(prod, ask_px, vol))
                buy_cap -= vol
            else: break

        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if bid_px > fair + te and sell_cap > 0 and pos > -self.SOFT_LIMIT:
                vol = min(od.buy_orders[bid_px], sell_cap)
                result.append(Order(prod, bid_px, -vol))
                sell_cap -= vol
            else: break

        # Passive: quote at constraint fair ± offset (very tight)
        mm_buy  = fair_r - offset
        mm_sell = fair_r + offset
        if best_bid and best_bid + 1 < fair: mm_buy  = best_bid  + 1
        if best_ask and best_ask - 1 > fair: mm_sell = best_ask - 1

        if buy_cap > 0 and pos < self.SOFT_LIMIT and mm_buy < fair:
            result.append(Order(prod, mm_buy, min(buy_cap, self.PASSIVE_SIZE)))
        if sell_cap > 0 and pos > -self.SOFT_LIMIT and mm_sell > fair:
            result.append(Order(prod, mm_sell, -min(sell_cap, self.PASSIVE_SIZE)))

    # ── SNACKPACK pairs: pair-sum EMA fair value ──────────────────────────
    def _trade_pair(self, prod, od, pos, buy_cap, sell_cap, mids, td, result):
        partner, init_sum = self.SNACK_PAIRS[prod]
        a, b = (prod, partner) if prod < partner else (partner, prod)
        sum_ema = td.get(f"psum_{a}_{b}", init_sum)
        partner_mid = mids.get(partner)
        if partner_mid is None:
            return self._trade_generic(prod, od, pos, buy_cap, sell_cap, mids, td, result)

        fair   = sum_ema - partner_mid
        fair_r = round(fair)
        te     = self.TAKE_EDGE_PAIR
        offset = self.MM_OFFSET_PRECISE

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        # Taker
        for ask_px in sorted(od.sell_orders.keys()):
            if ask_px < fair - te and buy_cap > 0 and pos < self.SOFT_LIMIT:
                vol = min(-od.sell_orders[ask_px], buy_cap)
                result.append(Order(prod, ask_px, vol))
                buy_cap -= vol
            else: break
        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if bid_px > fair + te and sell_cap > 0 and pos > -self.SOFT_LIMIT:
                vol = min(od.buy_orders[bid_px], sell_cap)
                result.append(Order(prod, bid_px, -vol))
                sell_cap -= vol
            else: break

        # Passive at constraint fair
        mm_buy  = fair_r - offset
        mm_sell = fair_r + offset
        if best_bid and best_bid + 1 < fair: mm_buy  = best_bid  + 1
        if best_ask and best_ask - 1 > fair: mm_sell = best_ask - 1

        if buy_cap > 0 and pos < self.SOFT_LIMIT and mm_buy < fair:
            result.append(Order(prod, mm_buy, min(buy_cap, self.PASSIVE_SIZE)))
        if sell_cap > 0 and pos > -self.SOFT_LIMIT and mm_sell > fair:
            result.append(Order(prod, mm_sell, -min(sell_cap, self.PASSIVE_SIZE)))

    # ── Generic products: book-based passive, EMA taker only ─────────────
    def _trade_generic(self, prod, od, pos, buy_cap, sell_cap, mids, td, result):
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        fair = td.get(f"e_{prod}")  # EMA — used ONLY for taking

        # ── Taker: only on large EMA dislocations ────────────────────────
        if fair is not None:
            te = self.TAKE_EDGE_GENERIC
            for ask_px in sorted(od.sell_orders.keys()):
                if ask_px < fair - te and buy_cap > 0 and pos < self.SOFT_LIMIT:
                    vol = min(-od.sell_orders[ask_px], buy_cap)
                    result.append(Order(prod, ask_px, vol))
                    buy_cap -= vol
                else: break
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if bid_px > fair + te and sell_cap > 0 and pos > -self.SOFT_LIMIT:
                    vol = min(od.buy_orders[bid_px], sell_cap)
                    result.append(Order(prod, bid_px, -vol))
                    sell_cap -= vol
                else: break

        # ── Passive: ALWAYS book-based (best_bid+1 / best_ask-1) ─────────
        # NEVER reference EMA here — avoids stale-price accumulation.
        # Gate on soft limit to prevent building up directional inventory.
        if best_bid is not None and buy_cap > 0 and pos < self.SOFT_LIMIT:
            result.append(Order(prod, best_bid + 1, min(buy_cap, self.PASSIVE_SIZE)))

        if best_ask is not None and sell_cap > 0 and pos > -self.SOFT_LIMIT:
            result.append(Order(prod, best_ask - 1, -min(sell_cap, self.PASSIVE_SIZE)))

    # ── Utility ───────────────────────────────────────────────────────────
    @staticmethod
    def _mid(od: OrderDepth) -> Optional[float]:
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
        return None
