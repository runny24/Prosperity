"""
Round 5 Strategy — v1
=======================
Key findings from data analysis (Days 2-4, 30000 ticks each):

GROUP MECHANICS (critical):
  1. PEBBLES: Hard constraint sum(XS+S+M+L+XL) = 50000 ± 3.
     XL is the "residual": XL_fair = 50000 − XS − S − M − L.
     Within-tick return corr: XL vs each other = −0.5; others are independent.
     → Use constraint as XL fair value (much tighter than EMA).

  2. SNACKPACK: Two mechanical anti-correlated pairs (1-tick return corr ≈ −0.92):
     • CHOC ↔ VAN  (sum ≈ 19941, std 76)
     • RASP ↔ PISTACHIO (sum ≈ 19574, std 180)
     STRAWBERRY is the independent "driver" of the RASP/STRAW/PIST triplet.
     → Use pair-sum EMA as fair value for both legs of each pair.

  3. ALL OTHER GROUPS (UV_VISOR, PANEL, ROBOT, SLEEP_POD, OXYGEN_SHAKE,
     MICROCHIP, GALAXY_SOUNDS, TRANSLATOR):
     Tick-by-tick correlations ≈ 0 (independent random walks within group).
     Long-run correlations exist but are purely trend-driven, not mechanical.
     → Standard EMA market-making per product.

STRATEGIES:
  A. EMA Market Making (all 50 products)
     - Fair = EMA(mid, α=0.10)
     - Quote bid=fair−offset, ask=fair+offset inside existing spread
     - Aggressive take when price beats fair by TAKE_EDGE

  B. PEBBLES_XL constraint (precision edge)
     - XL_fair = 50000 − mid(XS) − mid(S) − mid(M) − mid(L)
     - Quote tighter (offset=3) since fair is ~exact; XL market spread is 16pt

  C. SNACKPACK pair-constrained fair value
     - For CHOC: fair = EMA(CHOC+VAN) − VAN_mid
     - For VAN:  fair = EMA(CHOC+VAN) − CHOC_mid
     - For RASP: fair = EMA(RASP+PIST) − PIST_mid
     - For PIST: fair = EMA(RASP+PIST) − RASP_mid
     - Hedged inventory: when long CHOC, go short VAN (they offset each other)

TRANSFER FROM PREVIOUS ROUNDS:
  - EMA MM core loop ← ACO strategy, Rounds 1 & 2 (near-zero approach)
  - Adaptive offset (widens when position accumulates) ← R1/R2
  - Take-on-extreme ← HYDROGEL/VELVETFRUIT extreme module, R3/R4
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)


# ── Logger ────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self): self.logs = ""; self.max_log_length = 3750

    def print(self, *objects, sep=" ", end="\n"):
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state, orders, conversions, trader_data):
        bl = len(self.to_json([self.cs(state, ""), self.co(orders), conversions, "", ""]))
        m = (self.max_log_length - bl) // 3
        print(self.to_json([
            self.cs(state, self.t(state.traderData, m)),
            self.co(orders), conversions,
            self.t(trader_data, m), self.t(self.logs, m)
        ]))
        self.logs = ""

    def cs(self, s, td):
        return [s.timestamp, td,
                [[l.symbol, l.product, l.denomination] for l in s.listings.values()],
                {k: [v.buy_orders, v.sell_orders] for k, v in s.order_depths.items()},
                [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                 for a in s.own_trades.values() for t in a],
                [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                 for a in s.market_trades.values() for t in a],
                s.position, self.cobs(s.observations)]

    def cobs(self, obs):
        co = {}
        for p, o in obs.conversionObservations.items():
            co[p] = [o.bidPrice, o.askPrice, o.transportFees,
                     o.exportTariff, o.importTariff, o.sunlight, o.humidity]
        return [obs.plainValueObservations, co]

    def co(self, orders):
        return [[o.symbol, o.price, o.quantity] for a in orders.values() for o in a]

    def to_json(self, v):
        return json.dumps(v, cls=ProsperityEncoder, separators=(",", ":"))

    def t(self, v, m):
        return v if len(v) <= m else v[:m - 3] + "..."


logger = Logger()


# ── Trader ────────────────────────────────────────────────────────────────────
class Trader:

    # ── Hard position limit per product ───────────────────────────────────────
    LIMIT = 10

    # ── EMA parameters ────────────────────────────────────────────────────────
    EMA_ALPHA       = 0.10   # default for all products
    EMA_ALPHA_PAIR  = 0.05   # slower alpha for pair-sum EMA (more stable fair value)

    # ── Market-making offsets ─────────────────────────────────────────────────
    # Distance from fair to passive quote (in price units).
    # Wider = safer inventory, fewer fills. Narrower = more fills, more inventory risk.
    MM_OFFSET_DEFAULT  = 4   # standard for most products
    MM_OFFSET_XL       = 3   # PEBBLES_XL: constraint gives tight fair → quote tighter
    MM_OFFSET_PAIR     = 3   # SNACKPACK pairs: hedged so can quote tight
    MM_OFFSET_STRAWBERRY = 4 # SNACKPACK_STRAWBERRY: independent (no pair constraint)
    MM_OFFSET_WIDE     = 5   # wider (large-spread products like SNACKPACK/GALAXY)

    # Extra offset added per unit of |position| approaching limit.
    # At pos=5/10, add +1; at pos=8/10, add +1.6
    MM_SKEW_PER_UNIT = 0.2

    # ── Taking edge ───────────────────────────────────────────────────────────
    # Take aggressively when price is this many units better than fair.
    TAKE_EDGE_DEFAULT = 8    # aggressive take threshold
    TAKE_EDGE_XL      = 4    # XL constraint is tight, take at smaller edge
    TAKE_EDGE_PAIR    = 5    # SNACKPACK pairs

    # ── PEBBLES_XL constraint ─────────────────────────────────────────────────
    PEBBLES_SUM       = 50000
    PEBBLES_XL        = "PEBBLES_XL"
    PEBBLES_NON_XL    = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L"]

    # ── SNACKPACK pair structure ───────────────────────────────────────────────
    # CHOC ↔ VAN:  CHOC + VAN ≈ 19941
    # RASP ↔ PIST: RASP + PIST ≈ 19574
    # Each key maps to its partner and the historical mean of their sum.
    # Sum EMA key = "psum_<partnerA>_<partnerB>" (alphabetical).
    SNACK_PAIRS: Dict[str, Tuple[str, float]] = {
        "SNACKPACK_CHOCOLATE": ("SNACKPACK_VANILLA",   19941.0),
        "SNACKPACK_VANILLA":   ("SNACKPACK_CHOCOLATE", 19941.0),
        "SNACKPACK_RASPBERRY": ("SNACKPACK_PISTACHIO", 19574.0),
        "SNACKPACK_PISTACHIO": ("SNACKPACK_RASPBERRY", 19574.0),
    }
    SNACK_STRAWBERRY = "SNACKPACK_STRAWBERRY"  # independent

    # ── Per-product offset overrides ──────────────────────────────────────────
    # Products with wide native spreads benefit from a slightly wider offset.
    WIDE_SPREAD_PRODUCTS = {
        "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA",
        "SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY", "SNACKPACK_PISTACHIO",
        "GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_SOLAR_FLAMES",
        "GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_WINDS",
        "MICROCHIP_SQUARE", "PEBBLES_XL",
    }

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            td: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        # ── Step 1: collect all current mid-prices ───────────────────────────
        mids: Dict[str, float] = {}
        for prod, od in state.order_depths.items():
            m = self._mid(od)
            if m is not None:
                mids[prod] = m

        # ── Step 2: update EMAs for all products ─────────────────────────────
        for prod, mid in mids.items():
            k = f"e_{prod}"
            td[k] = self.EMA_ALPHA * mid + (1 - self.EMA_ALPHA) * td.get(k, mid)

        # ── Step 3: update pair-sum EMAs for SNACKPACK pairs ─────────────────
        for prod, (partner, _) in self.SNACK_PAIRS.items():
            if prod < partner:  # only once per pair (alphabetical guard)
                if prod in mids and partner in mids:
                    current_sum = mids[prod] + mids[partner]
                    k = f"psum_{prod}_{partner}"
                    td[k] = (self.EMA_ALPHA_PAIR * current_sum
                              + (1 - self.EMA_ALPHA_PAIR) * td.get(k, current_sum))

        # ── Step 4: compute fair values & generate orders ─────────────────────
        for prod, od in state.order_depths.items():
            pos = state.position.get(prod, 0)
            fair = self._fair_value(prod, mids, td)
            if fair is None:
                continue

            take_edge, mm_offset = self._get_params(prod)
            result = self._make_orders(prod, od, pos, fair, take_edge, mm_offset)
            orders[prod] = result

        tdo = json.dumps(td)
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ── Fair value calculation ────────────────────────────────────────────────
    def _fair_value(self, prod: str, mids: Dict[str, float], td: dict) -> Optional[float]:
        """
        Returns the best estimate of fair value for the product.
        Priority:
          1. PEBBLES_XL  → exact constraint (50000 − sum of others)
          2. SNACKPACK pairs → pair-sum EMA minus partner's mid
          3. All others → simple EMA of own mid
        """
        # 1. PEBBLES_XL: use hard constraint
        if prod == self.PEBBLES_XL:
            other_mids = [mids.get(p) for p in self.PEBBLES_NON_XL]
            if all(v is not None for v in other_mids):
                return self.PEBBLES_SUM - sum(other_mids)
            # Fallback: EMA
            return td.get(f"e_{prod}")

        # 2. SNACKPACK pairs: pair-sum EMA − partner_mid
        if prod in self.SNACK_PAIRS:
            partner, init_sum = self.SNACK_PAIRS[prod]
            # Pair-sum EMA key
            a, b = (prod, partner) if prod < partner else (partner, prod)
            sum_ema = td.get(f"psum_{a}_{b}", init_sum)
            partner_mid = mids.get(partner)
            if partner_mid is not None:
                return sum_ema - partner_mid
            # Fallback: own EMA
            return td.get(f"e_{prod}")

        # 3. Default: own EMA
        return td.get(f"e_{prod}")

    # ── Per-product parameters ────────────────────────────────────────────────
    def _get_params(self, prod: str) -> Tuple[float, float]:
        """Returns (take_edge, mm_offset) for a product."""
        if prod == self.PEBBLES_XL:
            return self.TAKE_EDGE_XL, self.MM_OFFSET_XL
        if prod in self.SNACK_PAIRS:
            return self.TAKE_EDGE_PAIR, self.MM_OFFSET_PAIR
        if prod in self.WIDE_SPREAD_PRODUCTS:
            return self.TAKE_EDGE_DEFAULT, self.MM_OFFSET_WIDE
        return self.TAKE_EDGE_DEFAULT, self.MM_OFFSET_DEFAULT

    # ── Order placement ───────────────────────────────────────────────────────
    def _make_orders(
        self, prod: str, od: OrderDepth,
        pos: int, fair: float,
        take_edge: float, mm_offset: float,
    ) -> List[Order]:
        """
        Places both taker and maker orders around the fair value.

        Taker  : cross the spread when price beats fair by ≥ take_edge.
        Maker  : passive quote inside existing spread at fair ± adjusted_offset.

        Inventory skew: offset widens linearly as |position| grows, which
        makes passive quotes on the crowded side harder to fill and naturally
        keeps inventory near zero.
        """
        result: List[Order] = []
        limit = self.LIMIT
        buy_cap  = limit - pos   # units we can still buy
        sell_cap = limit + pos   # units we can still sell

        # ── Inventory skew on passive offset ─────────────────────────────────
        # Positive pos → widen sell offset (harder to sell more), keep buy offset
        # Negative pos → widen buy offset, keep sell offset
        skew = abs(pos) * self.MM_SKEW_PER_UNIT
        buy_offset  = mm_offset + (skew if pos > 0 else 0)
        sell_offset = mm_offset + (skew if pos < 0 else 0)

        fair_r = round(fair)

        # ── Taker: buy cheap ─────────────────────────────────────────────────
        for ask_px in sorted(od.sell_orders.keys()):
            if ask_px < fair - take_edge and buy_cap > 0:
                vol = min(-od.sell_orders[ask_px], buy_cap)
                result.append(Order(prod, ask_px, vol))
                buy_cap -= vol
            else:
                break

        # ── Taker: sell expensive ─────────────────────────────────────────────
        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if bid_px > fair + take_edge and sell_cap > 0:
                vol = min(od.buy_orders[bid_px], sell_cap)
                result.append(Order(prod, bid_px, -vol))
                sell_cap -= vol
            else:
                break

        # ── Maker: passive inside spread ──────────────────────────────────────
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        # Penny-inside: quote one tick better than current best if room allows
        mm_buy  = fair_r - int(math.ceil(buy_offset))
        mm_sell = fair_r + int(math.ceil(sell_offset))

        if best_bid is not None and best_bid + 1 < fair:
            mm_buy = max(mm_buy, best_bid + 1)
        if best_ask is not None and best_ask - 1 > fair:
            mm_sell = min(mm_sell, best_ask - 1)

        # Only quote if we have capacity and quotes are on the right side of fair
        if buy_cap > 0 and mm_buy < fair:
            result.append(Order(prod, mm_buy, buy_cap))
        if sell_cap > 0 and mm_sell > fair:
            result.append(Order(prod, mm_sell, -sell_cap))

        return result

    # ── Utility ───────────────────────────────────────────────────────────────
    @staticmethod
    def _mid(od: OrderDepth) -> Optional[float]:
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
        return None
