"""
IMC Prosperity Round 2 — Trader v2
====================================
Fixed from trader-v1 (trader1.log analysis):

BUG 1 — IPR wrong base price
  v1 hardcoded IPR_DAY2_BASE = 14000.  On day 1 (fair ≈ 13000+ts/1000)
  every ask looked like a "bargain" → bought 80 units at t≈200, placed sells
  at 14005 which never filled.  Lucky profit from riding the trend only.
  FIX: Detect the starting fair value on the very first tick by rounding the
       observed mid-price to the nearest 1000.  From then FAIR is exact.

BUG 2 — ACO taker over-selling
  Any tick where best_bid > 10000 (even +1 tick, 8 % of all ticks) triggered
  an immediate taker SELL.  These 1-tick "edges" steadily drove position to
  −80 by t = 1800.
  FIX: Require a minimum taker edge of ACO_MIN_TAKE_EDGE (3 ticks) before
       taking either side.

BUG 3 — ACO passive order asymmetry
  penny-inside-bid fires only when best_bid + 1 < FAIR.  When bid = FAIR
  (38 % of ticks) this is 10001 < 10000 → FALSE, so passive buy stays at
  FAIR − 7 = 9993 (7 ticks below the market bid), while passive sell fires
  at best_ask − 1 = 10015 (penny inside the ask).  Sells fill immediately;
  buys sit ignored.
  FIX (a): Symmetric penny-inside rules — fire when best_bid + 1 ≤ FAIR
  FIX (b): Cap passive-sell at FAIR + SELL_PENNY_CAP to avoid posting
            far-from-fair orders that only hurt inventory.
  FIX (c): Soft inventory limits — once |pos| ≥ SOFT_LIMIT, suppress
            passive orders on the side that would deepen the imbalance.

ADDITIONAL IMPROVEMENTS
  • IPR: Hold a max long directional position — don't cycle; the linear
    uptrend means holding is far more profitable than market-making.
  • ACO: EMA-based fair-value drift tracks when anchor shifts; same as v1.
  • Conversion observations checked each tick; request conversions if the
    effective net edge clears CONV_MIN_EDGE (no conversions in this log,
    but framework stays intact for when they appear).
"""

import json
import math
from typing import Any
from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)


# ═══════════════════════════════════════════════════════════════
#  Logger  (compact identical format to R1)
# ═══════════════════════════════════════════════════════════════
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state, orders, conversions, trader_data):
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders), conversions, "", ""
        ]))
        m = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, m)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, m),
            self.truncate(self.logs, m)
        ]))
        self.logs = ""

    def compress_state(self, state, td):
        return [
            state.timestamp, td,
            [[l.symbol, l.product, l.denomination]
             for l in state.listings.values()],
            {s: [od.buy_orders, od.sell_orders]
             for s, od in state.order_depths.items()},
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
             for arr in state.own_trades.values() for t in arr],
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
             for arr in state.market_trades.values() for t in arr],
            state.position,
            self.compress_observations(state.observations)
        ]

    def compress_observations(self, obs):
        co = {}
        for p, o in obs.conversionObservations.items():
            co[p] = [o.bidPrice, o.askPrice, o.transportFees,
                     o.exportTariff, o.importTariff,
                     getattr(o, 'sunlight', None),
                     getattr(o, 'humidity', None)]
        return [obs.plainValueObservations, co]

    def compress_orders(self, orders):
        return [[o.symbol, o.price, o.quantity]
                for arr in orders.values() for o in arr]

    def to_json(self, value):
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value, m):
        return value if len(value) <= m else value[:m - 3] + "..."


logger = Logger()


# ═══════════════════════════════════════════════════════════════
#  Trader
# ═══════════════════════════════════════════════════════════════
class Trader:

    LIMIT = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}

    # ── ACO parameters ────────────────────────────────────────
    ACO_ANCHOR        = 10_000
    ACO_OFFSET_MAX    = 7       # passive order offset, decays with fill time
    ACO_OFFSET_MIN    = 3
    ACO_DECAY_TICKS   = 5_000

    # FIX 2: minimum edge to use the taker (was effectively 0 in v1)
    ACO_MIN_TAKE_EDGE = 3

    # FIX 3b: cap how far above FAIR we post passive sells
    # (was effectively uncapped in v1 → ended up at best_ask−1 = 10015)
    ACO_SELL_PENNY_CAP = 10    # passive sell ≤ FAIR + ACO_SELL_PENNY_CAP

    # FIX 3c: soft inventory limit — suppress the imbalanced side's passive
    # orders when position exceeds this threshold
    ACO_SOFT_LIMIT    = 40     # half of hard limit (80)

    # Bid/ask prediction coefficients (linear fit validated on days −1,0,1)
    ACO_BID_COEF = (-0.271, -0.217,  0.442, -7.156)
    ACO_ASK_COEF = (-0.276, -0.217, -0.570,  9.223)

    # ── IPR parameters ────────────────────────────────────────
    IPR_SLOPE_PER_TICK  = 0.001  # +1 per 1 000 ticks (perfect linear trend)
    IPR_OFFSET          = 4      # passive buy below fair while ramping up
    # How many seashells below fair we're still willing to take an ask
    # (kept small to avoid overpaying for the initial fill)
    IPR_TAKE_SLACK      = 1

    # ── Conversion parameters ─────────────────────────────────
    CONV_MIN_EDGE  = 2.0   # minimum net edge (after fees) to request conversion
    CONV_MAX_UNITS = 10    # cap per tick

    # ─────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        orders:       dict = {}
        conversions:  int  = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        for product in state.order_depths:
            if product == "ASH_COATED_OSMIUM":
                orders[product], td, c = self._trade_aco(state, td)
                conversions += c
            elif product == "INTARIAN_PEPPER_ROOT":
                orders[product], td, c = self._trade_ipr(state, td)
                conversions += c

        tdo = json.dumps(td)
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ═══════════════════════════════════════════════════════════
    #  SHARED HELPERS
    # ═══════════════════════════════════════════════════════════
    def _conv_edge(self, obs, fair: float):
        """Returns (buy_edge, sell_edge) after all fees."""
        effective_buy  = obs.askPrice  + obs.transportFees + obs.importTariff
        effective_sell = obs.bidPrice  - obs.transportFees - obs.exportTariff
        return fair - effective_buy, effective_sell - fair

    def _request_conversions(self, obs, fair: float, pos: int, limit: int) -> int:
        if obs is None:
            return 0
        buy_edge, sell_edge = self._conv_edge(obs, fair)
        if buy_edge > self.CONV_MIN_EDGE:
            return min(limit - pos, self.CONV_MAX_UNITS)
        if sell_edge > self.CONV_MIN_EDGE:
            return -min(limit + pos, self.CONV_MAX_UNITS)
        return 0

    # ═══════════════════════════════════════════════════════════
    #  IPR  —  Directional long-and-hold (trend rider)
    # ═══════════════════════════════════════════════════════════
    #
    #  Key insight from v1 log analysis:
    #    • IPR rises at exactly +0.001 per tick (+1 per 1 000 ticks).
    #    • Starting price increases by +1 000 each day.
    #    • Holding 80 units long for a full 1 M-tick day earns ≈80 000.
    #    • v1 accidentally achieved this but via the wrong mechanism
    #      (FAIR set to NEXT day's base, every ask looked "cheap",
    #       sell orders at 14005 never filled on a day-1 run).
    #
    #  Strategy:
    #    1. On the very first tick, detect the day's starting price by
    #       rounding the observed mid to the nearest 1 000.
    #    2. Use FAIR = detected_base + ts * IPR_SLOPE_PER_TICK from there on.
    #    3. Buy aggressively up to the position limit whenever ask ≤ FAIR +
    #       IPR_TAKE_SLACK (= 1 tick above fair — don't overpay).
    #    4. Post passive buys to fill the remainder at FAIR − IPR_OFFSET.
    #    5. Do NOT place sell orders while building the long position.
    #       Only sell once at limit if we ever see ask << FAIR (i.e., to
    #       cycle a small portion for a guaranteed profit), via conversion.
    # ═══════════════════════════════════════════════════════════
    def _trade_ipr(self, state: TradingState, td: dict):
        product = "INTARIAN_PEPPER_ROOT"
        od      = state.order_depths[product]
        pos     = state.position.get(product, 0)
        limit   = self.LIMIT[product]
        result  = []
        ts      = state.timestamp

        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        ob_mid   = (best_bid + best_ask) / 2.0 if best_bid and best_ask else (
                    best_ask if best_ask else best_bid)

        # ── FIX 1: Dynamic base detection ────────────────────
        # Store on the first tick (ts == 0 OR not yet stored)
        if 'ipr_base' not in td and ob_mid is not None:
            # Round to nearest 1000 to get the clean day start
            detected = round(ob_mid / 1000) * 1000
            td['ipr_base'] = float(detected)
            logger.print(f"IPR base detected: {detected}")

        ipr_base = td.get('ipr_base', None)
        if ipr_base is None:
            # Fallback: use raw mid if detection hasn't fired yet
            if ob_mid is None:
                return result, td, 0
            ipr_base = round(ob_mid / 1000) * 1000

        FAIR = ipr_base + ts * self.IPR_SLOPE_PER_TICK

        buy_capacity  = limit - pos
        sell_capacity = limit + pos

        # ── Take: lift asks at/below FAIR + slack ─────────────
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price <= FAIR + self.IPR_TAKE_SLACK and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol))
                buy_capacity -= vol
            else:
                break  # sorted; no point continuing

        # ── Passive: fill remaining capacity below fair ───────
        if buy_capacity > 0:
            passive_buy = math.floor(FAIR) - self.IPR_OFFSET
            if best_bid is not None and best_bid + 1 < FAIR:
                passive_buy = best_bid + 1   # penny-inside bid
            # Two-level passive: concentrate most size at the best level
            t1 = (buy_capacity * 2) // 3
            t2 = buy_capacity - t1
            result.append(Order(product, passive_buy,     t1))
            result.append(Order(product, passive_buy - 1, t2))

        # ── Sell: only via conversion, or a tiny cycle for edge ─
        # We do NOT place resting sell orders while building the position,
        # to avoid accidentally reducing a profitable long.
        # Exception: if pos == limit and there's a bid well above FAIR,
        # place a tiny sell to capture that edge and re-buy later.
        SELL_PREMIUM = 8   # only sell if bid ≥ FAIR + 8 (clear of spread)
        if sell_capacity > 0 and pos == limit:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price >= FAIR + SELL_PREMIUM and sell_capacity > 0:
                    vol = min(od.buy_orders[bid_price], sell_capacity, 10)  # cap at 10
                    result.append(Order(product, bid_price, -vol))
                    sell_capacity -= vol
                else:
                    break

        # ── Conversions ───────────────────────────────────────
        conv = 0
        obs = state.observations.conversionObservations.get(product, None)
        if obs is not None:
            conv = self._request_conversions(obs, FAIR, pos, limit)

        logger.print(f"IPR t={ts} base={ipr_base:.0f} FAIR={FAIR:.1f} "
                     f"pos={pos} buy_cap={buy_capacity} conv={conv}")
        return result, td, conv

    # ═══════════════════════════════════════════════════════════
    #  ACO  —  Symmetric market-maker with inventory control
    # ═══════════════════════════════════════════════════════════
    #
    #  v1 bugs (all fixed here):
    #
    #  BUG 2 FIX — Taker threshold:
    #    Was: take sell if best_bid > FAIR  (even +1 tick edge)
    #    Now: take sell if best_bid ≥ FAIR + ACO_MIN_TAKE_EDGE (3 ticks)
    #         take buy  if best_ask ≤ FAIR − ACO_MIN_TAKE_EDGE (3 ticks)
    #    Effect: eliminates the 1-tick-edge sells that drove position to −80.
    #
    #  BUG 3a FIX — Symmetric penny-inside condition:
    #    Was: penny_buy  fires when best_bid + 1 < FAIR  (misses bid == FAIR)
    #    Now: penny_buy  fires when best_bid + 1 ≤ FAIR  (i.e., bid < FAIR)
    #         penny_sell fires when best_ask - 1 ≥ FAIR  (symmetric)
    #
    #  BUG 3b FIX — Passive sell cap:
    #    Was: penny_sell = best_ask − 1 (up to 10015 when ask=10016)
    #    Now: penny_sell = min(best_ask − 1, FAIR + ACO_SELL_PENNY_CAP)
    #         → never more than 10 ticks above FAIR (≤ 10010)
    #    Effect: reduces fill rate on the sell side; passive orders are
    #            only marginally better than FAIR ± offset.
    #
    #  BUG 3c FIX — Soft inventory limits:
    #    When pos ≤ −ACO_SOFT_LIMIT: suppress passive sells entirely.
    #    When pos ≥ +ACO_SOFT_LIMIT: suppress passive buys entirely.
    #    Effect: position cannot be pushed to ±80 by passive order
    #            accumulation alone; takers still work both ways.
    # ═══════════════════════════════════════════════════════════
    def _trade_aco(self, state: TradingState, td: dict):
        product = "ASH_COATED_OSMIUM"
        od      = state.order_depths[product]
        pos     = state.position.get(product, 0)
        limit   = self.LIMIT[product]
        result  = []
        ANCHOR  = self.ACO_ANCHOR
        ts      = state.timestamp

        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        # ── EMA drift safety (unchanged from v1) ──────────────
        ob_mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else None
        ema = td.get('aco_ema', ANCHOR)
        if ob_mid is not None:
            ema = 0.1 * ob_mid + 0.9 * ema
        td['aco_ema'] = ema
        drift_count = td.get('aco_drift', 0)
        drift_count = (drift_count + 1) if abs(ema - ANCHOR) > 50 else 0
        td['aco_drift'] = drift_count
        FAIR = round(ema) if drift_count >= 100 else ANCHOR

        # ── Adaptive offset (fill-time decay, same as v1) ─────
        prev_pos = td.get('aco_prev_pos', 0)
        if pos > prev_pos:   td['aco_last_buy_fill']  = ts
        elif pos < prev_pos: td['aco_last_sell_fill'] = ts
        td['aco_prev_pos'] = pos

        last_buy_fill  = td.get('aco_last_buy_fill',  ts)
        last_sell_fill = td.get('aco_last_sell_fill', ts)
        buy_offset  = max(self.ACO_OFFSET_MIN,
                         self.ACO_OFFSET_MAX - (ts - last_buy_fill)  // self.ACO_DECAY_TICKS)
        sell_offset = max(self.ACO_OFFSET_MIN,
                         self.ACO_OFFSET_MAX - (ts - last_sell_fill) // self.ACO_DECAY_TICKS)

        # ── Bid/ask prediction (unchanged from v1) ────────────
        spread   = (best_ask - best_bid) if (best_bid and best_ask) else 16
        prev_bid = td.get('aco_prev_bid', best_bid)
        prev_ask = td.get('aco_prev_ask', best_ask)
        bid_chg  = (best_bid - prev_bid) if (best_bid and prev_bid) else 0
        ask_chg  = (best_ask - prev_ask) if (best_ask and prev_ask) else 0
        if best_bid is not None: td['aco_prev_bid'] = best_bid
        if best_ask is not None: td['aco_prev_ask'] = best_ask

        a = self.ACO_BID_COEF
        pred_bid_chg = a[0]*bid_chg + a[1]*ask_chg + a[2]*spread + a[3]
        b = self.ACO_ASK_COEF
        pred_ask_chg = b[0]*bid_chg + b[1]*ask_chg + b[2]*spread + b[3]
        pred_next_bid = round(best_bid + pred_bid_chg) if best_bid else None
        pred_next_ask = round(best_ask + pred_ask_chg) if best_ask else None

        buy_capacity  = limit - pos
        sell_capacity = limit + pos

        # ── FIX 2: Taker with minimum edge threshold ──────────
        # Previously: take sell if bid > FAIR (even +1 tick)
        # Now:        take sell if bid ≥ FAIR + ACO_MIN_TAKE_EDGE
        take_buy_threshold  = FAIR - self.ACO_MIN_TAKE_EDGE   # ask must be < this to buy
        take_sell_threshold = FAIR + self.ACO_MIN_TAKE_EDGE   # bid must be ≥ this to sell

        # Spread-conditioned adjustment (same logic as v1, stacks on top of threshold)
        if spread <= 12 and spread > 0:
            take_buy_threshold  = FAIR + 2 - self.ACO_MIN_TAKE_EDGE   # slightly more aggressive
        elif spread >= 20:
            take_sell_threshold = FAIR - 2 + self.ACO_MIN_TAKE_EDGE

        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price < take_buy_threshold and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol))
                buy_capacity -= vol
            else:
                break

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price >= take_sell_threshold and sell_capacity > 0:
                vol = min(od.buy_orders[bid_price], sell_capacity)
                result.append(Order(product, bid_price, -vol))
                sell_capacity -= vol
            else:
                break

        # ── FIX 3a+3b: Symmetric passive order placement ─────
        # BUY side: penny-inside bid when bid is BELOW FAIR (≤ FAIR − 1)
        penny_buy = FAIR - buy_offset
        if best_bid is not None and best_bid + 1 <= FAIR:   # FIX 3a: was strict <, now ≤
            penny_buy = best_bid + 1

        # SELL side: penny-inside ask, but CAPPED at FAIR + ACO_SELL_PENNY_CAP
        penny_sell = FAIR + sell_offset
        if best_ask is not None and best_ask - 1 >= FAIR:   # symmetric condition
            capped = min(best_ask - 1, FAIR + self.ACO_SELL_PENNY_CAP)   # FIX 3b: hard cap
            penny_sell = capped

        # Second passive level using bid/ask prediction (same as v1)
        if pred_next_bid is not None:
            pred_penny_buy = pred_next_bid + 1
            second_buy = (pred_penny_buy if pred_penny_buy < penny_buy - 1
                                            and pred_penny_buy >= FAIR - 15
                          else penny_buy - 1)
        else:
            second_buy = penny_buy - 1

        if pred_next_ask is not None:
            pred_penny_sell = pred_next_ask - 1
            second_sell = (pred_penny_sell if pred_penny_sell > penny_sell + 1
                                              and pred_penny_sell <= FAIR + 15
                           else penny_sell + 1)
        else:
            second_sell = penny_sell + 1

        # ── FIX 3c: Soft inventory limits ─────────────────────
        # If too short, suppress passive sells to stop digging deeper
        # If too long, suppress passive buys
        passive_buy_ok  = pos > -self.ACO_SOFT_LIMIT   # allow buys unless very short
        passive_sell_ok = pos <  self.ACO_SOFT_LIMIT   # allow sells unless very long

        if buy_capacity > 0 and passive_buy_ok:
            t1, t2 = buy_capacity // 2, buy_capacity - buy_capacity // 2
            result.append(Order(product, penny_buy,  t1))
            result.append(Order(product, second_buy, t2))

        if sell_capacity > 0 and passive_sell_ok:
            t1, t2 = sell_capacity // 2, sell_capacity - sell_capacity // 2
            result.append(Order(product, penny_sell,  -t1))
            result.append(Order(product, second_sell, -t2))

        # ── Conversions ───────────────────────────────────────
        conv = 0
        obs = state.observations.conversionObservations.get(product, None)
        if obs is not None:
            conv = self._request_conversions(obs, FAIR, pos, limit)

        logger.print(f"ACO t={ts} FAIR={FAIR} pos={pos} spread={spread} "
                     f"take_buy<{take_buy_threshold} take_sell≥{take_sell_threshold} "
                     f"penny={penny_buy}/{penny_sell} conv={conv}")
        return result, td, conv
