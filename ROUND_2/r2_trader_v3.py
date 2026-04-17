"""
IMC Prosperity Round 2 — Trader v3
====================================
Root cause analysis of v1 and v2 logs:

┌──────────────────────────────────────────────────────────────────────┐
│  v1 BUGS (trader1.log)                                               │
│  A. IPR base hardcoded to 14000; day-1 starts at 13000.             │
│     Every ask looked "cheap" → bought 80 units immediately,         │
│     sell orders at 14005 never filled. Lucky trend-ride profit.     │
│  B. ACO taker fires on ANY bid > FAIR, even +1 tick. Drove          │
│     position to −80 within 1800 ticks.                              │
│  C. ACO passive sell at best_ask−1 ≈ 10015 fills fast; passive      │
│     buy at FAIR−7 = 9993 never fills. Structural short bias.        │
├──────────────────────────────────────────────────────────────────────┤
│  v2 BUGS (278218.log) — introduced new issues                       │
│  D. Soft-limit sign flip: conditions were INVERTED.                 │
│     At pos=−40: v2 SUPPRESSED buys (pos>−40 = FALSE) and           │
│     ALLOWED more sells (pos<40 = TRUE). Opposite of intent.        │
│     This amplified the short instead of capping it.                 │
│  E. IPR take slack = 1. At t=0, FAIR=13000, asks at 13007−13010,   │
│     so no taker buys fire. Position built slowly via passive only.  │
│     Full 80-unit position not reached until t≈50000 — missed       │
│     50k ticks of trend earning.                                     │
│  F. Passive sell capped at 10010. This sold at worse prices (10010  │
│     vs 10015 in v1) while the position still hit −80 because of    │
│     the inverted soft limit. Net: lower edge AND same bad position. │
└──────────────────────────────────────────────────────────────────────┘

v3 fixes:
  1. IPR take slack = 10: takes asks up to FAIR+10, filling 80 units
     within a few ticks — same speed as v1 but with correct fair value.
  2. ACO soft limit corrected (sign flip fixed):
       passive_buy_ok  = pos <  SOFT_LIMIT   (True when short → allow buys)
       passive_sell_ok = pos > -SOFT_LIMIT   (False when short → suppress sells)
  3. ACO passive sell UN-capped (restored to best_ask−1 = ~10015).
     The cap was hurting edge without fixing the position — the inverted
     soft limit was the real problem. Once fixed, the cap is unnecessary.
  4. ACO reversion taker: when pos ≤ −SOFT_LIMIT, buy aggressively at
     asks up to FAIR+REVERT_SLACK (8), accelerating position recovery.
  5. Symmetric penny-inside condition retained (≤ not <) from v2.
"""

import json
import math
from typing import Any
from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)


# ═══════════════════════════════════════════════════════════════
#  Logger
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

    # ── IPR ───────────────────────────────────────────────────
    IPR_SLOPE_PER_TICK = 0.001   # +1 per 1000 ticks (exact, validated days −1,0,1)
    # FIX E: aggressive take so position fills within first few ticks.
    # v1 accidentally did this (wrong FAIR); v2 had slack=1 and filled too slowly.
    # With slack=10, asks at FAIR+7 to FAIR+10 are taken immediately at t=0.
    # The 7−10 tick premium is a tiny fraction of the ~1000-tick daily trend gain.
    IPR_TAKE_SLACK     = 10
    IPR_PASSIVE_OFFSET = 4      # passive buy below fair while ramping up

    # ── ACO ───────────────────────────────────────────────────
    ACO_ANCHOR       = 10_000
    ACO_OFFSET_MAX   = 7
    ACO_OFFSET_MIN   = 3
    ACO_DECAY_TICKS  = 5_000

    # Min edge to trigger taker (keep at 3 from v2 — eliminates 1−2 tick noise sells)
    ACO_MIN_TAKE_EDGE = 3

    # FIX D: corrected soft limit thresholds (SIGN WAS FLIPPED IN V2)
    #   passive_buy_ok  = pos <  SOFT_LIMIT   → True when short, suppresses when very long
    #   passive_sell_ok = pos > -SOFT_LIMIT   → True when long,  suppresses when very short
    ACO_SOFT_LIMIT   = 40

    # FIX (new): reversion taker — when pos exceeds soft limit, take more aggressively
    # to bring position back. Sell at asks up to FAIR+REVERT_SLACK when very short.
    ACO_REVERT_SLACK  = 8

    # Bid/ask prediction coefficients (validated days −1, 0, 1 — unchanged)
    ACO_BID_COEF = (-0.271, -0.217,  0.442, -7.156)
    ACO_ASK_COEF = (-0.276, -0.217, -0.570,  9.223)

    # ── Conversion ────────────────────────────────────────────
    CONV_MIN_EDGE  = 2.0
    CONV_MAX_UNITS = 10

    # ─────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        orders:      dict = {}
        conversions: int  = 0

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

    # ─────────────────────────────────────────────────────────
    def _conv_edge(self, obs, fair: float):
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
    #  IPR — Trend Rider (hold max long throughout the day)
    # ═══════════════════════════════════════════════════════════
    #  Fair value formula: FAIR = base + ts × 0.001
    #    base = detected from first tick mid-price (rounded to nearest 1000)
    #
    #  Entry:  take asks up to FAIR + IPR_TAKE_SLACK (= 10).
    #    - This fills 80 units within the first ~200 ticks, same as v1 but
    #      with correct fair-price knowledge (not a lucky overshoot).
    #    - Entry premium ≤ 10 ticks; trend earns ≥ 1000 ticks over a full day.
    #    - Remaining capacity filled passively via penny-inside-bid.
    #
    #  Hold:   never place resting sell orders while building the position.
    #    - Avoids the v1 bug of sell orders at wrong-day price (14005),
    #      and avoids the v2 bug of tiny take slack causing slow fill.
    #
    #  Exit at limit: a small taker sell fires only if bid ≥ FAIR+15 (clear
    #    outlier), capturing a high-edge cycle without reducing core position.
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
        ob_mid   = ((best_bid + best_ask) / 2.0 if best_bid and best_ask
                    else (best_ask if best_ask else best_bid))

        # ── Detect day's base price once on first available mid ──
        if 'ipr_base' not in td and ob_mid is not None:
            detected = round(ob_mid / 1000) * 1000
            td['ipr_base'] = float(detected)
            logger.print(f"IPR base={detected}")

        ipr_base = td.get('ipr_base', None)
        if ipr_base is None:
            if ob_mid is None:
                return result, td, 0
            ipr_base = round(ob_mid / 1000) * 1000

        FAIR = ipr_base + ts * self.IPR_SLOPE_PER_TICK
        buy_capacity  = limit - pos
        sell_capacity = limit + pos

        # ── Taker buys: fill aggressively at start ────────────
        # FIX E: slack=10 so market asks (usually FAIR+7 to FAIR+10) are taken
        # immediately, replicating v1's fast fill without wrong fair value.
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price <= FAIR + self.IPR_TAKE_SLACK and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol))
                buy_capacity -= vol
            else:
                break

        # ── Passive buys: fill remaining at penny-inside-bid ──
        if buy_capacity > 0:
            passive_buy = math.floor(FAIR) - self.IPR_PASSIVE_OFFSET
            if best_bid is not None and best_bid + 1 <= FAIR:  # penny-inside (v2 fix)
                passive_buy = best_bid + 1
            # Two-level: 2/3 at penny, 1/3 one tick below
            t1 = (buy_capacity * 2) // 3
            t2 = buy_capacity - t1
            result.append(Order(product, passive_buy,     t1))
            result.append(Order(product, passive_buy - 1, t2))

        # ── Micro sell: only on extreme bid outliers ──────────
        # When already at limit, sell a small amount at FAIR+15 or better.
        # This locks in a tiny extra premium without reducing core position.
        SELL_PREMIUM = 15
        if pos == limit and sell_capacity > 0:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price >= FAIR + SELL_PREMIUM and sell_capacity > 0:
                    vol = min(od.buy_orders[bid_price], sell_capacity, 5)
                    result.append(Order(product, bid_price, -vol))
                    sell_capacity -= vol
                else:
                    break

        # ── Conversions ───────────────────────────────────────
        conv = 0
        obs = state.observations.conversionObservations.get(product, None)
        if obs is not None:
            conv = self._request_conversions(obs, FAIR, pos, limit)

        logger.print(f"IPR t={ts} base={ipr_base:.0f} FAIR={FAIR:.1f} pos={pos}")
        return result, td, conv

    # ═══════════════════════════════════════════════════════════
    #  ACO — Symmetric market-maker with corrected inventory gate
    # ═══════════════════════════════════════════════════════════
    #  FIX D  (critical): correct the inverted soft-limit conditions from v2.
    #    v2 had:  passive_buy_ok  = pos > -SOFT  → FALSE at pos=−40 (suppressed buys!)
    #             passive_sell_ok = pos < +SOFT   → TRUE  at pos=−40 (allowed sells!)
    #    v3 has:  passive_buy_ok  = pos <  SOFT   → TRUE  at pos=−40 (allow buys ✓)
    #             passive_sell_ok = pos > -SOFT   → FALSE at pos=−40 (suppress sells ✓)
    #
    #  FIX F (reversed): passive sell is NOT capped at FAIR+10.
    #    The cap hurt edge (sold at 10010 vs 10015) without fixing position,
    #    because the inverted soft limit was the actual cause of going to −80.
    #    With the sign fix, the soft limit now works as intended, so the cap
    #    is no longer needed and restoring 10015 improves P&L.
    #
    #  Reversion taker (new): when pos ≤ −SOFT_LIMIT, the take-buy threshold
    #    is widened to FAIR + REVERT_SLACK (10008). This aggressively buys back
    #    when the market dips, accelerating position recovery.
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

        # ── EMA drift (unchanged) ─────────────────────────────
        ob_mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else None
        ema = td.get('aco_ema', ANCHOR)
        if ob_mid is not None:
            ema = 0.1 * ob_mid + 0.9 * ema
        td['aco_ema'] = ema
        drift_count = td.get('aco_drift', 0)
        drift_count = (drift_count + 1) if abs(ema - ANCHOR) > 50 else 0
        td['aco_drift'] = drift_count
        FAIR = round(ema) if drift_count >= 100 else ANCHOR

        # ── Adaptive fill-time offset (unchanged) ────────────
        prev_pos = td.get('aco_prev_pos', 0)
        if pos > prev_pos:   td['aco_last_buy_fill']  = ts
        elif pos < prev_pos: td['aco_last_sell_fill'] = ts
        td['aco_prev_pos'] = pos

        lbf = td.get('aco_last_buy_fill',  ts)
        lsf = td.get('aco_last_sell_fill', ts)
        buy_offset  = max(self.ACO_OFFSET_MIN,
                         self.ACO_OFFSET_MAX - (ts - lbf) // self.ACO_DECAY_TICKS)
        sell_offset = max(self.ACO_OFFSET_MIN,
                         self.ACO_OFFSET_MAX - (ts - lsf) // self.ACO_DECAY_TICKS)

        # ── Bid/ask prediction (unchanged) ───────────────────
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

        # ── Taker thresholds ──────────────────────────────────
        take_buy_threshold  = FAIR - self.ACO_MIN_TAKE_EDGE   # 9997 normally
        take_sell_threshold = FAIR + self.ACO_MIN_TAKE_EDGE   # 10003 normally

        # Spread-conditioned tweak (kept from v1/v2)
        if spread <= 12 and spread > 0:
            take_buy_threshold = FAIR + 2 - self.ACO_MIN_TAKE_EDGE
        elif spread >= 20:
            take_sell_threshold = FAIR - 2 + self.ACO_MIN_TAKE_EDGE

        # Reversion taker (new in v3): when very short, widen buy threshold
        # to FAIR + REVERT_SLACK so we take asks as soon as market dips slightly.
        # This accelerates recovery from deep short positions.
        if pos <= -self.ACO_SOFT_LIMIT:
            take_buy_threshold = FAIR + self.ACO_REVERT_SLACK   # 10008

        # Symmetric reversion when very long
        if pos >= self.ACO_SOFT_LIMIT:
            take_sell_threshold = FAIR - self.ACO_REVERT_SLACK   # 9992

        # ── Taker orders ──────────────────────────────────────
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

        # ── Passive order levels ──────────────────────────────
        # Buy: penny-inside bid when bid ≤ FAIR (symmetric fix from v2)
        penny_buy = FAIR - buy_offset
        if best_bid is not None and best_bid + 1 <= FAIR:   # bid is at/below fair
            penny_buy = best_bid + 1

        # Sell: penny-inside ask (UNCAPPED — restored from v1)
        # FIX F: removed the 10010 cap; the inverted soft-limit was the actual
        # cause of going to −80. With that fixed, selling at 10015 is pure edge.
        penny_sell = FAIR + sell_offset
        if best_ask is not None and best_ask - 1 >= FAIR:
            penny_sell = best_ask - 1

        # Second levels via bid/ask prediction
        if pred_next_bid is not None:
            pred_penny_buy = pred_next_bid + 1
            second_buy = (pred_penny_buy
                          if pred_penny_buy < penny_buy - 1 and pred_penny_buy >= FAIR - 15
                          else penny_buy - 1)
        else:
            second_buy = penny_buy - 1

        if pred_next_ask is not None:
            pred_penny_sell = pred_next_ask - 1
            second_sell = (pred_penny_sell
                           if pred_penny_sell > penny_sell + 1 and pred_penny_sell <= FAIR + 15
                           else penny_sell + 1)
        else:
            second_sell = penny_sell + 1

        # ── FIX D: Corrected soft inventory gate ──────────────
        # CORRECT:  allow buys  when pos is below  +SOFT_LIMIT (not very long)
        #           allow sells when pos is above  −SOFT_LIMIT (not very short)
        # v2 BUG:   conditions were swapped → amplified imbalance instead of capping it
        passive_buy_ok  = pos <  self.ACO_SOFT_LIMIT   # True when short or neutral ✓
        passive_sell_ok = pos > -self.ACO_SOFT_LIMIT   # True when long  or neutral ✓

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
                     f"sell_ok={passive_sell_ok} buy_ok={passive_buy_ok} "
                     f"take<{take_buy_threshold}/≥{take_sell_threshold}")
        return result, td, conv
