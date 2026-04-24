"""
IMC Prosperity Round 2 — Trader v9
====================================
Version history and what changed in each version:

v3 (= Shane's best, ~8,974 total: IPR=7,291 ACO=1,683)
  - Correct IPR base detection + take_slack=10 for fast fill
  - ACO soft-limit sign fixed (critical v2 bug)
  - ACO passive sell uncapped (restored edge)
  - ACO reversion taker at FAIR+8 (almost never fires — avg ask ≈ 10012)

v4 (8,829 total: IPR=7,270 ACO=1,559) — WORSE
  - Added rule-discovery signals (momentum, book imbalance)
  - Suppressed ACO sells on momentum signal + hold window
  - Verdict: over-engineering hurt. Reverted to v3 for v6.

v5 (8,546 total: IPR=7,343 ACO=1,203) — MUCH WORSE
  - Fix C: graduated reversion taker (FAIR+12/+15) — MAIN KILLER
    Avg ask ≈ 10012, so threshold FAIR+12 fires constantly.
  - Verdict: all "fixes" were actually bugs. Full revert to v3 for v6.

v6 (~8,620–8,724 total: IPR=7,243–7,270 ACO=1,350–1,481)
  - IPR: identical to v3; ACO: identical to v3
  - Conversions: CONV_MIN_EDGE lowered 2.0 → 0.5, CONV_MAX_UNITS raised 10 → 40
  - bid() added: returns 200 for Market Access Fee auction

v7 (7,996 total: IPR=7,270 ACO=726) — MUCH WORSE — REVERTED
  - ACO: changed passive sell to fixed FAIR+2 / FAIR+3
  - Failure: burst filled 54 units passively in 4 ticks → pos=-56;
    reversion bought back at FAIR+6/+7. Avg sell: 10003 (was 10007), net -625 ACO.
  - Key lesson: reversion mode (forced buy at FAIR+8) is the real killer, not
    the threshold itself. With ACO_SOFT_LIMIT=40 we hit reversion too easily.

v8 (stable baseline) — REVERTED TO v6 EXACTLY
  - ACO: best_ask-1 passive sell restored; bid()=200; CONV_MIN_EDGE=0.5

v9 (this file) — AGGRESSIVE HIGH-VOLUME ACO
  Hypothesis: top traders get 4,000+ ACO by trading 4× more volume at FAIR+1/+2
  events (bot bids there ~6% of ticks) that v6's FAIR+3 threshold completely misses.

  Three changes from v8:
  1. ACO_MIN_TAKE_EDGE = 1 (was 3)
       → taker sell when bid ≥ FAIR+1 (vs FAIR+3)
       → taker buy  when ask ≤ FAIR-1 (vs FAIR-3, also more buy coverage)
  2. ACO_SOFT_LIMIT = 75 (was 40)
       → allow position to drift ±75 before gating passive orders
       → prevents premature reversion trigger from burst accumulation
  3. Reversion mode REMOVED (was: forced buy at FAIR+8 when pos ≤ -40)
       → was the main PnL killer in v7 (bought at FAIR+6-8 during each burst)
       → replaced by day-end close: if ts ≥ 90000 and |pos| > 20, force close
         aggressively (cross spread up to FAIR±20) to avoid end-of-day
         mark-to-market loss on residual position

  Expected dynamics:
  - Sell at FAIR+1 to FAIR+9 on every high-bid tick → large short accumulates
  - Cover organically at FAIR-1 to FAIR-8 when bot's ask dips (frequent)
  - No expensive forced buybacks → avg buy stays near FAIR-4 to FAIR-6
  - Net edge per unit: sell avg ≈ FAIR+2, buy avg ≈ FAIR-4 → ~6 ticks
  - Volume: ~4× v8 → estimated ACO PnL 3,000–4,000 (vs 1,400 in v8)
  - Day-end close cost: if pos=-40 at t=90000, pays ~8 ticks to close =
    40×8 = 320 XIRECS one-time cost, acceptable

  Spread adjustment removed: at ACO_MIN_TAKE_EDGE=1, the old spread-based
  threshold shifts (±2) would push thresholds across FAIR (buy above FAIR,
  sell below FAIR) which is clearly wrong. Removed entirely for v9.
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

    # ── Market Access Fee bid ─────────────────────────────────
    # Top 50% of bids (by value) win extra 25% order-book volume.
    # Winners pay the fee; losers pay nothing and trade without extra volume.
    # Testing always uses 80% of base quotes (randomized) regardless of bid.
    # In the live run, winners get ~25% more ACO/IPR quotes → ~425 extra PnL.
    # Bid < 425 is net positive if we win. Bidding 200 is likely top 50%
    # (many teams will bid 0 or nothing) while keeping the fee well below benefit.
    def bid(self):
        return 200

    LIMIT = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}

    # ── IPR ───────────────────────────────────────────────────
    IPR_SLOPE_PER_TICK = 0.001   # +1 per 1000 ticks (exact, validated days −1,0,1)
    IPR_TAKE_SLACK     = 10      # take asks up to FAIR+10 for fast fill at day start
    IPR_PASSIVE_OFFSET = 4       # passive buy below fair while ramping up

    # ── ACO ───────────────────────────────────────────────────
    ACO_ANCHOR       = 10_000
    ACO_OFFSET_MAX   = 7
    ACO_OFFSET_MIN   = 3
    ACO_DECAY_TICKS  = 5_000

    # v9: lowered from 3 → 1. Captures FAIR+1 and FAIR+2 bid events (~6% of ticks)
    # that v6's threshold of 3 completely missed. Also takes buys at FAIR-1 (was FAIR-3)
    # giving more organic coverage to close the short position.
    ACO_MIN_TAKE_EDGE = 1

    # v9: raised from 40 → 75. Allows large position drift without triggering
    # the reversion block (which buys at FAIR+8 and destroys edge).
    # Hard position limit is 80; we stop posting passives at ±75.
    ACO_SOFT_LIMIT   = 75

    # v9: NOT USED — reversion mode removed. Setting kept for reference only.
    # Replaced by day-end close (ts ≥ 90000) in _trade_aco.
    ACO_REVERT_SLACK  = 999

    # Bid/ask prediction coefficients (validated days −1, 0, 1)
    ACO_BID_COEF = (-0.271, -0.217,  0.442, -7.156)
    ACO_ASK_COEF = (-0.276, -0.217, -0.570,  9.223)

    # ── Conversion ────────────────────────────────────────────
    # v6 CHANGE: lowered from 2.0 → 0.5 and raised units from 10 → 40.
    # Tested logs all have empty conversionObservations so these parameters
    # had zero effect in testing. In the live run, observations may be populated
    # (the access fee grants +25% volume on the XIRECS exchange). We want to
    # capture any positive edge that appears. Min edge of 0.5 means we still
    # require a meaningful arb (not just rounding noise) before converting.
    CONV_MIN_EDGE  = 0.5
    CONV_MAX_UNITS = 40

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
        """Return (buy_edge, sell_edge) for a conversion observation.

        buy_edge  = fair - effective_buy_cost
                  = fair - (askPrice + transportFees + importTariff)
        sell_edge = effective_sell_revenue - fair
                  = (bidPrice - transportFees - exportTariff) - fair

        Positive buy_edge  → converting INTO product is profitable.
        Positive sell_edge → converting OUT OF product is profitable.
        """
        effective_buy  = obs.askPrice  + obs.transportFees + obs.importTariff
        effective_sell = obs.bidPrice  - obs.transportFees - obs.exportTariff
        return fair - effective_buy, effective_sell - fair

    def _request_conversions(self, obs, fair: float, pos: int, limit: int,
                             product: str) -> int:
        """Return conversion units to request this tick.

        Positive = buy conversions (receive product, pay cash).
        Negative = sell conversions (deliver product, receive cash).
        """
        if obs is None:
            return 0

        buy_edge, sell_edge = self._conv_edge(obs, fair)

        # Log observations every tick so we can see if they ever populate
        logger.print(
            f"CONV {product} obs: bid={obs.bidPrice} ask={obs.askPrice} "
            f"tf={obs.transportFees} et={obs.exportTariff} it={obs.importTariff} "
            f"buy_edge={buy_edge:.2f} sell_edge={sell_edge:.2f}"
        )

        if buy_edge > self.CONV_MIN_EDGE:
            units = min(limit - pos, self.CONV_MAX_UNITS)
            if units > 0:
                logger.print(f"CONV {product} BUY {units} @ edge={buy_edge:.2f}")
            return units

        if sell_edge > self.CONV_MIN_EDGE:
            units = min(limit + pos, self.CONV_MAX_UNITS)
            if units > 0:
                logger.print(f"CONV {product} SELL {units} @ edge={sell_edge:.2f}")
            return -units

        return 0

    # ═══════════════════════════════════════════════════════════
    #  IPR — Trend Rider (hold max long throughout the day)
    # ═══════════════════════════════════════════════════════════
    #  Fair value formula: FAIR = base + ts × 0.001
    #    base = detected from first tick mid-price (rounded to nearest 1000)
    #
    #  Entry:  take asks up to FAIR + IPR_TAKE_SLACK (= 10).
    #    Fills 80 units within the first ~200 ticks. Entry premium ≤ 10 ticks;
    #    trend earns ≥ 1000 ticks over a full day.
    #
    #  Hold:   never place resting sell orders while building the position.
    #
    #  Micro sell: only if bid ≥ FAIR+15 (clear outlier), captures a tiny
    #    extra premium without reducing the core long position.
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
            if best_bid is not None and best_bid + 1 <= FAIR:
                passive_buy = best_bid + 1
            # Two-level: 2/3 at penny, 1/3 one tick below
            t1 = (buy_capacity * 2) // 3
            t2 = buy_capacity - t1
            result.append(Order(product, passive_buy,     t1))
            result.append(Order(product, passive_buy - 1, t2))

        # ── Micro sell: only on extreme bid outliers ──────────
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
            conv = self._request_conversions(obs, FAIR, pos, limit, product)

        logger.print(f"IPR t={ts} base={ipr_base:.0f} FAIR={FAIR:.1f} pos={pos}")
        return result, td, conv

    # ═══════════════════════════════════════════════════════════
    #  ACO — High-volume aggressive taker, no reversion, day-end close
    # ═══════════════════════════════════════════════════════════
    #  v9 changes vs v8:
    #  - ACO_MIN_TAKE_EDGE = 1: sell at FAIR+1, buy at FAIR-1
    #  - ACO_SOFT_LIMIT = 75: allow large position drift
    #  - Reversion mode REMOVED (was the PnL killer in v7 at FAIR+8)
    #  - Day-end close: last 10k ticks, force-close position vs FAIR±20
    #  - Spread-based threshold shift REMOVED (crosses FAIR at edge=1)
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

        # ── EMA drift ─────────────────────────────────────────
        ob_mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else None
        ema = td.get('aco_ema', ANCHOR)
        if ob_mid is not None:
            ema = 0.1 * ob_mid + 0.9 * ema
        td['aco_ema'] = ema
        drift_count = td.get('aco_drift', 0)
        drift_count = (drift_count + 1) if abs(ema - ANCHOR) > 50 else 0
        td['aco_drift'] = drift_count
        FAIR = round(ema) if drift_count >= 100 else ANCHOR

        # ── Adaptive fill-time offset ─────────────────────────
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

        # ── Bid/ask prediction ────────────────────────────────
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

        # ── Taker thresholds (v9: edge=1, no spread adjustment) ──
        # We take sells when bid ≥ FAIR+1 and buys when ask ≤ FAIR-1.
        # No spread-based shift: at edge=1, shifting ±2 would cross FAIR.
        take_buy_threshold  = FAIR - self.ACO_MIN_TAKE_EDGE   # FAIR-1 = 9999
        take_sell_threshold = FAIR + self.ACO_MIN_TAKE_EDGE   # FAIR+1 = 10001

        # Day-end close (last 10k ticks): force-close residual position.
        # Accepts up to FAIR±20 to avoid end-of-day mark-to-market losses.
        # A residual of -40 at FAIR+20 costs 40×20=800 vs risk of holding.
        if ts >= 90_000:
            if pos < -20:
                take_buy_threshold  = FAIR + 20   # buy at any reasonable ask to close
            elif pos > 20:
                take_sell_threshold = FAIR - 20   # sell at any reasonable bid to close

        # NO reversion mode: removed in v9.
        # v7/v8 had: if pos <= -40: take_buy_threshold = FAIR+8
        # This forced buys at FAIR+6-8 during every burst → killed edge.
        # Instead: let position drift to ±75 and recover organically.

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
        # Buy: penny-inside bid when bid ≤ FAIR
        penny_buy = FAIR - buy_offset
        if best_bid is not None and best_bid + 1 <= FAIR:
            penny_buy = best_bid + 1

        # Sell: penny-inside ask (uncapped — the 10010 cap from v2 hurt edge
        # and was unnecessary once the soft-limit sign was fixed)
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

        # ── Corrected soft inventory gate ─────────────────────
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
            conv = self._request_conversions(obs, FAIR, pos, limit, product)

        logger.print(f"ACO t={ts} FAIR={FAIR} pos={pos} spread={spread} "
                     f"sell_ok={passive_sell_ok} buy_ok={passive_buy_ok} "
                     f"take<{take_buy_threshold}/>=={take_sell_threshold}")
        return result, td, conv
