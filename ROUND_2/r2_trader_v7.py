"""
IMC Prosperity Round 2 — Trader v7
====================================
Version history and what changed in each version:

v3 (= Shane's best, 8,974 total: IPR=7,291 ACO=1,683)
  - Correct IPR base detection + take_slack=10 for fast fill
  - ACO soft-limit sign fixed (critical v2 bug)
  - ACO passive sell uncapped (restored edge)
  - ACO reversion taker at FAIR+8 (almost never fires — avg ask ≈ 10012)

v4 (8,829 total: IPR=7,270 ACO=1,559) — WORSE
  - Added rule-discovery signals (momentum, book imbalance)
  - Suppressed ACO sells on momentum signal + hold window
  - Verdict: over-engineering hurt. Reverted to v3 for v6.

v5 (8,546 total: IPR=7,343 ACO=1,203) — MUCH WORSE
  - Fix C graduated reversion (FAIR+12/+15) main killer — avg ask ≈ 10012,
    so threshold fires constantly. Keep REVERT_SLACK=8.

v6 (~8,700 test, ~9,500 est. live with MAF)
  - v3 base + aggressive conversion (CONV_MIN_EDGE=0.5, CONV_MAX_UNITS=40)
  - Added bid()=200 for Market Access Fee auction

v7 (this file) — passive sell repositioned to capture missed bid events
  Root cause analysis of leaderboard gap (~2,500 XiRECs behind top traders):
    Order book data from 200k ticks revealed:
    - Bot bids cluster at FAIR-7 to FAIR-1 (~70% of ticks)
    - Bid occasionally rises to FAIR+1 (~5% of ticks, avg vol 12)
    - Bid occasionally rises to FAIR+2 (~1% of ticks, avg vol 12)
    - Bid above FAIR+3 (~2% of ticks) → caught by taker (ACO_MIN_TAKE_EDGE=3)
    - Bid at FAIR+2: taker threshold NOT met (3>2), passive sell at FAIR+9
      is too high → ZERO fills. Completely missed opportunity.
    - Bid at FAIR+1: same — taker misses it, passive sell too high.

  v7 FIX — two-level passive sell strategy:
    Level 1 (penny_sell): FAIR+2, always.
      - Replaces the best_ask-1 override (which put us at FAIR+9, never fills)
      - Fills when bid reaches FAIR+2 (taker threshold=3 doesn't catch this)
      - 2-tick edge, ~1% of ticks → estimated +240-480 XiRECs per run
    Level 2 (second_sell): FAIR+3, always.
      - Sits alongside the taker at FAIR+3 bid events
      - If bid volume at FAIR+3 exceeds what taker takes, second level fills
      - Also fills when bid hits FAIR+3 from below after taker capacity used
      - ~2% of ticks → estimated +200-240 XiRECs per run

  Why NOT lower the taker threshold to FAIR+1?
    - Reversion cost: each extra sell at 1-tick edge pushes pos toward -80,
      triggering reversion buys at FAIR+8 (8-tick cost). Net negative.
    - Passive approach avoids this: passive fills only when we have capacity,
      doesn't systematically deepen the short position.

  Why NOT lower REVERT_SLACK (reversion buy threshold)?
    - Asks minimum at FAIR+8 (5% of ticks). Lower threshold = taker never fires.
    - We'd be stuck short at -80 with no way back.

  Expected total improvement: +440-720 XiRECs ACO per run (test, 80% vol).
  With MAF live run (125% vol): proportionally scaled, ~+700-1,100.
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

    ACO_MIN_TAKE_EDGE = 3        # min edge to trigger taker (eliminates 1-2 tick noise)

    # Corrected soft-limit thresholds (sign was flipped in v2):
    #   passive_buy_ok  = pos <  SOFT_LIMIT   → True when short, suppress when very long
    #   passive_sell_ok = pos > -SOFT_LIMIT   → True when long,  suppress when very short
    ACO_SOFT_LIMIT   = 40

    # Reversion taker threshold: FAIR+8 almost never fires (avg ask ≈ 10012).
    # DO NOT raise this — v5 raised it to FAIR+12/+15 and fired constantly,
    # costing 659 extra ticks. Keep at 8.
    ACO_REVERT_SLACK  = 8

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
    #  ACO — Symmetric market-maker with corrected inventory gate
    # ═══════════════════════════════════════════════════════════
    #  Identical to v3 (= Shane's best). Do NOT modify without careful log analysis.
    #
    #  Key invariants validated by log analysis:
    #  - avg sell ≈ 10007.73, avg buy ≈ 9998.03 → solid edge both sides
    #  - reversion taker (FAIR+8) almost never fires → correct threshold
    #  - sell floor (tried in v5) blocks profitable 10003-10007 fills → wrong idea
    #  - graduated reversion FAIR+12/+15 (tried in v5) fires constantly → wrong idea
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

        # ── Taker thresholds ──────────────────────────────────
        take_buy_threshold  = FAIR - self.ACO_MIN_TAKE_EDGE   # 9997 normally
        take_sell_threshold = FAIR + self.ACO_MIN_TAKE_EDGE   # 10003 normally

        if spread <= 12 and spread > 0:
            take_buy_threshold = FAIR + 2 - self.ACO_MIN_TAKE_EDGE
        elif spread >= 20:
            take_sell_threshold = FAIR - 2 + self.ACO_MIN_TAKE_EDGE

        # Reversion taker: FAIR+8 almost never fires (avg ask ≈ 10012).
        # Keep threshold here — raising it (v5 mistake) burns ticks constantly.
        if pos <= -self.ACO_SOFT_LIMIT:
            take_buy_threshold = FAIR + self.ACO_REVERT_SLACK   # 10008
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
        # Buy: penny-inside bid when bid ≤ FAIR (unchanged from v3/v6)
        penny_buy = FAIR - buy_offset
        if best_bid is not None and best_bid + 1 <= FAIR:
            penny_buy = best_bid + 1

        # Second buy level via prediction (unchanged)
        if pred_next_bid is not None:
            pred_penny_buy = pred_next_bid + 1
            second_buy = (pred_penny_buy
                          if pred_penny_buy < penny_buy - 1 and pred_penny_buy >= FAIR - 15
                          else penny_buy - 1)
        else:
            second_buy = penny_buy - 1

        # Sell: v7 CHANGE — fixed two-level passive sell instead of best_ask-1.
        #
        # v3/v6 used penny_sell = best_ask - 1 ≈ FAIR+9 (when best ask = FAIR+10).
        # Order book analysis showed best_ask rarely comes below FAIR+8, so bids
        # almost never reach FAIR+9. Result: passive sell at FAIR+9 = near-zero fills.
        #
        # Data showed bid reaches FAIR+2 ~1% of ticks (avg vol 12) and FAIR+1 ~5%,
        # but our taker threshold is FAIR+3 so these events generate ZERO fills.
        #
        # Fix: post two passive sell levels:
        #   Level 1 at FAIR+2: fills when bid = FAIR+2 (taker misses at threshold 3)
        #   Level 2 at FAIR+3: sits alongside taker, captures any excess bid vol
        #
        # We do NOT lower the taker threshold: selling at 1-2 tick edge aggressively
        # deepens the chronic short and triggers expensive FAIR+8 reversion buys.
        # Passive fills are gentler — they only fire when we have surplus capacity.
        penny_sell  = int(FAIR) + 2   # FAIR+2: fills when bid reaches FAIR+2
        second_sell = int(FAIR) + 3   # FAIR+3: absorbs excess taker-level bid volume

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
                     f"take<{take_buy_threshold}/≥{take_sell_threshold}")
        return result, td, conv
