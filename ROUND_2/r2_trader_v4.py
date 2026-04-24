"""
IMC Prosperity Round 2 — Trader v4
====================================
Built on v3 (P&L 8,974.81: IPR=7,291, ACO=1,683).

New in v4: rule-discovery signals from summary.csv / top_rules.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IPR SIGNAL (score=0.788, recall=1.0 across all 3 days — very strong)
─────────────────────────────────────────────────────────────────────
  Rule: time_frac ∈ (0.05, 0.15] AND book_imbalance > 0.47 AND total_levels > 3
  Interpretation: early-day, bid-heavy, deep book → XIRECS external market
  is about to offer a favorable buy conversion on IPR.

  v4 changes:
    1. "Hold window": when signal active, suppress ALL passive sells
       (overrides even the FAIR+15 micro-sell). We want 100% position
       through the predicted XIRECS conversion event.
    2. Boost conversions: raise CONV_MAX_UNITS to 20 during the signal.
       If XIRECS is offering a buy below our FAIR, take up to 20 units
       to lock in the edge (position can briefly exceed soft limit during conv).
    3. Tighter spread gate (secondary rule): also suppress sells when
       spread_over_mid <= 0.0011 AND book_imbalance > 0.47, confirming
       the tight-spread "settled" phase before the price move.

ACO SIGNAL (score=0.405, recall=0.33 — weaker, use with caution)
─────────────────────────────────────────────────────────────────────
  Rule: time_frac <= 0.05 AND bid_volume_1 <= 10 AND momentum_5 <= -2
        AND momentum_20 > 3
  Interpretation: very early day, thin bid side, short-term dip within
  medium-term upswing → favorable ACO buy opportunity.

  v4 changes:
    1. Track 20-tick mid-price history to compute momentum_5 and momentum_20.
    2. When signal fires: lower take_buy_threshold by 3 extra ticks
       (buy asks up to FAIR+3 instead of requiring FAIR-3 or below).
    3. Suppress passive sell for that tick (don't add more short while
       a buy dip is predicted).
    4. Signal is gated by early-day window (ts <= 50k) so it can't fire
       during the rest of the day.

All v3 logic (soft limits, reversion taker, dynamic base, pred offsets)
is preserved unchanged.
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
    IPR_SLOPE_PER_TICK = 0.001
    IPR_TAKE_SLACK     = 10     # aggressive entry fill (fills 80 units within ~200 ticks)
    IPR_PASSIVE_OFFSET = 4

    # Signal parameters (from top_rules.csv analysis)
    IPR_SIG_TS_LO       = 50_000    # time_frac > 0.05  → ts > 50k
    IPR_SIG_TS_HI       = 150_000   # time_frac <= 0.15 → ts <= 150k
    IPR_SIG_IMBALANCE   = 0.47      # book_imbalance threshold
    IPR_SIG_MIN_LEVELS  = 3         # total_levels > 3
    IPR_SIG_SPREAD_MOM  = 0.0011    # spread_over_mid gate (secondary rule)

    # Conversion boost during signal window
    CONV_MIN_EDGE      = 2.0
    CONV_MAX_UNITS     = 10          # normal
    CONV_SIGNAL_UNITS  = 20          # boosted during IPR signal window

    # ── ACO ───────────────────────────────────────────────────
    ACO_ANCHOR       = 10_000
    ACO_OFFSET_MAX   = 7
    ACO_OFFSET_MIN   = 3
    ACO_DECAY_TICKS  = 5_000
    ACO_MIN_TAKE_EDGE = 3
    ACO_SOFT_LIMIT   = 40
    ACO_REVERT_SLACK  = 8

    # ACO signal parameters
    ACO_SIG_TS_MAX      = 50_000    # time_frac <= 0.05
    ACO_SIG_BID_VOL_MAX = 10        # bid_volume_1 <= 10
    ACO_SIG_MOM5_MAX    = -2        # momentum_5 <= -2
    ACO_SIG_MOM20_MIN   = 3         # momentum_20 > 3
    ACO_SIG_TAKE_BOOST  = 3         # extra ticks of buy aggressiveness when signal fires

    # Bid/ask prediction coefficients (validated, unchanged from v1–v3)
    ACO_BID_COEF = (-0.271, -0.217,  0.442, -7.156)
    ACO_ASK_COEF = (-0.276, -0.217, -0.570,  9.223)

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

    def _request_conversions(self, obs, fair: float, pos: int,
                             limit: int, max_units: int = None) -> int:
        if obs is None:
            return 0
        if max_units is None:
            max_units = self.CONV_MAX_UNITS
        buy_edge, sell_edge = self._conv_edge(obs, fair)
        if buy_edge > self.CONV_MIN_EDGE:
            return min(limit - pos, max_units)
        if sell_edge > self.CONV_MIN_EDGE:
            return -min(limit + pos, max_units)
        return 0

    # ═══════════════════════════════════════════════════════════
    #  IPR — Trend Rider + rule-signal hold window
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

        # ── Detect day's base price once ─────────────────────
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

        # ── Rule signal: book_imbalance + timing + depth ─────
        # Compute features from current order book
        bid_vol_1 = od.buy_orders.get(best_bid, 0)  if best_bid else 0
        ask_vol_1 = -od.sell_orders.get(best_ask, 0) if best_ask else 0
        total_vol = bid_vol_1 + ask_vol_1
        book_imbalance = bid_vol_1 / total_vol if total_vol > 0 else 0.5

        total_levels = len(od.buy_orders) + len(od.sell_orders)
        spread = (best_ask - best_bid) if (best_bid and best_ask) else 999
        spread_over_mid = spread / ob_mid if ob_mid else 1.0

        # Primary rule: early-day bid-heavy deep book
        ipr_signal = (
            self.IPR_SIG_TS_LO < ts <= self.IPR_SIG_TS_HI
            and book_imbalance > self.IPR_SIG_IMBALANCE
            and total_levels > self.IPR_SIG_MIN_LEVELS
        )
        # Secondary confirmation: tight spread
        ipr_signal_tight = ipr_signal and spread_over_mid <= self.IPR_SIG_SPREAD_MOM

        if ipr_signal:
            td['ipr_signal_ticks'] = td.get('ipr_signal_ticks', 0) + 1

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
            t1 = (buy_capacity * 2) // 3
            t2 = buy_capacity - t1
            result.append(Order(product, passive_buy,     t1))
            result.append(Order(product, passive_buy - 1, t2))

        # ── Micro sell: suppressed during signal window ───────
        # When signal fires, hold 100% of position through the predicted
        # XIRECS event. Only allow micro-sells outside the signal window.
        SELL_PREMIUM = 15
        if pos == limit and sell_capacity > 0 and not ipr_signal:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price >= FAIR + SELL_PREMIUM and sell_capacity > 0:
                    vol = min(od.buy_orders[bid_price], sell_capacity, 5)
                    result.append(Order(product, bid_price, -vol))
                    sell_capacity -= vol
                else:
                    break

        # ── Conversions: boosted during signal window ─────────
        conv = 0
        obs = state.observations.conversionObservations.get(product, None)
        if obs is not None:
            max_units = self.CONV_SIGNAL_UNITS if ipr_signal else self.CONV_MAX_UNITS
            conv = self._request_conversions(obs, FAIR, pos, limit, max_units)

        logger.print(
            f"IPR t={ts} base={ipr_base:.0f} FAIR={FAIR:.1f} pos={pos} "
            f"imbal={book_imbalance:.3f} sig={ipr_signal}"
        )
        return result, td, conv

    # ═══════════════════════════════════════════════════════════
    #  ACO — Symmetric market-maker + momentum dip signal
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

        # ── EMA drift (unchanged from v3) ────────────────────
        ob_mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else None
        ema = td.get('aco_ema', ANCHOR)
        if ob_mid is not None:
            ema = 0.1 * ob_mid + 0.9 * ema
        td['aco_ema'] = ema
        drift_count = td.get('aco_drift', 0)
        drift_count = (drift_count + 1) if abs(ema - ANCHOR) > 50 else 0
        td['aco_drift'] = drift_count
        FAIR = round(ema) if drift_count >= 100 else ANCHOR

        # ── Momentum tracking (new in v4) ─────────────────────
        # Keep a circular buffer of the last 20 mid prices to compute
        # momentum_5 (5-tick change) and momentum_20 (20-tick change).
        mid_hist = td.get('aco_mid_hist', [])
        if ob_mid is not None:
            mid_hist.append(ob_mid)
            if len(mid_hist) > 20:
                mid_hist = mid_hist[-20:]
        td['aco_mid_hist'] = mid_hist

        momentum_5  = (ob_mid - mid_hist[-6])  if (ob_mid and len(mid_hist) >= 6)  else 0.0
        momentum_20 = (ob_mid - mid_hist[0])   if (ob_mid and len(mid_hist) >= 20) else 0.0

        # ── ACO rule signal ───────────────────────────────────
        # time_frac <= 0.05: ts <= 50,000
        # bid_volume_1 <= 10: thin bid side
        # momentum_5 <= -2: short-term dip
        # momentum_20 > 3: medium-term upswing
        bid_vol_1 = od.buy_orders.get(best_bid, 0) if best_bid else 0
        aco_signal = (
            ts <= self.ACO_SIG_TS_MAX
            and bid_vol_1 <= self.ACO_SIG_BID_VOL_MAX
            and momentum_5  <= self.ACO_SIG_MOM5_MAX
            and momentum_20 >  self.ACO_SIG_MOM20_MIN
        )
        if aco_signal:
            td['aco_signal_ticks'] = td.get('aco_signal_ticks', 0) + 1

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

        # Spread-conditioned tweak
        if spread <= 12 and spread > 0:
            take_buy_threshold = FAIR + 2 - self.ACO_MIN_TAKE_EDGE
        elif spread >= 20:
            take_sell_threshold = FAIR - 2 + self.ACO_MIN_TAKE_EDGE

        # Reversion taker (v3): very short position → widen buy threshold
        if pos <= -self.ACO_SOFT_LIMIT:
            take_buy_threshold = FAIR + self.ACO_REVERT_SLACK   # 10008
        if pos >= self.ACO_SOFT_LIMIT:
            take_sell_threshold = FAIR - self.ACO_REVERT_SLACK   # 9992

        # Signal boost (v4): momentum dip in early session → buy more aggressively
        # Raises buy threshold by ACO_SIG_TAKE_BOOST (3 extra ticks).
        # This fires the taker buy on asks at FAIR+3 = 10003, capturing
        # the predicted dip-reversal before the market recovers.
        if aco_signal:
            take_buy_threshold = max(take_buy_threshold,
                                     FAIR + self.ACO_SIG_TAKE_BOOST)

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
        penny_buy = FAIR - buy_offset
        if best_bid is not None and best_bid + 1 <= FAIR:
            penny_buy = best_bid + 1

        penny_sell = FAIR + sell_offset
        if best_ask is not None and best_ask - 1 >= FAIR:
            penny_sell = best_ask - 1

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

        # ── Soft inventory gate (v3 sign-corrected) ──────────
        passive_buy_ok  = pos <  self.ACO_SOFT_LIMIT
        passive_sell_ok = pos > -self.ACO_SOFT_LIMIT

        # Signal: suppress passive sells during dip-buy signal
        # We don't want to deepen the short while the model says "buy the dip"
        if aco_signal:
            passive_sell_ok = False

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

        logger.print(
            f"ACO t={ts} FAIR={FAIR} pos={pos} spread={spread} "
            f"m5={momentum_5:.1f} m20={momentum_20:.1f} sig={aco_signal} "
            f"sell_ok={passive_sell_ok} buy_ok={passive_buy_ok} "
            f"take<{take_buy_threshold}/≥{take_sell_threshold}"
        )
        return result, td, conv
