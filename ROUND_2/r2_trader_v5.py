"""
IMC Prosperity Round 2 — Trader v5
====================================
Built on v4 (P&L: IPR=7,270, ACO=1,559 — 286109.log)

Root cause analysis of v4 log reveals two structural ACO problems:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUG A — Passive buy falls 8 ticks behind market when bid >= FAIR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  v4 condition: `best_bid + 1 <= FAIR` → penny_buy = best_bid+1
  When bid = 10000 (=FAIR): 10001 <= 10000 is FALSE
  → falls back to FAIR - buy_offset = 9993 (8 ticks below market!)

  12.2% of ticks have bid >= 10000. During these ticks our passive
  buys never fill; every sell order that fires deepens the short.

  Fix: penny_buy = min(best_bid + 1, FAIR + 3)
  When bid=10000: penny_buy = 10001 → we BECOME the new best bid
  → bots selling will hit US instead of the market maker
  → fills 9 ticks closer to market on those 117 ticks per 1000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUG B — Passive sell follows ask down to terrible prices
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  penny_sell = best_ask - 1 has no lower bound.
  When ask temporarily dips to 10004: penny_sell = 10003 (only 3
  ticks edge!). Second_sell via prediction can be just as low.

  v4 log: 145 of 229 sold units went at avg 10004.39 (below 10008).
  Lost ~523 P&L ticks vs a floor of FAIR + 8.

  Worse: we're chronically short (-36 avg pos). Every bad sell at
  10003 deepens the short AND earns minimal edge.

  Fix: Apply hard floor to BOTH passive sell levels.
    penny_sell  = max(penny_sell,  FAIR + ACO_SELL_FLOOR)  [=10008]
    second_sell = max(second_sell, FAIR + ACO_SELL_FLOOR + 1) [=10009]
  When ask < 10009: we place no-fill sell orders above the ask —
  effectively sitting out bad-spread periods rather than selling cheap.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Additional changes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  C. Graduated reversion taker (v3 had a flat FAIR+8 for pos<=-40):
       pos <= -55: take_buy_threshold = FAIR + 15  (take almost any ask)
       pos <= -40: take_buy_threshold = FAIR + 12  (more aggressive than +8)
     v4 avg ask is 10012, so FAIR+8 threshold rarely triggers real fills.
     FAIR+12 captures ~40% of all asks; FAIR+15 captures ~60%.

  D. Passive sell skew when short: when pos < -20, reduce passive
     sell size to 1/3 of capacity (down from 1/2). Reduces drip-selling
     into the short without fully suppressing inventory management.
     Symmetric: when pos > 20, reduce passive buy to 1/3.

All v4 changes (IPR hold-window, ACO momentum signal, v3 fixes) kept.
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
    IPR_SLOPE_PER_TICK  = 0.001
    IPR_TAKE_SLACK      = 10
    IPR_PASSIVE_OFFSET  = 4

    IPR_SIG_TS_LO       = 50_000
    IPR_SIG_TS_HI       = 150_000
    IPR_SIG_IMBALANCE   = 0.47
    IPR_SIG_MIN_LEVELS  = 3
    IPR_SIG_SPREAD_MOM  = 0.0011

    CONV_MIN_EDGE       = 2.0
    CONV_MAX_UNITS      = 10
    CONV_SIGNAL_UNITS   = 20

    # ── ACO ───────────────────────────────────────────────────
    ACO_ANCHOR          = 10_000
    ACO_OFFSET_MAX      = 7
    ACO_OFFSET_MIN      = 3
    ACO_DECAY_TICKS     = 5_000
    ACO_MIN_TAKE_EDGE   = 3
    ACO_SOFT_LIMIT      = 40

    # FIX A: cap for passive buy above fair — never pay > FAIR+3 passively
    ACO_BUY_CAP_ABOVE   = 3

    # FIX B: floor for passive sell — never sell below FAIR+8 passively
    # This prevents penny sell from chasing the ask down to 10003 range
    ACO_SELL_FLOOR      = 8

    # FIX C: graduated reversion taker thresholds (raised from flat FAIR+8)
    ACO_REVERT_THRESH_MID  = 12   # when pos <= -SOFT_LIMIT:   FAIR+12
    ACO_REVERT_THRESH_HIGH = 15   # when pos <= -SOFT_LIMIT-15: FAIR+15

    # ACO signal (v4, kept)
    ACO_SIG_TS_MAX      = 50_000
    ACO_SIG_BID_VOL_MAX = 10
    ACO_SIG_MOM5_MAX    = -2
    ACO_SIG_MOM20_MIN   = 3
    ACO_SIG_TAKE_BOOST  = 3

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
    #  IPR — Trend Rider (unchanged from v4)
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

        bid_vol_1 = od.buy_orders.get(best_bid, 0)  if best_bid else 0
        ask_vol_1 = -od.sell_orders.get(best_ask, 0) if best_ask else 0
        total_vol = bid_vol_1 + ask_vol_1
        book_imbalance = bid_vol_1 / total_vol if total_vol > 0 else 0.5
        total_levels = len(od.buy_orders) + len(od.sell_orders)
        spread = (best_ask - best_bid) if (best_bid and best_ask) else 999
        spread_over_mid = spread / ob_mid if ob_mid else 1.0

        ipr_signal = (
            self.IPR_SIG_TS_LO < ts <= self.IPR_SIG_TS_HI
            and book_imbalance > self.IPR_SIG_IMBALANCE
            and total_levels > self.IPR_SIG_MIN_LEVELS
        )
        if ipr_signal:
            td['ipr_signal_ticks'] = td.get('ipr_signal_ticks', 0) + 1

        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price <= FAIR + self.IPR_TAKE_SLACK and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol))
                buy_capacity -= vol
            else:
                break

        if buy_capacity > 0:
            passive_buy = math.floor(FAIR) - self.IPR_PASSIVE_OFFSET
            if best_bid is not None and best_bid + 1 <= FAIR:
                passive_buy = best_bid + 1
            t1 = (buy_capacity * 2) // 3
            t2 = buy_capacity - t1
            result.append(Order(product, passive_buy,     t1))
            result.append(Order(product, passive_buy - 1, t2))

        SELL_PREMIUM = 15
        if pos == limit and sell_capacity > 0 and not ipr_signal:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price >= FAIR + SELL_PREMIUM and sell_capacity > 0:
                    vol = min(od.buy_orders[bid_price], sell_capacity, 5)
                    result.append(Order(product, bid_price, -vol))
                    sell_capacity -= vol
                else:
                    break

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
    #  ACO — Market maker with structural bug fixes
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

        # ── EMA drift ────────────────────────────────────────
        ob_mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else None
        ema = td.get('aco_ema', ANCHOR)
        if ob_mid is not None:
            ema = 0.1 * ob_mid + 0.9 * ema
        td['aco_ema'] = ema
        drift_count = td.get('aco_drift', 0)
        drift_count = (drift_count + 1) if abs(ema - ANCHOR) > 50 else 0
        td['aco_drift'] = drift_count
        FAIR = round(ema) if drift_count >= 100 else ANCHOR

        # ── Momentum tracking (v4, kept) ─────────────────────
        mid_hist = td.get('aco_mid_hist', [])
        if ob_mid is not None:
            mid_hist.append(ob_mid)
            if len(mid_hist) > 20:
                mid_hist = mid_hist[-20:]
        td['aco_mid_hist'] = mid_hist

        momentum_5  = (ob_mid - mid_hist[-6])  if (ob_mid and len(mid_hist) >= 6)  else 0.0
        momentum_20 = (ob_mid - mid_hist[0])   if (ob_mid and len(mid_hist) >= 20) else 0.0

        # ── ACO rule signal (v4, kept) ────────────────────────
        bid_vol_1 = od.buy_orders.get(best_bid, 0) if best_bid else 0
        aco_signal = (
            ts <= self.ACO_SIG_TS_MAX
            and bid_vol_1 <= self.ACO_SIG_BID_VOL_MAX
            and momentum_5  <= self.ACO_SIG_MOM5_MAX
            and momentum_20 >  self.ACO_SIG_MOM20_MIN
        )
        if aco_signal:
            td['aco_signal_ticks'] = td.get('aco_signal_ticks', 0) + 1

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
        take_buy_threshold  = FAIR - self.ACO_MIN_TAKE_EDGE
        take_sell_threshold = FAIR + self.ACO_MIN_TAKE_EDGE

        if spread <= 12 and spread > 0:
            take_buy_threshold = FAIR + 2 - self.ACO_MIN_TAKE_EDGE
        elif spread >= 20:
            take_sell_threshold = FAIR - 2 + self.ACO_MIN_TAKE_EDGE

        # FIX C: Graduated reversion taker
        # v4 used flat FAIR+8. Mean ask is ~10012 so that rarely triggered.
        # Now FAIR+12 fires ~40% of the time; FAIR+15 fires ~60% of the time.
        if pos <= -(self.ACO_SOFT_LIMIT + 15):        # pos <= -55
            take_buy_threshold = FAIR + self.ACO_REVERT_THRESH_HIGH   # 10015
        elif pos <= -self.ACO_SOFT_LIMIT:              # pos <= -40
            take_buy_threshold = FAIR + self.ACO_REVERT_THRESH_MID    # 10012

        # Symmetric long-side reversion
        if pos >= (self.ACO_SOFT_LIMIT + 15):
            take_sell_threshold = FAIR - self.ACO_REVERT_THRESH_HIGH
        elif pos >= self.ACO_SOFT_LIMIT:
            take_sell_threshold = FAIR - self.ACO_REVERT_THRESH_MID

        # v4 signal boost (kept)
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

        # ── FIX A: Passive buy — always penny inside bid, cap at FAIR+3 ──
        # Old: penny_buy = best_bid+1 ONLY if best_bid+1 <= FAIR, else FAIR-7
        # Bug: when bid=10000 (FAIR), 10001 > 10000 → falls back to 9993!
        #      Our passive buy is 8 ticks behind market, never fills.
        # Fix: always penny inside bid, just cap to avoid chasing too high.
        if best_bid is not None:
            penny_buy = min(best_bid + 1, FAIR + self.ACO_BUY_CAP_ABOVE)
        else:
            penny_buy = FAIR - buy_offset

        # ── FIX B: Passive sell — hard floor to avoid terrible edge ──────
        # Old: penny_sell = best_ask - 1 (no floor)
        # Bug: when ask dips to 10004 → sell at 10003 (only 3 ticks edge!)
        #      145 of 229 units sold below 10008, losing ~523 P&L ticks
        # Fix: floor at FAIR + ACO_SELL_FLOOR (= 10008)
        if best_ask is not None and best_ask - 1 >= FAIR:
            penny_sell = best_ask - 1
        else:
            penny_sell = FAIR + sell_offset
        penny_sell = max(penny_sell, FAIR + self.ACO_SELL_FLOOR)

        # Second level via prediction
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
        # Apply sell floor to second level too
        second_sell = max(second_sell, FAIR + self.ACO_SELL_FLOOR + 1)

        # ── Soft inventory gate (v3 sign-corrected) ──────────
        passive_buy_ok  = pos <  self.ACO_SOFT_LIMIT
        passive_sell_ok = pos > -self.ACO_SOFT_LIMIT

        # v4 signal: suppress sells during early-day dip
        if aco_signal:
            passive_sell_ok = False

        # FIX D: Position-skewed sizing — reduce sell orders when short,
        # reduce buy orders when long. Prevents drip-selling into a short.
        if pos < -20:
            # Short: cut sell to 1/3 of capacity, keep full buy
            sell_t1 = sell_capacity // 3
            sell_t2 = sell_t1
            buy_t1  = buy_capacity // 2
            buy_t2  = buy_capacity - buy_t1
        elif pos > 20:
            # Long: cut buy to 1/3 of capacity, keep full sell
            buy_t1  = buy_capacity // 3
            buy_t2  = buy_t1
            sell_t1 = sell_capacity // 2
            sell_t2 = sell_capacity - sell_t1
        else:
            buy_t1  = buy_capacity // 2
            buy_t2  = buy_capacity - buy_t1
            sell_t1 = sell_capacity // 2
            sell_t2 = sell_capacity - sell_t1

        if buy_capacity > 0 and passive_buy_ok:
            result.append(Order(product, penny_buy,  buy_t1))
            result.append(Order(product, second_buy, buy_t2))

        if sell_capacity > 0 and passive_sell_ok:
            result.append(Order(product, penny_sell,  -sell_t1))
            result.append(Order(product, second_sell, -sell_t2))

        # ── Conversions ───────────────────────────────────────
        conv = 0
        obs = state.observations.conversionObservations.get(product, None)
        if obs is not None:
            conv = self._request_conversions(obs, FAIR, pos, limit)

        logger.print(
            f"ACO t={ts} FAIR={FAIR} pos={pos} spread={spread} "
            f"pbuy={penny_buy} psell={penny_sell} "
            f"sell_ok={passive_sell_ok} buy_ok={passive_buy_ok} "
            f"take<{take_buy_threshold}/≥{take_sell_threshold}"
        )
        return result, td, conv
