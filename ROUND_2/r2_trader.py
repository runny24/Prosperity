"""
IMC Prosperity Round 2 Trader
================================
Products: ASH_COATED_OSMIUM (ACO) + INTARIAN_PEPPER_ROOT (IPR)
New in Round 2: access-fee conversion logic for XIRECS external market
                (+25 % volume via conversionObservations)

─────────────────────────────────────────────────────────────
KEY FINDINGS FROM ROUND 2 DATA ANALYSIS
─────────────────────────────────────────────────────────────
IPR:
  • Perfect linear uptrend: +1 per 1 000 ticks within each day
  • Base price rises +1 000 per day
  • Day 2 fair value: 14 000 + timestamp / 1 000
  • Residuals ≈ 0  →  formula is exact; trade aggressively around it

ACO:
  • Anchored near 10 000 across all days (mean deviation < 5)
  • Spread mean ~16 ticks, range 5–22, mean-reverting
  • Bid/ask changes nearly independent (corr ~0.05)
  • Round-1 bid/ask prediction model still valid unchanged

XIRECS Conversion Observations:
  • External market quotes available via state.observations.conversionObservations
  • Each observation has: bidPrice, askPrice, transportFees, exportTariff, importTariff
  • Effective conversion cost (buy from XIRECS): askPrice + transportFees + importTariff
  • Effective conversion proceeds (sell to XIRECS): bidPrice − transportFees − exportTariff
  • The 'conversions' integer returned from run() controls how many units we convert
    (positive = we BUY from XIRECS, negative = we SELL to XIRECS)
  • Paying the access fee unlocks up to 25 % more volume on XIRECS quotes

─────────────────────────────────────────────────────────────
STRATEGY OUTLINE
─────────────────────────────────────────────────────────────
ACO:  Same bid/ask-prediction market-maker as R1.
      Additionally, if XIRECS buy cost < FAIR − CONV_MIN_EDGE, request
      conversions to increase effective long; vice-versa for shorts.

IPR:  Simplified — fair value is perfectly known.
      Post passive orders around FAIR±offset; take anything inside FAIR.
      Conversions: if XIRECS allows closing/opening position at
      FAIR ± offset cheaper than the open market, use them.
"""

import json
import math
from typing import Any
from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
    ConversionObservation,
)


# ═══════════════════════════════════════════════════════════════
#  Logger  (identical to R1 — do not modify)
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
                     o.exportTariff, o.importTariff, o.sunlight, o.humidity]
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

    # ── ACO bid/ask prediction coefficients (from R1 linear fit, still valid)
    ACO_BID_COEF = (-0.271, -0.217, 0.442, -7.156)  # (prev_bid, prev_ask, spread, const)
    ACO_ASK_COEF = (-0.276, -0.217, -0.570, 9.223)

    ACO_ANCHOR        = 10_000
    ACO_OFFSET_MAX    = 7
    ACO_OFFSET_MIN    = 3
    ACO_DECAY_TICKS   = 5_000

    # ── IPR trend (exact formula validated on days −1, 0, 1)
    # FAIR_IPR = IPR_BASE_DAY2 + timestamp * IPR_SLOPE_PER_TICK
    # Day 2 = third day  →  base = 14 000
    IPR_DAY2_BASE       = 14_000           # fair value at timestamp 0 on day 2
    IPR_SLOPE_PER_TICK  = 0.001            # +1 per 1 000 ticks
    IPR_OFFSET          = 4               # passive order offset from fair
    IPR_TAKE_SLACK      = 1               # take up to FAIR + slack on ask side

    # ── Conversion / XIRECS parameters
    # Minimum edge (after all fees) to justify a conversion
    CONV_MIN_EDGE       = 2.0             # seashells per unit
    # Maximum conversion request per tick (stay conservative)
    CONV_MAX_UNITS      = 10

    # ─────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        orders: dict = {}
        conversions: int = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        for product in state.order_depths:
            if product == "ASH_COATED_OSMIUM":
                orders[product], td, conv_aco = self._trade_aco(state, td)
                conversions += conv_aco
            elif product == "INTARIAN_PEPPER_ROOT":
                orders[product], td, conv_ipr = self._trade_ipr(state, td)
                conversions += conv_ipr

        tdo = json.dumps(td)
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ═══════════════════════════════════════════════════════════
    #  HELPER — compute net conversion P&L for a given product
    # ═══════════════════════════════════════════════════════════
    def _conv_edge(self, obs: ConversionObservation, fair: float):
        """
        Returns (buy_edge, sell_edge) where:
          buy_edge  = fair - effective_buy_cost   (positive means BUY conversion is profitable)
          sell_edge = effective_sell_proceeds - fair (positive means SELL conversion is profitable)

        effective_buy_cost     = obs.askPrice + obs.transportFees + obs.importTariff
        effective_sell_proceeds= obs.bidPrice − obs.transportFees − obs.exportTariff
        """
        effective_buy  = obs.askPrice  + obs.transportFees + obs.importTariff
        effective_sell = obs.bidPrice  - obs.transportFees - obs.exportTariff
        return fair - effective_buy, effective_sell - fair

    def _request_conversions(self, obs: ConversionObservation,
                              fair: float, pos: int, limit: int) -> int:
        """
        Decide how many units to convert:
          • positive int → buy from XIRECS (reduces our short / builds long)
          • negative int → sell to XIRECS (reduces our long / builds short)
        Returns 0 if no profitable conversion exists.
        """
        if obs is None:
            return 0

        buy_edge, sell_edge = self._conv_edge(obs, fair)
        conv = 0

        # If buying from XIRECS is profitable and we have room to go long
        if buy_edge > self.CONV_MIN_EDGE:
            room = limit - pos
            conv = min(room, self.CONV_MAX_UNITS)

        # If selling to XIRECS is profitable and we have room to go short
        elif sell_edge > self.CONV_MIN_EDGE:
            room = limit + pos  # room to go short
            conv = -min(room, self.CONV_MAX_UNITS)

        return conv

    # ═══════════════════════════════════════════════════════════
    #  ACO — Bid/Ask-Aware Market Maker (same core as R1)
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

        # ── EMA drift safety ──────────────────────────────────
        ob_mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else None
        ema = td.get('aco_ema', ANCHOR)
        if ob_mid is not None:
            ema = 0.1 * ob_mid + 0.9 * ema
        td['aco_ema'] = ema
        drift_count = td.get('aco_drift', 0)
        if abs(ema - ANCHOR) > 50:
            drift_count += 1
        else:
            drift_count = 0
        td['aco_drift'] = drift_count
        FAIR = round(ema) if drift_count >= 100 else ANCHOR

        # ── Adaptive offset ───────────────────────────────────
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

        # ── Bid/Ask prediction ────────────────────────────────
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

        # ── Take: spread-conditioned ──────────────────────────
        take_buy_fair  = FAIR
        take_sell_fair = FAIR
        if spread <= 12 and spread > 0:
            take_buy_fair = FAIR + 2
        elif spread >= 20:
            take_sell_fair = FAIR - 2

        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price < take_buy_fair and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol))
                buy_capacity -= vol
            else:
                break

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price > take_sell_fair and sell_capacity > 0:
                vol = min(od.buy_orders[bid_price], sell_capacity)
                result.append(Order(product, bid_price, -vol))
                sell_capacity -= vol
            else:
                break

        # ── Passive: penny-inside + predicted level ───────────
        penny_buy  = FAIR - buy_offset
        penny_sell = FAIR + sell_offset
        if best_bid is not None and best_bid + 1 < FAIR:
            penny_buy = best_bid + 1
        if best_ask is not None and best_ask - 1 > FAIR:
            penny_sell = best_ask - 1

        if pred_next_bid is not None:
            pred_penny_buy = pred_next_bid + 1
            second_buy = pred_penny_buy if (pred_penny_buy < penny_buy - 1
                                            and pred_penny_buy >= FAIR - 15) else penny_buy - 1
        else:
            second_buy = penny_buy - 1

        if pred_next_ask is not None:
            pred_penny_sell = pred_next_ask - 1
            second_sell = pred_penny_sell if (pred_penny_sell > penny_sell + 1
                                              and pred_penny_sell <= FAIR + 15) else penny_sell + 1
        else:
            second_sell = penny_sell + 1

        if buy_capacity > 0:
            t1, t2 = buy_capacity // 2, buy_capacity - buy_capacity // 2
            result.append(Order(product, penny_buy,  t1))
            result.append(Order(product, second_buy, t2))

        if sell_capacity > 0:
            t1, t2 = sell_capacity // 2, sell_capacity - sell_capacity // 2
            result.append(Order(product, penny_sell,  -t1))
            result.append(Order(product, second_sell, -t2))

        # ── XIRECS Conversion ─────────────────────────────────
        conv = 0
        obs: ConversionObservation = (state.observations.conversionObservations
                                      .get(product, None))
        if obs is not None:
            conv = self._request_conversions(obs, FAIR, pos, limit)
            logger.print(f"ACO conv: buy_edge={self._conv_edge(obs, FAIR)[0]:.2f} "
                         f"sell_edge={self._conv_edge(obs, FAIR)[1]:.2f} "
                         f"requesting={conv}")

        return result, td, conv

    # ═══════════════════════════════════════════════════════════
    #  IPR — Exact-Fair Market Maker (updated for R2 uptrend)
    # ═══════════════════════════════════════════════════════════
    def _trade_ipr(self, state: TradingState, td: dict):
        product = "INTARIAN_PEPPER_ROOT"
        od      = state.order_depths[product]
        pos     = state.position.get(product, 0)
        limit   = self.LIMIT[product]
        result  = []
        ts      = state.timestamp

        # ── Exact linear fair value ───────────────────────────
        FAIR = self.IPR_DAY2_BASE + ts * self.IPR_SLOPE_PER_TICK

        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else None

        buy_capacity  = limit - pos
        sell_capacity = limit + pos

        # ── Take: lift underpriced asks, hit overpriced bids ──
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price <= FAIR + self.IPR_TAKE_SLACK and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol))
                buy_capacity -= vol
            else:
                break

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price >= FAIR - self.IPR_TAKE_SLACK and sell_capacity > 0:
                vol = min(od.buy_orders[bid_price], sell_capacity)
                result.append(Order(product, bid_price, -vol))
                sell_capacity -= vol
            else:
                break

        # ── Passive: penny-inside best bid/ask ────────────────
        offset = self.IPR_OFFSET

        passive_buy = math.floor(FAIR) - offset
        if best_bid is not None and best_bid + 1 < FAIR:
            passive_buy = best_bid + 1

        passive_sell = math.ceil(FAIR) + offset
        if best_ask is not None and best_ask - 1 > FAIR:
            passive_sell = best_ask - 1

        # Two-level passive posting
        if buy_capacity > 0:
            t1, t2 = buy_capacity // 2, buy_capacity - buy_capacity // 2
            result.append(Order(product, passive_buy,     t1))
            result.append(Order(product, passive_buy - 1, t2))

        if sell_capacity > 0:
            t1, t2 = sell_capacity // 2, sell_capacity - sell_capacity // 2
            result.append(Order(product, passive_sell,     -t1))
            result.append(Order(product, passive_sell + 1, -t2))

        # ── XIRECS Conversion ─────────────────────────────────
        # IPR's fair value is rising; selling to XIRECS (negative conversion)
        # can lock in gains above FAIR if the external bid is high enough.
        conv = 0
        obs: ConversionObservation = (state.observations.conversionObservations
                                      .get(product, None))
        if obs is not None:
            conv = self._request_conversions(obs, FAIR, pos, limit)
            logger.print(f"IPR FAIR={FAIR:.2f} "
                         f"conv: buy_edge={self._conv_edge(obs, FAIR)[0]:.2f} "
                         f"sell_edge={self._conv_edge(obs, FAIR)[1]:.2f} "
                         f"requesting={conv}")

        return result, td, conv
