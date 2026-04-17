"""
IMC Prosperity 4 — Round 1 Trader (predict_v1_3)
=================================================
Base: baseline.py (unchanged for both ACO and IPR logic).

The only addition is a size-skew signal in _trade_aco, gated on the
3 price deciles where v1.1 beat the baseline (confirmed from 176554.png):

  Decile 2: [9996.25, 9998.25)   — just below peg
  Decile 6: [10002.25, 10004.25) — just above peg
  Decile 8: [10006.75, ∞)        — highest prices

When obs_mid is in one of these ranges AND |signal| >= 1.0:
  - Bullish signal → keep full bid size, reduce ask to 40%
  - Bearish signal → keep full ask size, reduce bid to 40%

Signal = -0.45 * mid_change (reversion) + 0.05 * OFI
Day-boundary guard: prev_mid reset at ts=0.

IPR: byte-for-byte identical to baseline.
"""

import json
import math
from typing import Any
from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState
)


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict, conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions, "", "",
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list:
        return [
            state.timestamp, trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades):
        out = []
        for arr in trades.values():
            for t in arr:
                out.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return out

    def compress_observations(self, obs: Observation) -> list:
        co = {}
        for p, o in obs.conversionObservations.items():
            co[p] = [o.bidPrice, o.askPrice, o.transportFees,
                     o.exportTariff, o.importTariff, o.sunlight, o.humidity]
        return [obs.plainValueObservations, co]

    def compress_orders(self, orders):
        out = []
        for arr in orders.values():
            for o in arr:
                out.append([o.symbol, o.price, o.quantity])
        return out

    def to_json(self, value) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[:max_length - 3] + "..."


logger = Logger()


class Trader:

    LIMIT = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}

    ACO_FAIR   = 10000
    ACO_OFFSET = 4       # used by A and B; C overrides dynamically
    ACO_UNWIND_THRESHOLD = 30   # used by B and C

    # ── IPR config (baseline, unchanged) ────────────────────────
    IPR_BASE_POS              = 75
    IPR_MM_BUY_OFFSET         = 8
    IPR_MM_SELL_OFFSET        = 9
    IPR_PASSIVE_OFFSET        = 3
    IPR_SPRINT_CYCLE_TICKS    = 30000
    IPR_MIN_OBS               = 5
    IPR_HIGH_CONFIDENCE_R2    = 0.80
    IPR_CIRCUIT_BREAKER_SIGMA = 3.0

    def run(self, state: TradingState):
        orders: dict[Symbol, list[Order]] = {}
        conversions = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        for product in state.order_depths:
            if product == "ASH_COATED_OSMIUM":
                orders[product], td = self._trade_aco_A(state, td)
            elif product == "INTARIAN_PEPPER_ROOT":
                orders[product], td = self._trade_ipr(state, td)

        trader_data_out = json.dumps(td)
        logger.flush(state, orders, conversions, trader_data_out)
        return orders, conversions, trader_data_out

    # ─────────────────────────────────────────────────────────────
    #  ACO — baseline logic + gated size-skew signal
    # ─────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────
    #  STRATEGY A — Hardcode fair=10000, aggressive taking, tighter quotes
    #  Estimated PnL: ~3500–4000 / day
    #
    #  Changes vs current:
    #   • fair = 10000 (hardcoded) — replaces VW-mid calculation
    #   • Takes ALL orders that cross fair (not just those that cross VW-mid)
    #   • Offset reduced 7 → 4 (quotes at 9996/10004 instead of 9993/10007)
    #   • Removed: EMA, signal, OFI, gate, size-skew logic
    # ─────────────────────────────────────────────────────────────────────────────
    def _trade_aco_A(self, state, td):
        product = "ASH_COATED_OSMIUM"
        od      = state.order_depths[product]
        pos     = state.position.get(product, 0)
        limit   = self.LIMIT[product]
        result  = []
    
        FAIR   = 10000
        OFFSET = 4
    
        buy_capacity  = limit - pos
        sell_capacity = limit + pos
    
        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
    
        # ── Take: buy every ask below fair, sell every bid above fair ────
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price < FAIR and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol))
                buy_capacity -= vol
                logger.print(f"ACO-A TAKE BUY  {vol}@{ask_price}")
            else:
                break
    
        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price > FAIR and sell_capacity > 0:
                vol = min(od.buy_orders[bid_price], sell_capacity)
                result.append(Order(product, bid_price, -vol))
                sell_capacity -= vol
                logger.print(f"ACO-A TAKE SELL {vol}@{bid_price}")
            else:
                break
    
        # ── Passive quotes at fair ± OFFSET, penny inside spread if room ─
        buy_price  = FAIR - OFFSET
        sell_price = FAIR + OFFSET
    
        if best_bid is not None and best_bid + 1 < FAIR:
            buy_price = best_bid + 1
        if best_ask is not None and best_ask - 1 > FAIR:
            sell_price = best_ask - 1
    
        if buy_capacity > 0:
            t1 = buy_capacity // 2
            t2 = buy_capacity - t1
            result.append(Order(product, buy_price,     t1))
            result.append(Order(product, buy_price - 1, t2))
    
        if sell_capacity > 0:
            t1 = sell_capacity // 2
            t2 = sell_capacity - t1
            result.append(Order(product, sell_price,     -t1))
            result.append(Order(product, sell_price + 1, -t2))
    
        logger.print(f"ACO-A pos={pos} buy_q={buy_capacity} ask_q={sell_capacity} "
                    f"buy_p={buy_price} sell_p={sell_price}")
        return result, td
    
    # ─────────────────────────────────────────────────────────────
    #  IPR — byte-for-byte identical to baseline.py
    # ─────────────────────────────────────────────────────────────
    def _trade_ipr(self, state: TradingState, td: dict) -> tuple[list[Order], dict]:
        product = "INTARIAN_PEPPER_ROOT"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT[product]
        base = self.IPR_BASE_POS
        ts = state.timestamp
        result: list[Order] = []

        buy_capacity  = limit - pos
        sell_capacity = limit + pos

        ob_mid = None
        best_ask_ipr = None
        best_bid_ipr = None
        if od.sell_orders and od.buy_orders:
            best_ask_ipr = min(od.sell_orders.keys())
            best_bid_ipr = max(od.buy_orders.keys())
            ob_mid = (best_ask_ipr + best_bid_ipr) / 2

        last_ts = td.get('ipr_last_ts', -1)
        if ts < last_ts or ts == 0:
            td['ipr_n']     = 0
            td['ipr_sx']    = 0.0
            td['ipr_sy']    = 0.0
            td['ipr_sxy']   = 0.0
            td['ipr_sx2']   = 0.0
            td['ipr_sy2']   = 0.0
            td['ipr_phase'] = 'ramp'
        td['ipr_last_ts'] = ts

        if ob_mid is not None:
            n   = td.get('ipr_n', 0) + 1
            sx  = td.get('ipr_sx', 0.0)  + ts
            sy  = td.get('ipr_sy', 0.0)  + ob_mid
            sxy = td.get('ipr_sxy', 0.0) + ts * ob_mid
            sx2 = td.get('ipr_sx2', 0.0) + ts * ts
            sy2 = td.get('ipr_sy2', 0.0) + ob_mid * ob_mid
            td['ipr_n']   = n
            td['ipr_sx']  = sx
            td['ipr_sy']  = sy
            td['ipr_sxy'] = sxy
            td['ipr_sx2'] = sx2
            td['ipr_sy2'] = sy2
        else:
            n   = td.get('ipr_n', 0)
            sx  = td.get('ipr_sx', 0.0)
            sy  = td.get('ipr_sy', 0.0)
            sxy = td.get('ipr_sxy', 0.0)
            sx2 = td.get('ipr_sx2', 0.0)
            sy2 = td.get('ipr_sy2', 0.0)

        slope = None
        intercept = None
        r_squared = 0.0
        resid_std = float('inf')

        denom = n * sx2 - sx * sx
        if n >= self.IPR_MIN_OBS and denom > 0:
            slope     = (n * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / n
            ss_tot = sy2 - n * (sy/n) ** 2
            ss_res = sy2 - 2 * slope * sxy - 2 * intercept * sy \
                     + slope * slope * sx2 + 2 * slope * intercept * sx \
                     + n * intercept * intercept
            if ss_tot > 0:
                r_squared = max(0.0, 1.0 - ss_res / ss_tot)
                resid_std = math.sqrt(max(0.0, ss_res / n))

        model_fair = None
        if slope is not None and intercept is not None:
            model_fair = intercept + slope * ts

        if model_fair is not None and ob_mid is not None:
            w_model = min(r_squared, 0.95)
            fair = w_model * model_fair + (1 - w_model) * ob_mid
        elif model_fair is not None:
            fair = model_fair
        elif ob_mid is not None:
            fair = ob_mid
        else:
            return result, td

        circuit_tripped = False
        if model_fair is not None and ob_mid is not None and resid_std < float('inf') and resid_std > 0:
            if abs(ob_mid - model_fair) > self.IPR_CIRCUIT_BREAKER_SIGMA * resid_std:
                circuit_tripped = True

        phase = td.get('ipr_phase', 'ramp')
        ols_confident = (slope is not None and r_squared >= self.IPR_HIGH_CONFIDENCE_R2)

        if phase == 'ramp' and pos >= base:
            phase = 'cycle'
            td['ipr_phase'] = 'cycle'

        if phase == 'cycle' and pos <= base:
            idle_since = td.get('ipr_idle_since', ts)
            if ts == idle_since:
                td['ipr_idle_since'] = ts
            elif ts - idle_since >= self.IPR_SPRINT_CYCLE_TICKS:
                phase = 'sprint'
                td['ipr_phase'] = 'sprint'
        elif phase == 'cycle' and pos > base:
            td['ipr_idle_since'] = ts

        if ols_confident and slope is not None and slope <= 0:
            phase = 'flatten'

        logger.print(
            f"IPR ts={ts} fair={fair:.1f} slope={slope if slope is not None else '?'}"
            f" R2={r_squared:.3f} phase={phase} pos={pos} mid={ob_mid}"
        )

        if phase == 'ramp':
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price < fair and buy_capacity > 0 and pos < base:
                    vol = min(-od.sell_orders[ask_price], buy_capacity, base - pos)
                    if vol > 0:
                        result.append(Order(product, ask_price, vol))
                        buy_capacity -= vol
                        pos += vol
                else:
                    break
            if buy_capacity > 0 and pos < base and od.sell_orders and not circuit_tripped:
                best_ask = min(od.sell_orders.keys())
                want = min(base - pos, buy_capacity)
                take_vol = min(-od.sell_orders[best_ask], want)
                if take_vol > 0:
                    result.append(Order(product, best_ask, take_vol))
                    buy_capacity -= take_vol
                    pos += take_vol
                    logger.print(f"IPR RAMP BUY {take_vol}@{best_ask}")
            if buy_capacity > 0 and pos < base:
                buy_price = math.floor(fair) - self.IPR_PASSIVE_OFFSET
                if best_bid_ipr is not None and best_bid_ipr + 1 < fair:
                    buy_price = best_bid_ipr + 1
                result.append(Order(product, buy_price, min(buy_capacity, base - pos)))
            return result, td

        if phase == 'cycle':
            mm_buy_price  = math.floor(fair) - self.IPR_MM_BUY_OFFSET
            mm_sell_price = math.ceil(fair)  + self.IPR_MM_SELL_OFFSET
            if best_ask_ipr is not None and best_ask_ipr - 1 > fair:
                mm_sell_price = min(mm_sell_price, best_ask_ipr - 1)
            if best_bid_ipr is not None and best_bid_ipr + 1 < fair:
                mm_buy_price = max(mm_buy_price, best_bid_ipr + 1)
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price < fair and buy_capacity > 0 and pos < limit:
                    vol = min(-od.sell_orders[ask_price], buy_capacity, limit - pos)
                    if vol > 0:
                        result.append(Order(product, ask_price, vol))
                        buy_capacity -= vol
                        pos += vol
                else:
                    break
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price > fair and sell_capacity > 0 and pos > base:
                    vol = min(od.buy_orders[bid_price], sell_capacity, pos - base)
                    if vol > 0:
                        result.append(Order(product, bid_price, -vol))
                        sell_capacity -= vol
                        pos -= vol
                else:
                    break
            if pos < limit and buy_capacity > 0:
                qty = min(buy_capacity, limit - pos)
                result.append(Order(product, mm_buy_price, qty))
            if pos > base and sell_capacity > 0:
                qty = min(sell_capacity, pos - base)
                result.append(Order(product, mm_sell_price, -qty))
            return result, td

        if phase == 'flatten':
            if pos > 0 and sell_capacity > 0:
                sell_price = math.ceil(fair) + self.IPR_MM_SELL_OFFSET
                if best_ask_ipr is not None and best_ask_ipr - 1 > fair:
                    sell_price = min(sell_price, best_ask_ipr - 1)
                result.append(Order(product, sell_price, -min(pos, sell_capacity)))
            return result, td

        if phase == 'sprint':
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price < fair and buy_capacity > 0:
                    vol = min(-od.sell_orders[ask_price], buy_capacity)
                    result.append(Order(product, ask_price, vol))
                    buy_capacity -= vol
                    pos += vol
                else:
                    break
            if buy_capacity > 0 and pos < limit and od.sell_orders:
                best_ask = min(od.sell_orders.keys())
                take_vol = min(-od.sell_orders[best_ask], buy_capacity)
                if take_vol > 0:
                    result.append(Order(product, best_ask, take_vol))
                    buy_capacity -= take_vol
                    pos += take_vol
                    logger.print(f"IPR SPRINT BUY {take_vol}@{best_ask}")
            if buy_capacity > 0 and pos < limit:
                buy_price = math.floor(fair) - 1
                if best_bid_ipr is not None and best_bid_ipr + 1 < fair:
                    buy_price = max(buy_price, best_bid_ipr + 1)
                result.append(Order(product, buy_price, buy_capacity))
            return result, td

        return result, td