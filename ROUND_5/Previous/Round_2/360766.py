"""
v14 — Near-Zero ACO + Optimized IPR Cycling
=============================================
ACO:
- FAIR=10000 fixed
- SOFT_LIMIT=25 (tighter — keeps position near zero where fill quality
  is best: +6.1pt sell edge at pos 0 vs +2.0pt at pos -50)
- MIN_TAKE_EDGE=3 (proven: avoids cheap sells on shifted sims)
- REVERT_SLACK=8 (recover from extreme positions)
- Adaptive offset 7→3
- No hardcoded timestamps

IPR:
- Mid-based OLS cycling (proven 7623)
- Ramp capped at 12u/tick (avoids level-2 asks, saves ~107 PnL)
- Aggressive flatten + drawdown safety
"""

import json, math
from typing import Any
from datamodel import (Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState)

class Logger:
    def __init__(self): self.logs = ""; self.max_log_length = 3750
    def print(self, *objects, sep=" ", end="\n"): self.logs += sep.join(map(str, objects)) + end
    def flush(self, state, orders, conversions, trader_data):
        bl = len(self.to_json([self.cs(state, ""), self.co(orders), conversions, "", ""]))
        m = (self.max_log_length - bl) // 3
        print(self.to_json([self.cs(state, self.t(state.traderData, m)), self.co(orders), conversions, self.t(trader_data, m), self.t(self.logs, m)]))
        self.logs = ""
    def cs(self, s, td):
        return [s.timestamp, td, [[l.symbol,l.product,l.denomination] for l in s.listings.values()],
                {k:[v.buy_orders,v.sell_orders] for k,v in s.order_depths.items()},
                [[t.symbol,t.price,t.quantity,t.buyer,t.seller,t.timestamp] for a in s.own_trades.values() for t in a],
                [[t.symbol,t.price,t.quantity,t.buyer,t.seller,t.timestamp] for a in s.market_trades.values() for t in a],
                s.position, self.cobs(s.observations)]
    def cobs(self, obs):
        co = {}
        for p, o in obs.conversionObservations.items():
            co[p] = [o.bidPrice,o.askPrice,o.transportFees,o.exportTariff,o.importTariff,o.sunlight,o.humidity]
        return [obs.plainValueObservations, co]
    def co(self, orders): return [[o.symbol,o.price,o.quantity] for a in orders.values() for o in a]
    def to_json(self, v): return json.dumps(v, cls=ProsperityEncoder, separators=(",",":"))
    def t(self, v, m): return v if len(v) <= m else v[:m-3] + "..."

logger = Logger()

class Trader:
    LIMIT = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}

    # ACO — near-zero focus
    ACO_ANCHOR = 10000
    ACO_OFFSET_MAX = 7; ACO_OFFSET_MIN = 3; ACO_DECAY_TICKS = 5000
    ACO_MIN_TAKE_EDGE = 3
    ACO_SOFT_LIMIT = 25        # TIGHTER: stay near zero for best fill quality
    ACO_REVERT_SLACK = 8

    # IPR — optimized cycling
    IPR_BASE_POS = 75; IPR_PASSIVE_OFFSET = 3; IPR_RAMP_MAX_PER_TICK = 12
    IPR_BUY_OFFSET_MAX = 10; IPR_BUY_OFFSET_MIN = 1
    IPR_SELL_OFFSET_MAX = 8; IPR_SELL_OFFSET_MIN = 3
    IPR_OFFSET_DECAY_TICKS = 5000; IPR_SPRINT_CYCLE_TICKS = 30000
    IPR_MIN_OBS = 5; IPR_HIGH_CONFIDENCE_R2 = 0.80; IPR_CIRCUIT_BREAKER_SIGMA = 3.0

    def run(self, state):
        orders = {}; conversions = 0
        try: td = json.loads(state.traderData) if state.traderData else {}
        except: td = {}
        for p in state.order_depths:
            if p == "ASH_COATED_OSMIUM": orders[p], td = self._aco(state, td)
            elif p == "INTARIAN_PEPPER_ROOT": orders[p], td = self._ipr(state, td)
        tdo = json.dumps(td); logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ═══════════════════════════════════════════════════════════════
    #  ACO — Near-Zero Market Maker
    # ═══════════════════════════════════════════════════════════════
    def _aco(self, state, td):
        product = "ASH_COATED_OSMIUM"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT[product]; result = []
        ANCHOR = self.ACO_ANCHOR; ts = state.timestamp

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        ob_mid = ((best_bid + best_ask) / 2.0) if best_bid is not None and best_ask is not None else None

        # Emergency EMA drift
        ema = td.get('aco_ema', ANCHOR)
        if ob_mid is not None: ema = 0.1 * ob_mid + 0.9 * ema
        td['aco_ema'] = ema
        dc = td.get('aco_drift', 0)
        if abs(ema - ANCHOR) > 50: dc += 1
        else: dc = 0
        td['aco_drift'] = dc
        FAIR = round(ema) if dc >= 100 else ANCHOR

        # Adaptive offset
        prev_pos = td.get('aco_prev_pos', 0)
        if pos > prev_pos: td['aco_lbf'] = ts
        elif pos < prev_pos: td['aco_lsf'] = ts
        td['aco_prev_pos'] = pos
        lbf = td.get('aco_lbf', ts); lsf = td.get('aco_lsf', ts)
        buy_offset = max(self.ACO_OFFSET_MIN, self.ACO_OFFSET_MAX - (ts - lbf) // self.ACO_DECAY_TICKS)
        sell_offset = max(self.ACO_OFFSET_MIN, self.ACO_OFFSET_MAX - (ts - lsf) // self.ACO_DECAY_TICKS)

        # Spread
        spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else 16

        # Taking thresholds: edge=3 baseline
        take_buy_threshold = FAIR - self.ACO_MIN_TAKE_EDGE   # 9997
        take_sell_threshold = FAIR + self.ACO_MIN_TAKE_EDGE  # 10003

        # Reversion: if position extreme, widen thresholds to recover
        if pos <= -self.ACO_SOFT_LIMIT:
            take_buy_threshold = FAIR + self.ACO_REVERT_SLACK   # 10008
        if pos >= self.ACO_SOFT_LIMIT:
            take_sell_threshold = FAIR - self.ACO_REVERT_SLACK  # 9992

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # Taker orders
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price < take_buy_threshold and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol)); buy_capacity -= vol
            else: break

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price >= take_sell_threshold and sell_capacity > 0:
                vol = min(od.buy_orders[bid_price], sell_capacity)
                result.append(Order(product, bid_price, -vol)); sell_capacity -= vol
            else: break

        # Passive quotes with tight inventory gate
        penny_buy = FAIR - buy_offset
        penny_sell = FAIR + sell_offset
        if best_bid is not None and best_bid + 1 <= FAIR: penny_buy = best_bid + 1
        if best_ask is not None and best_ask - 1 >= FAIR: penny_sell = best_ask - 1

        passive_buy_ok = pos < self.ACO_SOFT_LIMIT    # suppress buys when pos ≥ 25
        passive_sell_ok = pos > -self.ACO_SOFT_LIMIT   # suppress sells when pos ≤ -25

        if buy_capacity > 0 and passive_buy_ok:
            t1 = buy_capacity // 2; t2 = buy_capacity - t1
            result.append(Order(product, penny_buy, t1))
            result.append(Order(product, penny_buy - 1, t2))
        if sell_capacity > 0 and passive_sell_ok:
            t1 = sell_capacity // 2; t2 = sell_capacity - t1
            result.append(Order(product, penny_sell, -t1))
            result.append(Order(product, penny_sell + 1, -t2))

        return result, td

    # ═══════════════════════════════════════════════════════════════
    #  IPR — Optimized Cycling
    # ═══════════════════════════════════════════════════════════════
    def _ipr(self, state, td):
        product = "INTARIAN_PEPPER_ROOT"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT[product]; base = self.IPR_BASE_POS; ts = state.timestamp
        result = []; buy_capacity = limit - pos; sell_capacity = limit + pos
        best_ask_ipr = min(od.sell_orders.keys()) if od.sell_orders else None
        best_bid_ipr = max(od.buy_orders.keys()) if od.buy_orders else None
        ob_mid = (best_ask_ipr + best_bid_ipr) / 2.0 if best_ask_ipr and best_bid_ipr else None

        last_ts = td.get('ipr_last_ts', -1)
        if ts < last_ts or ts == 0:
            td['ipr_n']=0; td['ipr_sx']=0.0; td['ipr_sy']=0.0; td['ipr_sxy']=0.0
            td['ipr_sx2']=0.0; td['ipr_sy2']=0.0; td['ipr_phase']='ramp'
        td['ipr_last_ts'] = ts
        if ob_mid is not None:
            n=td.get('ipr_n',0)+1; sx=td.get('ipr_sx',0.0)+ts; sy=td.get('ipr_sy',0.0)+ob_mid
            sxy=td.get('ipr_sxy',0.0)+ts*ob_mid; sx2=td.get('ipr_sx2',0.0)+ts*ts
            sy2=td.get('ipr_sy2',0.0)+ob_mid*ob_mid
            td['ipr_n']=n; td['ipr_sx']=sx; td['ipr_sy']=sy
            td['ipr_sxy']=sxy; td['ipr_sx2']=sx2; td['ipr_sy2']=sy2
        else:
            n=td.get('ipr_n',0); sx=td.get('ipr_sx',0.0); sy=td.get('ipr_sy',0.0)
            sxy=td.get('ipr_sxy',0.0); sx2=td.get('ipr_sx2',0.0); sy2=td.get('ipr_sy2',0.0)

        slope=None; intercept=None; r_squared=0.0; resid_std=float('inf')
        denom = n*sx2 - sx*sx
        if n >= self.IPR_MIN_OBS and denom > 0:
            slope = (n*sxy - sx*sy) / denom; intercept = (sy - slope*sx) / n
            ss_tot = sy2 - n*(sy/n)**2
            ss_res = sy2 - 2*slope*sxy - 2*intercept*sy + slope*slope*sx2 + 2*slope*intercept*sx + n*intercept*intercept
            if ss_tot > 0:
                r_squared = max(0.0, 1.0 - ss_res/ss_tot)
                resid_std = math.sqrt(max(0.0, ss_res/n))
        model_fair = None
        if slope is not None and intercept is not None: model_fair = intercept + slope * ts
        if model_fair is not None and ob_mid is not None:
            w = min(r_squared, 0.95); fair = w * model_fair + (1-w) * ob_mid
        elif model_fair is not None: fair = model_fair
        elif ob_mid is not None: fair = ob_mid
        else: return result, td

        circuit_tripped = False
        if model_fair and ob_mid and resid_std < float('inf') and resid_std > 0:
            if abs(ob_mid - model_fair) > self.IPR_CIRCUIT_BREAKER_SIGMA * resid_std:
                circuit_tripped = True

        ols_confident = (slope is not None and r_squared >= self.IPR_HIGH_CONFIDENCE_R2)
        phase = td.get('ipr_phase', 'ramp')
        if phase == 'ramp' and pos >= base: phase = 'cycle'; td['ipr_phase'] = 'cycle'
        if phase == 'cycle' and pos <= base:
            ids = td.get('ipr_idle_since', ts)
            if ts == ids: td['ipr_idle_since'] = ts
            elif ts - ids >= self.IPR_SPRINT_CYCLE_TICKS:
                phase = 'sprint'; td['ipr_phase'] = 'sprint'
        elif phase == 'cycle' and pos > base: td['ipr_idle_since'] = ts
        if ols_confident and slope is not None and slope <= 0: phase = 'flatten'

        if ob_mid is not None and pos > 10:
            rh = td.get('ipr_recent_high', ob_mid)
            if ob_mid > rh: rh = ob_mid
            td['ipr_recent_high'] = rh
            if rh - ob_mid > 30: phase = 'flatten'

        # ── RAMP (with volume cap) ───────────────────────────────
        if phase == 'ramp':
            ramp_budget = self.IPR_RAMP_MAX_PER_TICK  # cap per tick
            for ap in sorted(od.sell_orders.keys()):
                if ap < fair and buy_capacity > 0 and pos < base and ramp_budget > 0:
                    v = min(-od.sell_orders[ap], buy_capacity, base - pos, ramp_budget)
                    if v > 0: result.append(Order(product, ap, v)); buy_capacity -= v; pos += v; ramp_budget -= v
                else: break
            if buy_capacity > 0 and pos < base and od.sell_orders and not circuit_tripped and ramp_budget > 0:
                ba = min(od.sell_orders.keys()); want = min(base - pos, buy_capacity, ramp_budget)
                tv = min(-od.sell_orders[ba], want)
                if tv > 0: result.append(Order(product, ba, tv)); buy_capacity -= tv; pos += tv
            if buy_capacity > 0 and pos < base:
                bp = math.floor(fair) - self.IPR_PASSIVE_OFFSET
                if best_bid_ipr and best_bid_ipr + 1 < fair: bp = best_bid_ipr + 1
                result.append(Order(product, bp, min(buy_capacity, base - pos)))
            return result, td

        # ── CYCLE ────────────────────────────────────────────────
        if phase == 'cycle':
            lbf = td.get('ipr_last_buy_fill', ts); lsf = td.get('ipr_last_sell_fill', ts)
            bi = ts - lbf; si = ts - lsf
            bd = bi // self.IPR_OFFSET_DECAY_TICKS; sd = si // self.IPR_OFFSET_DECAY_TICKS
            cbo = max(self.IPR_BUY_OFFSET_MIN, self.IPR_BUY_OFFSET_MAX - bd)
            cso = max(self.IPR_SELL_OFFSET_MIN, self.IPR_SELL_OFFSET_MAX - sd)
            mb = math.floor(fair) - cbo; ms = math.ceil(fair) + cso
            if best_ask_ipr and best_ask_ipr - 1 > fair: ms = min(ms, best_ask_ipr - 1)
            if best_bid_ipr and best_bid_ipr + 1 < fair: mb = max(mb, best_bid_ipr + 1)
            for ap in sorted(od.sell_orders.keys()):
                if ap < fair and buy_capacity > 0 and pos < limit:
                    v = min(-od.sell_orders[ap], buy_capacity, limit - pos)
                    if v > 0: result.append(Order(product, ap, v)); buy_capacity -= v; pos += v
                else: break
            for bp in sorted(od.buy_orders.keys(), reverse=True):
                if bp > fair and sell_capacity > 0 and pos > base:
                    v = min(od.buy_orders[bp], sell_capacity, pos - base)
                    if v > 0: result.append(Order(product, bp, -v)); sell_capacity -= v; pos -= v
                else: break
            if product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.buyer == "SUBMISSION": td['ipr_last_buy_fill'] = ts
                    elif trade.seller == "SUBMISSION": td['ipr_last_sell_fill'] = ts
            if pos < limit and buy_capacity > 0:
                result.append(Order(product, mb, min(buy_capacity, limit - pos)))
            if pos > base and sell_capacity > 0:
                result.append(Order(product, ms, -(pos - base)))
            return result, td

        # ── FLATTEN ──────────────────────────────────────────────
        if phase == 'flatten':
            if pos > 0 and sell_capacity > 0:
                for bp in sorted(od.buy_orders.keys(), reverse=True):
                    if sell_capacity > 0 and pos > 0:
                        v = min(od.buy_orders[bp], sell_capacity, pos)
                        if v > 0: result.append(Order(product, bp, -v)); sell_capacity -= v; pos -= v
                    else: break
                if pos > 0 and sell_capacity > 0:
                    sp = math.ceil(fair) + 1
                    if best_ask_ipr and best_ask_ipr - 1 > fair: sp = min(sp, best_ask_ipr - 1)
                    result.append(Order(product, sp, -min(pos, sell_capacity)))
            return result, td

        # ── SPRINT ───────────────────────────────────────────────
        if phase == 'sprint':
            for ap in sorted(od.sell_orders.keys()):
                if ap < fair and buy_capacity > 0:
                    v = min(-od.sell_orders[ap], buy_capacity)
                    result.append(Order(product, ap, v)); buy_capacity -= v; pos += v
                else: break
            if buy_capacity > 0 and pos < limit and od.sell_orders:
                ba = min(od.sell_orders.keys()); tv = min(-od.sell_orders[ba], buy_capacity)
                if tv > 0: result.append(Order(product, ba, tv)); buy_capacity -= tv; pos += tv
            if buy_capacity > 0 and pos < limit:
                bp = math.floor(fair) - 1
                if best_bid_ipr and best_bid_ipr + 1 < fair: bp = max(bp, best_bid_ipr + 1)
                result.append(Order(product, bp, buy_capacity))
            return result, td
        return result, td