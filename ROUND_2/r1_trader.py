"""
ACO Bid/Ask-Aware Strategy
============================
Models bid and ask INDEPENDENTLY instead of using mid-price.

Key insights:
- Bid/ask correlation is only 0.054 (nearly independent)
- Next bid change predictable from: prev_bid_chg, prev_ask_chg, spread
- Spread has strong boundary effects (tight→widens, wide→narrows)

Strategy:
1. Predict next bid/ask using linear model of recent changes + spread
2. Post second passive level at predicted penny (when better than penny-1)
3. When spread is tight (≤12): take asks up to FAIR+2 (spread will widen,
   making our buy profitable as ask rises)

ACO adaptive offset (7→3) and EMA drift safety included.
IPR: proven baseline with wider buy offset.
"""

import json, math
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
    def flush(self, state, orders, conversions, trader_data):
        base_length = len(self.to_json([self.compress_state(state, ""),
            self.compress_orders(orders), conversions, "", ""]))
        m = (self.max_log_length - base_length) // 3
        print(self.to_json([self.compress_state(state, self.truncate(state.traderData, m)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, m), self.truncate(self.logs, m)]))
        self.logs = ""
    def compress_state(self, state, td):
        return [state.timestamp, td,
                [[l.symbol,l.product,l.denomination] for l in state.listings.values()],
                {s:[od.buy_orders,od.sell_orders] for s,od in state.order_depths.items()},
                [[t.symbol,t.price,t.quantity,t.buyer,t.seller,t.timestamp] for arr in state.own_trades.values() for t in arr],
                [[t.symbol,t.price,t.quantity,t.buyer,t.seller,t.timestamp] for arr in state.market_trades.values() for t in arr],
                state.position, self.compress_observations(state.observations)]
    def compress_observations(self, obs):
        co = {}
        for p, o in obs.conversionObservations.items():
            co[p] = [o.bidPrice,o.askPrice,o.transportFees,o.exportTariff,o.importTariff,o.sunlight,o.humidity]
        return [obs.plainValueObservations, co]
    def compress_orders(self, orders):
        return [[o.symbol,o.price,o.quantity] for arr in orders.values() for o in arr]
    def to_json(self, value):
        return json.dumps(value, cls=ProsperityEncoder, separators=(",",":"))
    def truncate(self, value, m):
        return value if len(value) <= m else value[:m-3] + "..."

logger = Logger()


class Trader:

    LIMIT = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}

    # ACO bid/ask prediction coefficients (from linear fit on day 0 full data)
    # next_bid_change = a*prev_bid_chg + b*prev_ask_chg + c*spread + d
    ACO_BID_COEF = (-0.271, -0.217, 0.442, -7.156)
    # next_ask_change = a*prev_bid_chg + b*prev_ask_chg + c*spread + d
    ACO_ASK_COEF = (-0.276, -0.217, -0.570, 9.223)

    ACO_ANCHOR = 10000
    ACO_OFFSET_MAX = 7
    ACO_OFFSET_MIN = 3
    ACO_DECAY_TICKS = 5000

    # IPR config
    IPR_BASE_POS = 75; IPR_PASSIVE_OFFSET = 3
    IPR_BUY_OFFSET_MAX = 10; IPR_BUY_OFFSET_MIN = 1
    IPR_SELL_OFFSET_MAX = 8; IPR_SELL_OFFSET_MIN = 3
    IPR_OFFSET_DECAY_TICKS = 5000; IPR_SPRINT_CYCLE_TICKS = 30000
    IPR_MIN_OBS = 5; IPR_HIGH_CONFIDENCE_R2 = 0.80; IPR_CIRCUIT_BREAKER_SIGMA = 3.0

    def run(self, state: TradingState):
        orders = {}; conversions = 0
        try: td = json.loads(state.traderData) if state.traderData else {}
        except: td = {}
        for product in state.order_depths:
            if product == "ASH_COATED_OSMIUM":
                orders[product], td = self._trade_aco(state, td)
            elif product == "INTARIAN_PEPPER_ROOT":
                orders[product], td = self._trade_ipr(state, td)
        tdo = json.dumps(td); logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    def _trade_aco(self, state, td):
        product = "ASH_COATED_OSMIUM"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT[product]
        result = []
        ANCHOR = self.ACO_ANCHOR
        ts = state.timestamp

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        # ── EMA drift safety ─────────────────────────────────────
        ob_mid = None
        if best_bid is not None and best_ask is not None:
            ob_mid = (best_bid + best_ask) / 2.0
        ema = td.get('aco_ema', ANCHOR)
        if ob_mid is not None:
            ema = 0.1 * ob_mid + 0.9 * ema
        td['aco_ema'] = ema
        drift_count = td.get('aco_drift', 0)
        if abs(ema - ANCHOR) > 50: drift_count += 1
        else: drift_count = 0
        td['aco_drift'] = drift_count
        FAIR = round(ema) if drift_count >= 100 else ANCHOR

        # ── Adaptive offset ──────────────────────────────────────
        prev_pos = td.get('aco_prev_pos', 0)
        if pos > prev_pos: td['aco_last_buy_fill'] = ts
        elif pos < prev_pos: td['aco_last_sell_fill'] = ts
        td['aco_prev_pos'] = pos

        last_buy_fill = td.get('aco_last_buy_fill', ts)
        last_sell_fill = td.get('aco_last_sell_fill', ts)
        buy_offset = max(self.ACO_OFFSET_MIN, self.ACO_OFFSET_MAX - (ts - last_buy_fill) // self.ACO_DECAY_TICKS)
        sell_offset = max(self.ACO_OFFSET_MIN, self.ACO_OFFSET_MAX - (ts - last_sell_fill) // self.ACO_DECAY_TICKS)

        # ── Bid/Ask prediction ───────────────────────────────────
        spread = (best_ask - best_bid) if (best_bid and best_ask) else 16
        prev_bid = td.get('aco_prev_bid', best_bid)
        prev_ask = td.get('aco_prev_ask', best_ask)

        bid_chg = (best_bid - prev_bid) if (best_bid and prev_bid) else 0
        ask_chg = (best_ask - prev_ask) if (best_ask and prev_ask) else 0

        if best_bid is not None: td['aco_prev_bid'] = best_bid
        if best_ask is not None: td['aco_prev_ask'] = best_ask

        # Predict next bid and ask
        a = self.ACO_BID_COEF
        pred_bid_chg = a[0]*bid_chg + a[1]*ask_chg + a[2]*spread + a[3]
        b = self.ACO_ASK_COEF
        pred_ask_chg = b[0]*bid_chg + b[1]*ask_chg + b[2]*spread + b[3]

        pred_next_bid = round(best_bid + pred_bid_chg) if best_bid else None
        pred_next_ask = round(best_ask + pred_ask_chg) if best_ask else None

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # ── Take: baseline + spread-conditioned ──────────────────
        # When spread is tight, take asks slightly above fair
        # (spread will widen, making our position profitable)
        take_buy_fair = FAIR
        take_sell_fair = FAIR
        if spread <= 12 and spread > 0:
            take_buy_fair = FAIR + 2  # willing to buy higher when spread is tight
        elif spread >= 20:
            take_sell_fair = FAIR - 2  # willing to sell lower when spread is wide

        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price < take_buy_fair and buy_capacity > 0:
                vol = min(-od.sell_orders[ask_price], buy_capacity)
                result.append(Order(product, ask_price, vol))
                buy_capacity -= vol
            else: break

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price > take_sell_fair and sell_capacity > 0:
                vol = min(od.buy_orders[bid_price], sell_capacity)
                result.append(Order(product, bid_price, -vol))
                sell_capacity -= vol
            else: break

        # ── Passive: penny-inside + predicted level ──────────────
        penny_buy = FAIR - buy_offset
        penny_sell = FAIR + sell_offset
        if best_bid is not None and best_bid + 1 < FAIR:
            penny_buy = best_bid + 1
        if best_ask is not None and best_ask - 1 > FAIR:
            penny_sell = best_ask - 1

        # Second level: use predicted next penny if it's a BETTER price
        # (cheaper buy or richer sell than penny-1/penny+1)
        if pred_next_bid is not None:
            pred_penny_buy = pred_next_bid + 1
            # Use predicted if it's cheaper than penny-1 but still reasonable
            if pred_penny_buy < penny_buy - 1 and pred_penny_buy >= FAIR - 15:
                second_buy = pred_penny_buy
            else:
                second_buy = penny_buy - 1
        else:
            second_buy = penny_buy - 1

        if pred_next_ask is not None:
            pred_penny_sell = pred_next_ask - 1
            if pred_penny_sell > penny_sell + 1 and pred_penny_sell <= FAIR + 15:
                second_sell = pred_penny_sell
            else:
                second_sell = penny_sell + 1
        else:
            second_sell = penny_sell + 1

        if buy_capacity > 0:
            t1 = buy_capacity // 2
            t2 = buy_capacity - t1
            result.append(Order(product, penny_buy, t1))
            result.append(Order(product, second_buy, t2))

        if sell_capacity > 0:
            t1 = sell_capacity // 2
            t2 = sell_capacity - t1
            result.append(Order(product, penny_sell, -t1))
            result.append(Order(product, second_sell, -t2))

        return result, td

    # ═══════════════════════════════════════════════════════════════
    #  IPR — Proven baseline (unchanged from hardened final)
    # ═══════════════════════════════════════════════════════════════
    def _trade_ipr(self, state, td):
        product = "INTARIAN_PEPPER_ROOT"
        od = state.order_depths[product]; pos = state.position.get(product, 0)
        limit = self.LIMIT[product]; base = self.IPR_BASE_POS; ts = state.timestamp
        result = []; buy_capacity = limit - pos; sell_capacity = limit + pos
        best_ask_ipr = min(od.sell_orders.keys()) if od.sell_orders else None
        best_bid_ipr = max(od.buy_orders.keys()) if od.buy_orders else None
        ob_mid = (best_ask_ipr + best_bid_ipr) / 2.0 if best_ask_ipr and best_bid_ipr else None

        last_ts = td.get('ipr_last_ts', -1)
        if ts < last_ts or ts == 0:
            td['ipr_n']=0;td['ipr_sx']=0.0;td['ipr_sy']=0.0;td['ipr_sxy']=0.0;td['ipr_sx2']=0.0;td['ipr_sy2']=0.0;td['ipr_phase']='ramp'
        td['ipr_last_ts'] = ts
        if ob_mid is not None:
            n=td.get('ipr_n',0)+1;sx=td.get('ipr_sx',0.0)+ts;sy=td.get('ipr_sy',0.0)+ob_mid
            sxy=td.get('ipr_sxy',0.0)+ts*ob_mid;sx2=td.get('ipr_sx2',0.0)+ts*ts;sy2=td.get('ipr_sy2',0.0)+ob_mid*ob_mid
            td['ipr_n']=n;td['ipr_sx']=sx;td['ipr_sy']=sy;td['ipr_sxy']=sxy;td['ipr_sx2']=sx2;td['ipr_sy2']=sy2
        else:
            n=td.get('ipr_n',0);sx=td.get('ipr_sx',0.0);sy=td.get('ipr_sy',0.0);sxy=td.get('ipr_sxy',0.0);sx2=td.get('ipr_sx2',0.0);sy2=td.get('ipr_sy2',0.0)
        slope=None;intercept=None;r_squared=0.0;resid_std=float('inf')
        denom=n*sx2-sx*sx
        if n>=self.IPR_MIN_OBS and denom>0:
            slope=(n*sxy-sx*sy)/denom;intercept=(sy-slope*sx)/n
            ss_tot=sy2-n*(sy/n)**2;ss_res=sy2-2*slope*sxy-2*intercept*sy+slope*slope*sx2+2*slope*intercept*sx+n*intercept*intercept
            if ss_tot>0:r_squared=max(0.0,1.0-ss_res/ss_tot);resid_std=math.sqrt(max(0.0,ss_res/n))
        model_fair=None
        if slope is not None and intercept is not None:model_fair=intercept+slope*ts
        if model_fair is not None and ob_mid is not None:
            w=min(r_squared,0.95);fair=w*model_fair+(1-w)*ob_mid
        elif model_fair is not None:fair=model_fair
        elif ob_mid is not None:fair=ob_mid
        else:return result,td
        circuit_tripped=False
        if model_fair and ob_mid and resid_std<float('inf') and resid_std>0:
            if abs(ob_mid-model_fair)>self.IPR_CIRCUIT_BREAKER_SIGMA*resid_std:circuit_tripped=True
        phase=td.get('ipr_phase','ramp')
        ols_confident=(slope is not None and r_squared>=self.IPR_HIGH_CONFIDENCE_R2)
        if phase=='ramp' and pos>=base:phase='cycle';td['ipr_phase']='cycle'
        if phase=='cycle' and pos<=base:
            idle_since=td.get('ipr_idle_since',ts)
            if ts==idle_since:td['ipr_idle_since']=ts
            elif ts-idle_since>=self.IPR_SPRINT_CYCLE_TICKS:phase='sprint';td['ipr_phase']='sprint'
        elif phase=='cycle' and pos>base:td['ipr_idle_since']=ts
        if ols_confident and slope is not None and slope<=0:phase='flatten'

        # Short-window reversal detection
        if ob_mid is not None and pos > 10:
            recent_high = td.get('ipr_recent_high', ob_mid)
            if ob_mid > recent_high: recent_high = ob_mid
            td['ipr_recent_high'] = recent_high
            if recent_high - ob_mid > 30: phase = 'flatten'

        if phase=='ramp':
            for ap in sorted(od.sell_orders.keys()):
                if ap<fair and buy_capacity>0 and pos<base:
                    v=min(-od.sell_orders[ap],buy_capacity,base-pos)
                    if v>0:result.append(Order(product,ap,v));buy_capacity-=v;pos+=v
                else:break
            if buy_capacity>0 and pos<base and od.sell_orders and not circuit_tripped:
                ba=min(od.sell_orders.keys());want=min(base-pos,buy_capacity);tv=min(-od.sell_orders[ba],want)
                if tv>0:result.append(Order(product,ba,tv));buy_capacity-=tv;pos+=tv
            if buy_capacity>0 and pos<base:
                bp=math.floor(fair)-self.IPR_PASSIVE_OFFSET
                if best_bid_ipr and best_bid_ipr+1<fair:bp=best_bid_ipr+1
                result.append(Order(product,bp,min(buy_capacity,base-pos)))
            return result,td
        if phase=='cycle':
            lbf=td.get('ipr_last_buy_fill',ts);lsf=td.get('ipr_last_sell_fill',ts)
            bi=ts-lbf;si=ts-lsf;bd=bi//self.IPR_OFFSET_DECAY_TICKS;sd=si//self.IPR_OFFSET_DECAY_TICKS
            cbo=max(self.IPR_BUY_OFFSET_MIN,self.IPR_BUY_OFFSET_MAX-bd);cso=max(self.IPR_SELL_OFFSET_MIN,self.IPR_SELL_OFFSET_MAX-sd)
            mm_buy=math.floor(fair)-cbo;mm_sell=math.ceil(fair)+cso
            if best_ask_ipr and best_ask_ipr-1>fair:mm_sell=min(mm_sell,best_ask_ipr-1)
            if best_bid_ipr and best_bid_ipr+1<fair:mm_buy=max(mm_buy,best_bid_ipr+1)
            for ap in sorted(od.sell_orders.keys()):
                if ap<fair and buy_capacity>0 and pos<limit:
                    v=min(-od.sell_orders[ap],buy_capacity,limit-pos)
                    if v>0:result.append(Order(product,ap,v));buy_capacity-=v;pos+=v
                else:break
            for bp in sorted(od.buy_orders.keys(),reverse=True):
                if bp>fair and sell_capacity>0 and pos>base:
                    v=min(od.buy_orders[bp],sell_capacity,pos-base)
                    if v>0:result.append(Order(product,bp,-v));sell_capacity-=v;pos-=v
                else:break
            if product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.buyer=="SUBMISSION":td['ipr_last_buy_fill']=ts
                    elif trade.seller=="SUBMISSION":td['ipr_last_sell_fill']=ts
            if pos<limit and buy_capacity>0:result.append(Order(product,mm_buy,min(buy_capacity,limit-pos)))
            if pos>base and sell_capacity>0:result.append(Order(product,mm_sell,-(pos-base)))
            return result,td
        if phase=='flatten':
            if pos>0 and sell_capacity>0:
                for bid_price in sorted(od.buy_orders.keys(),reverse=True):
                    if sell_capacity>0 and pos>0:
                        v=min(od.buy_orders[bid_price],sell_capacity,pos)
                        if v>0:result.append(Order(product,bid_price,-v));sell_capacity-=v;pos-=v
                    else:break
                if pos>0 and sell_capacity>0:
                    sp=math.ceil(fair)+1
                    if best_ask_ipr and best_ask_ipr-1>fair:sp=min(sp,best_ask_ipr-1)
                    result.append(Order(product,sp,-min(pos,sell_capacity)))
            return result,td
        if phase=='sprint':
            for ap in sorted(od.sell_orders.keys()):
                if ap<fair and buy_capacity>0:
                    v=min(-od.sell_orders[ap],buy_capacity);result.append(Order(product,ap,v));buy_capacity-=v;pos+=v
                else:break
            if buy_capacity>0 and pos<limit and od.sell_orders:
                ba=min(od.sell_orders.keys());tv=min(-od.sell_orders[ba],buy_capacity)
                if tv>0:result.append(Order(product,ba,tv));buy_capacity-=tv;pos+=tv
            if buy_capacity>0 and pos<limit:
                bp=math.floor(fair)-1
                if best_bid_ipr and best_bid_ipr+1<fair:bp=max(bp,best_bid_ipr+1)
                result.append(Order(product,bp,buy_capacity))
            return result,td
        return result,td