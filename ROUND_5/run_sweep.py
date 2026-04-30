"""Fast parameter sweep — run as: python3 run_sweep.py"""
import sys, os, json
sys.path.insert(0,'.')
import backtester_r5 as bt
from collections import defaultdict
from datamodel import Listing, Observation, OrderDepth, TradingState, Order
from typing import Dict, List, Optional

# ── Lightweight trader (no Logger overhead) ───────────────────────────────────
class FastTrader:
    LIMIT=10; HALF_POS=5
    FAST_ALPHA=0.03; SLOW_ALPHA=0.003
    SIGNAL_STRONG=60; SIGNAL_WEAK=25
    WARMUP_TICKS=3000; DRAWDOWN_PTS=350; COOLDOWN_TICKS=5000
    PASSIVE_SIZE=2; RAMP_PER_TICK=5
    PEBBLES_SUM=50000; PEBBLES_XL="PEBBLES_XL"
    PEBBLES_NON_XL=["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L"]
    XL_OFFSET=3; XL_TAKE_EDGE=4
    WHITELIST={"PEBBLES_XL","OXYGEN_SHAKE_MORNING_BREATH","ROBOT_MOPPING",
               "GALAXY_SOUNDS_BLACK_HOLES","SLEEP_POD_COTTON","MICROCHIP_CIRCLE"}

    def run(self, state):
        try: td=json.loads(state.traderData) if state.traderData else {}
        except: td={}
        ts=state.timestamp
        mids={}
        for p,od in state.order_depths.items():
            if od.buy_orders and od.sell_orders:
                mids[p]=(max(od.buy_orders)+min(od.sell_orders))/2.0
        prev_ts=td.get("prev_ts",-1)
        if ts<prev_ts or ts==0:
            for p in list(td.keys()):
                if any(p.startswith(x) for x in ["fe_","se_","cbu_","cbp_","cbd_"]): del td[p]
        td["prev_ts"]=ts
        for p in self.WHITELIST:
            if p in mids:
                fe=td.get(f"fe_{p}",mids[p]); se=td.get(f"se_{p}",mids[p])
                td[f"fe_{p}"]=self.FAST_ALPHA*mids[p]+(1-self.FAST_ALPHA)*fe
                td[f"se_{p}"]=self.SLOW_ALPHA*mids[p]+(1-self.SLOW_ALPHA)*se
        orders={}
        for p,od in state.order_depths.items():
            if p not in self.WHITELIST: orders[p]=[]; continue
            pos=state.position.get(p,0)
            orders[p]=self._trade_xl(p,od,pos,mids,td,ts) if p==self.PEBBLES_XL else self._trade_ema(p,od,pos,mids,td,ts)
        return orders,0,json.dumps(td)

    def _mid(self,od): return (max(od.buy_orders)+min(od.sell_orders))/2.0 if od.buy_orders and od.sell_orders else None

    def _trade_xl(self,p,od,pos,mids,td,ts):
        r=[]; bc=self.LIMIT-pos; sc=self.LIMIT+pos
        bb=max(od.buy_orders) if od.buy_orders else None
        ba=min(od.sell_orders) if od.sell_orders else None
        others=[mids.get(x) for x in self.PEBBLES_NON_XL]
        if any(v is None for v in others): return r
        cf=self.PEBBLES_SUM-sum(others); cfr=round(cf); te=self.XL_TAKE_EDGE
        for ap in sorted(od.sell_orders):
            if ap<cf-te and bc>0: v=min(-od.sell_orders[ap],bc); r.append(Order(p,ap,v)); bc-=v
            else: break
        for bp in sorted(od.buy_orders,reverse=True):
            if bp>cf+te and sc>0: v=min(od.buy_orders[bp],sc); r.append(Order(p,bp,-v)); sc-=v
            else: break
        # EMA bias
        et=self._ema_target(p,td,ts)
        mb=cfr-self.XL_OFFSET; ms=cfr+self.XL_OFFSET
        if bb and bb+1<cf: mb=bb+1
        if ba and ba-1>cf: ms=ba-1
        bsoft=self.LIMIT if et>=0 else self.HALF_POS
        ssoft=self.LIMIT if et<=0 else self.HALF_POS
        if bc>0 and pos<bsoft and mb<cf: r.append(Order(p,mb,min(bc,self.PASSIVE_SIZE)))
        if sc>0 and pos>-ssoft and ms>cf: r.append(Order(p,ms,-min(sc,self.PASSIVE_SIZE)))
        return r

    def _trade_ema(self,p,od,pos,mids,td,ts):
        raw=self._ema_target(p,td,ts); tgt=self._cb(p,raw,pos,mids.get(p),td,ts)
        return self._approach(p,od,pos,tgt)

    def _ema_target(self,p,td,ts):
        if ts<self.WARMUP_TICKS: return 0
        fe=td.get(f"fe_{p}"); se=td.get(f"se_{p}")
        if fe is None or se is None: return 0
        s=fe-se
        if s>self.SIGNAL_STRONG: return self.LIMIT
        elif s>self.SIGNAL_WEAK: return self.HALF_POS
        elif s<-self.SIGNAL_STRONG: return -self.LIMIT
        elif s<-self.SIGNAL_WEAK: return -self.HALF_POS
        return 0

    def _cb(self,p,raw,pos,mid,td,ts):
        if mid is None: return raw
        if ts<td.get(f"cbu_{p}",0): return 0
        if raw!=td.get(f"cbd_{p}",0): td[f"cbp_{p}"]=mid
        td[f"cbd_{p}"]=raw
        if raw>0:
            best=td.get(f"cbp_{p}",mid)
            if mid>best: td[f"cbp_{p}"]=mid; best=mid
            if best-mid>self.DRAWDOWN_PTS:
                td[f"cbu_{p}"]=ts+self.COOLDOWN_TICKS; td[f"cbp_{p}"]=mid; td[f"cbd_{p}"]=0; return 0
        elif raw<0:
            best=td.get(f"cbp_{p}",mid)
            if mid<best: td[f"cbp_{p}"]=mid; best=mid
            if mid-best>self.DRAWDOWN_PTS:
                td[f"cbu_{p}"]=ts+self.COOLDOWN_TICKS; td[f"cbp_{p}"]=mid; td[f"cbd_{p}"]=0; return 0
        return raw

    def _approach(self,p,od,pos,tgt):
        if tgt==0 and pos==0: return []
        r=[]; bc=self.LIMIT-pos; sc=self.LIMIT+pos
        bb=max(od.buy_orders) if od.buy_orders else None
        ba=min(od.sell_orders) if od.sell_orders else None
        if pos<tgt:
            need=tgt-pos
            if ba and bc>0:
                v=min(-od.sell_orders[ba],bc,need,self.RAMP_PER_TICK)
                if v>0: r.append(Order(p,ba,v)); bc-=v; need-=v
            if bb and bc>0 and need>0: r.append(Order(p,bb+1,min(bc,self.PASSIVE_SIZE,need)))
        elif pos>tgt:
            need=pos-tgt
            if bb and sc>0:
                v=min(od.buy_orders[bb],sc,need,self.RAMP_PER_TICK)
                if v>0: r.append(Order(p,bb,-v)); sc-=v; need-=v
            if ba and sc>0 and need>0: r.append(Order(p,ba-1,-min(sc,self.PASSIVE_SIZE,need)))
        return r

# ── Runner ─────────────────────────────────────────────────────────────────────
def run_q(trader, prices, mkt):
    pos=defaultdict(int); real=defaultdict(float); td=""; prev=defaultdict(list)
    for (day,ts) in sorted(prices.keys()):
        tp=prices[(day,ts)]; ods={p:bt.build_order_depth(r) for p,r in tp.items()}
        mk=defaultdict(list)
        for t in mkt.get((day,ts),[]): mk[t.symbol].append(t)
        st=TradingState(timestamp=ts,traderData=td,listings={p:Listing(p,p,"X") for p in tp},
            order_depths=ods,own_trades=dict(prev),market_trades=dict(mk),position=dict(pos),observations=Observation())
        try:
            r=trader.run(st); od2,_,ntd=r if len(r)==3 else (*r,""); td=ntd or ""
        except: od2={}
        prev2=defaultdict(list)
        for p2,orders in (od2 or {}).items():
            if p2 in ods and orders:
                for f in bt.match_orders(orders,ods[p2],p2,pos,real,ts): prev2[p2].append(f)
        prev=prev2
    lm={}
    for tp in prices.values():
        for p,r in tp.items():
            mp=r.get("mid_price","").strip()
            if mp: lm[p]=float(mp)
    return {p:real.get(p,0)+pos.get(p,0)*lm.get(p,0) for p in set(list(real)+list(lm))}

def score(pnl):
    tot=sum(v for v in pnl.values() if v!=0)
    return tot,pnl.get("PEBBLES_XL",0),pnl.get("ROBOT_MOPPING",0),pnl.get("SLEEP_POD_COTTON",0),pnl.get("OXYGEN_SHAKE_MORNING_BREATH",0),pnl.get("GALAXY_SOUNDS_BLACK_HOLES",0),pnl.get("MICROCHIP_CIRCLE",0)

if __name__=="__main__":
    prices=bt.load_prices(".",[2,3,4]); mkt=bt.load_market_trades(".",[2,3,4])
    print("Loaded. Sweeping...")
    hdr=f"{'label':<16} {'total':>9} | {'XL':>7} {'ROBOT':>6} {'SLEEP':>6} {'OXY':>6} {'GALAXY':>7} {'MC':>6}"
    print(hdr); print("-"*78); sys.stdout.flush()

    configs=[
        (3000,60,350,5000,2,"baseline"),
        (3000,60,350,5000,5,"xlps5"),
        (3000,60,350,5000,10,"xlps10"),
        (1000,60,350,5000,2,"warm1k"),
        (500, 60,350,5000,2,"warm500"),
        (3000,40,350,5000,2,"sig40"),
        (3000,50,350,5000,2,"sig50"),
        (3000,60,200,5000,2,"dd200"),
        (3000,60,500,5000,2,"dd500"),
        (3000,60,350,2000,2,"cool2k"),
        (3000,60,350,8000,2,"cool8k"),
        (1000,60,350,5000,5,"w1k+xp5"),
        (1000,40,250,3000,5,"combo_A"),
        (1500,50,300,4000,5,"combo_B"),
        (500, 40,200,3000,5,"combo_C"),
    ]
    results=[]
    for warmup,sig,dd,cool,xp,label in configs:
        t=FastTrader(); t.WARMUP_TICKS=warmup; t.SIGNAL_STRONG=sig
        t.DRAWDOWN_PTS=dd; t.COOLDOWN_TICKS=cool; t.PASSIVE_SIZE=xp
        pnl=run_q(t,prices,mkt); s=score(pnl); results.append((s[0],label,s))
        tot,xl,rob,sl,oxy,gal,mc=s
        print(f"{label:<16} {tot:>9,.0f} | {xl:>7,.0f} {rob:>6,.0f} {sl:>6,.0f} {oxy:>6,.0f} {gal:>7,.0f} {mc:>6,.0f}")
        sys.stdout.flush()

    results.sort(reverse=True)
    print("\n--- RANKED ---")
    for tot,label,s in results:
        xl,rob,sl,oxy,gal,mc=s[1],s[2],s[3],s[4],s[5],s[6]
        print(f"{label:<16} {tot:>9,.0f}")
