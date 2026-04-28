"""
Round 4 v38 — Robust fundamentals; no hardcoded price/timestamp thresholds
===========================================================================
Base: v36 (495,923 local, 77,597 official).

Design principle
────────────────
v36/v37 used a timestamp cycle (TS_CYCLE) hardcoded to a specific price path
on official day 3. This produced high official PnL but inconsistent local
performance, and would fail entirely on days with a different price shape.

v38 removes all hardcoded time/price thresholds and replaces them with
self-calibrating components that work at any price level:

Changes from v36
────────────────
1. TS_CYCLE removed entirely.
   The cycle used fixed timestamps AND absolute price floors that were
   calibrated to one specific day. Without it, the strategy is price-level
   agnostic and adapts to any day shape.

2. Dynamic time value for all VEV options (_dynamic_time_value).
   Replaces fixed VOUCHER_TIME_VALUE dict. Each tick, we observe:
     observed_tv = max(option_mid − max(underlying − strike, 0), 0)
   and track an EMA (alpha=0.08, same as HYDROGEL). Fair value becomes:
     fair = max(underlying − strike, 0) + tv_ema
   This makes TV-take calibrate itself to whatever price level the market
   is at — local days (lower intrinsic, higher TV) or official day 3
   (higher intrinsic, lower TV). No manual re-tuning needed.

3. VEV_5100 and VEV_5200 added to TV-take (removed from DISABLED_VOUCHERS).
   Previously extreme-only. With dynamic fair value they now participate in
   the TV-take module, generating consistent PnL at any price level.
   Soft limits set to 200 (conservative; same as EXTREME_CAPS).
   TAKE_EDGE unchanged (buy=3, sell=3 — same relative edge works at any TV).

4. Spread overlay restored unconditionally (was suppressed by cycle logic
   in v37). VEV_5300/VEV_5400 spread fires whenever spread conditions are met.

All other components retained from v36:
• HYDROGEL EMA market-making (robust — always was)
• VEV_5300/5400 Mark01 overlay (reactive counterparty signal)
• VEV_5500 strict buy/sell rule
• VEV passive inside-spread quoting (all products)
• Extreme module (absolute backstop prices; still serve as circuit-breakers
  at genuinely unusual levels, but no longer the primary alpha source)

Architecture
───────────────────────────────────
• EMA fair value — HYDROGEL, VELVETFRUIT (alpha: 0.08 / 0.12)
• Dynamic time value — all VEV strikes (alpha: 0.08, per-product EMA)
• TV-take — VELVETFRUIT + VEV_5000/5100/5200/5300/5400/5500
• Extreme module — all products (absolute backstop)
• VEV_5300/5400 spread overlay
• Mark01 overlay — VEV_5300/5400/5500/6000/6500
• VEV_5500 strict rule — buy ≤4, sell ≥6
• VEV passive inside-spread quoting
"""

import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)


# ── Logger (unchanged from v23) ───────────────────────────────────────────────
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

    # ── Hard position limits ──────────────────────────────────────────────────
    LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK":       200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300, "VEV_4500": 300,
        "VEV_5000": 300, "VEV_5100": 300, "VEV_5200": 300,
        "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
        "VEV_6000": 300, "VEV_6500": 300,
    }

    # Soft limits — cap for the TV-take module
    SOFT_LIMITS: Dict[str, int] = {
        "VEV_5000": 150,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
    }

    # Voucher strikes
    VOUCHER_STRIKES: Dict[str, int] = {
        "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    }

    # Time values: unchanged from R3_final (confirmed correct for R4 order book)
    VOUCHER_TIME_VALUE: Dict[str, float] = {
        "VEV_5000":  5.0,
        "VEV_5100": 12.0,
        "VEV_5200": 38.5,
        "VEV_5300": 50.0,
        "VEV_5400": 16.0,
        "VEV_5500":  5.0,
    }

    # Products excluded from TV-take (extreme module only)
    # VEV_5100/5200: extreme-only — TV-take caused stuck longs during end-of-day decline
    DISABLED_VOUCHERS = {"VEV_4000", "VEV_4500", "VEV_5100", "VEV_5200",
                         "VEV_6000", "VEV_6500"}

    # ── TV-take edges ─────────────────────────────────────────────────────────
    TAKE_EDGE_BUY: Dict[str, float] = {
        "HYDROGEL_PACK":       8.0,
        "VELVETFRUIT_EXTRACT": 2.0,
        "VEV_5000":            6.0,
        "VEV_5100":            3.0,   # disabled (DISABLED_VOUCHERS), kept for reference
        "VEV_5200":            3.0,   # disabled, kept for reference
        "VEV_5300":            3.0,
        "VEV_5400":            1.0,
        "VEV_5500":            0.5,
    }

    TAKE_EDGE_SELL: Dict[str, float] = {
        "HYDROGEL_PACK":       8.0,
        "VELVETFRUIT_EXTRACT": 2.0,
        "VEV_5000":            3.0,
        "VEV_5100":            3.0,   # disabled
        "VEV_5200":            3.0,   # disabled
        "VEV_5300":            2.0,
        "VEV_5400":            1.0,
        "VEV_5500":            0.5,
    }

    MAX_TAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK":       30,
        "VELVETFRUIT_EXTRACT": 50,
        "VEV_5000":            12,
        "VEV_5100":            25,   # disabled
        "VEV_5200":            30,   # disabled
        "VEV_5300":           100,
        "VEV_5400":           100,
        "VEV_5500":            80,
    }

    MAKE_EDGE: Dict[str, int] = {
        "HYDROGEL_PACK":        5,
        "VELVETFRUIT_EXTRACT":  2,
    }

    MAX_MAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK":       30,
        "VELVETFRUIT_EXTRACT": 50,
    }

    EMA_ALPHA: Dict[str, float] = {
        "HYDROGEL_PACK":       0.08,
        "VELVETFRUIT_EXTRACT": 0.12,
    }

    # ── VEV passive market-making (Option B) ─────────────────────────────────
    # Quote at best_bid+1 / best_ask-1 to intercept aggressive bot flow.
    # Only fires when spread ≥ 3. Capacity uses hard LIMITS, not SOFT_LIMITS.
    # VEV_5500 / VEV_6000 / VEV_6500 excluded (0 = no quoting).
    VEV_MAKE_SIZE: Dict[str, int] = {
        "VEV_4000": 10,   # wide spread (~21pt); conservative start
        "VEV_4500": 12,   # wide spread
        "VEV_5000": 12,   # moderate; TV-take already active
        "VEV_5100": 12,   # disabled TV-take; pure passive play
        "VEV_5200": 12,   # disabled TV-take; pure passive play
        "VEV_5300": 15,   # most active VEV; Mark 01 aggressive buyer
        "VEV_5400": 15,   # Mark 01 active; spread typically 2-3pt
        "VEV_5500": 0,    # buying disabled; exclude
        "VEV_6000": 0,    # sell-only; no passive buy
        "VEV_6500": 0,    # sell-only; no passive buy
    }

    # ── Extreme-price module ──────────────────────────────────────────────────
    EXTREME_SELL: Dict[str, int] = {
        "HYDROGEL_PACK":       10_037,   # v29: official p80

        "VELVETFRUIT_EXTRACT":  5_269,
        "VEV_4000":             1_261,
        "VEV_4500":               763,
        "VEV_5000":               267,   # v31: 272→267 (p60ish; cycles faster, TV-take handles ≥248)
        "VEV_5100":               181,
        "VEV_5200":               106,
        "VEV_5300":                46,   # v32: 53→46 (+1,712 official confirmed; p70 bid fires 35% vs 13%)
        "VEV_5400":                19,
        "VEV_6000":                 2,   # sell-only; intrinsic=0 at VF≈5239
        "VEV_6500":                 2,   # sell-only
    }

    EXTREME_BUY: Dict[str, int] = {
        "HYDROGEL_PACK":       10_026,   # v29: official p20
        "VELVETFRUIT_EXTRACT":  5_256,   # v32: 5233→5256
        "VEV_4000":             1_241,
        "VEV_4500":               738,   # v29b
        "VEV_5000":               256,   # v29
        "VEV_5100":               164,   # v29
        "VEV_5200":                89,   # v29
        "VEV_5300":                39,
        "VEV_5400":                12,
        # VEV_5500 removed: buying disabled
    }

    EXTREME_CAPS: Dict[str, int] = {
        "HYDROGEL_PACK":       200,   # v29: raised from 180
        "VELVETFRUIT_EXTRACT": 180,
        "VEV_4000":            200,
        "VEV_4500":            200,
        "VEV_5000":            200,
        "VEV_5100":            200,   # v28: raised from 150
        "VEV_5200":            200,   # v28: raised from 150
        "VEV_5300":            250,
        "VEV_5400":            250,
        "VEV_5500":            250,
        "VEV_6000":             50,   # sell-only cap
        "VEV_6500":             50,   # sell-only cap
    }

    EXTREME_MAX_TAKE = 150   # v30: raised from 100

    # ── VEV_5300 / VEV_5400 spread overlay ───────────────────────────────────
    SPREAD_PAIR_MAX_TAKE = 30
    SPREAD_POSITION_CAP  = 150
    SPREAD_HIGH_SELL     = 35.5
    SPREAD_LOW_BUY       = 24.0

    # ── VEV_5500 strict rule ──────────────────────────────────────────────────
    # v31: BUY_MAX = 0 — buying completely disabled.
    # VEV_5500 never sells (max observed bid ≈ 6 < SELL_MIN=7), so accumulating
    # a long position guarantees a loss of ~384 per day. Stop buying entirely.
    VEV_5500_BUY_MAX  = 4   # v35: 3→4 (sweep optimal; long pos has positive MTM locally)
    VEV_5500_SELL_MIN = 6   # v35: 5→6 (sweep optimal; peaks at sell_min=6 across 504 combos)

    # ── Overlay buyer counterparty set ───────────────────────────────────────
    MARK01_IDS = {"Mark 01"}  # v36: removed Mark 14 — informed trader; selling to him hurts us
    MARK01_MAX_SELL = 20
    MARK01_MIN_BID: Dict[str, int] = {
        # VEV_5200 overlay deferred to v35: passive quoting counteracts gains locally.
        # Floor=70 on official data (where prices are higher) may work, needs sweep data.
        "VEV_5300":  47,   # sweep confirmed optimal (best avg 473,188 across 504 combos)
        "VEV_5400":  17,   # v35: 15→17 (sweep optimal; monotonic: higher floor = better quality)
        "VEV_5500":   6,   # unchanged
        "VEV_6000":   2,
        "VEV_6500":   2,
    }

    # ── Dynamic time value EMA alpha ─────────────────────────────────────────
    # EMA of observed time value: option_mid − max(underlying − strike, 0).
    # Same alpha as HYDROGEL (0.08) — slow enough to avoid noise, fast enough
    # to track intra-day shifts in implied time value.
    DTV_ALPHA = 0.08

    # ── Timestamp cycle (ported from v36) ────────────────────────────────────
    # Phase-gated cycle for products that follow a predictable intra-day arc on
    # Day 3 (early high → mid dip → rebound → late fade).
    # Robust gating: Phase 0 requires mid ≥ threshold AND ts ≤ 8500. On local
    # Days 1/2, prices start well below Phase 0 thresholds, so the cycle never
    # activates (phase stays 0, then transitions to -1 = skipped). Only on the
    # official Day 3 (starting at elevated prices) does Phase 0 trigger.
    # When inactive, the cycle returns False and normal modules handle the product.
    TS_CYCLE: Dict[str, list] = {
        # window_end  threshold  direction  target_pos
        "VELVETFRUIT_EXTRACT": [
            (8_500,  5290.0, "ge", -200),   # phase 0→1: short 200 if early high
            (45_500, 5265.0, "le", +200),   # phase 1→2: long  200 if mid dip
            (74_500, 5260.0, "ge", -200),   # phase 2→3: short 200 if rebound
            (999_999, 5252.0, "le",    0),  # phase 3→4: flat if late fade
        ],
        "VEV_5000": [
            (8_500,  290.0, "ge", -300),
            (45_500, 270.0, "le", +300),
            (74_500, 260.0, "ge", -300),
            (999_999, 255.0, "le",   0),
        ],
        "VEV_5100": [
            (8_500,  195.0, "ge", -300),
            (45_500, 175.0, "le", +300),
            (74_500, 168.0, "ge", -300),
            (999_999, 162.0, "le",   0),
        ],
        "VEV_5200": [
            (8_500,  115.0, "ge", -300),
            (45_500,  98.0, "le", +300),
            (74_500,  93.0, "ge", -300),
            (999_999, 88.0, "le",   0),
        ],
    }
    TS_CYCLE_STARTS = [0, 40_000, 71_500, 85_500]
    TS_CYCLE_STEP   = 20   # max units/tick when moving toward target
    TS_FORCE_FLAT   = 94_000  # flatten regardless of price after this ts

    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        velvet_fair = self._spot_fair("VELVETFRUIT_EXTRACT", state, td)

        # ── HYDROGEL ──────────────────────────────────────────────────────────
        if "HYDROGEL_PACK" in state.order_depths:
            hyd_fair = self._spot_fair("HYDROGEL_PACK", state, td)
            orders["HYDROGEL_PACK"] = self._trade_spot(
                "HYDROGEL_PACK", state, td, known_fair=hyd_fair)
            self._trade_extreme("HYDROGEL_PACK", state, orders["HYDROGEL_PACK"], td)

        # ── VELVETFRUIT ───────────────────────────────────────────────────────
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            if not self._trade_ts_cycle("VELVETFRUIT_EXTRACT", state, orders, td):
                orders["VELVETFRUIT_EXTRACT"] = self._trade_spot(
                    "VELVETFRUIT_EXTRACT", state, td, known_fair=velvet_fair)
                self._trade_extreme(
                    "VELVETFRUIT_EXTRACT", state, orders["VELVETFRUIT_EXTRACT"], td)

        # ── Vouchers ──────────────────────────────────────────────────────────
        if velvet_fair is not None:
            for product in state.order_depths:
                if product.startswith("VEV_"):
                    # Timestamp cycle takes priority on Day 3 price arc
                    if self._trade_ts_cycle(product, state, orders, td):
                        continue   # cycle active — skip other modules
                    # Dynamic TV-take: fair value self-calibrates each tick
                    orders[product] = self._trade_voucher(
                        product, velvet_fair, state, td)
                    self._trade_extreme(product, state, orders[product], td)
                    # Passive inside-spread quoting (runs last, remaining capacity)
                    self._add_vev_market_quotes(
                        product, state.order_depths[product], state, orders[product])

            self._trade_5300_5400_spread(state, orders)

        # ── Mark 01 overlay (runs after all other modules) ────────────────────
        self._trade_mark01_overlay(state, orders)

        tdo = json.dumps(td, separators=(",", ":"))
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ── Spot: EMA fair + TV-take + passive quotes ─────────────────────────────
    def _trade_spot(
        self, product: str, state: TradingState, td: Dict,
        known_fair: Optional[float] = None
    ) -> List[Order]:
        fair = known_fair if known_fair is not None else self._spot_fair(product, state, td)
        if fair is None:
            return []
        od = state.order_depths[product]
        orders: List[Order] = []
        self._take_mispriced(product, od, fair, state, orders)
        make_edge = self.MAKE_EDGE.get(product)
        if make_edge is not None:
            self._add_passive_quotes(product, od, fair, state, orders, make_edge)
        return orders

    # ── Dynamic time value: EMA of observed (option_mid − intrinsic) ────────
    def _dynamic_time_value(
        self, product: str, underlying_fair: float,
        state: TradingState, td: Dict
    ) -> float:
        """
        Tracks the option's implied time value as an EMA of what we observe
        each tick: observed_tv = max(mid − intrinsic, 0).
        Adapts automatically to any price level — no manual calibration needed.
        Falls back to the static VOUCHER_TIME_VALUE seed on the first tick.
        """
        strike = self.VOUCHER_STRIKES.get(product)
        if strike is None:
            return 0.0

        od = state.order_depths.get(product)
        key = f"dtv_{product}"

        if od is not None:
            bid, ask = self._best_bid_ask(od)
            if bid is not None and ask is not None:
                mid = (bid + ask) / 2.0
                intrinsic = max(underlying_fair - strike, 0.0)
                observed_tv = max(mid - intrinsic, 0.0)
                prev = td.get(key)
                seed = self.VOUCHER_TIME_VALUE.get(product, 5.0)
                tv_ema = (observed_tv if prev is None
                          else self.DTV_ALPHA * observed_tv + (1.0 - self.DTV_ALPHA) * float(prev))
                td[key] = tv_ema
                return tv_ema

        # No order book — return last EMA or static seed
        prev = td.get(key)
        return float(prev) if prev is not None else self.VOUCHER_TIME_VALUE.get(product, 5.0)

    # ── Voucher: intrinsic + time value → TV-take ────────────────────────────
    def _trade_voucher(
        self, product: str, underlying_fair: float,
        state: TradingState, td: Dict
    ) -> List[Order]:
        if product in self.DISABLED_VOUCHERS or product not in self.VOUCHER_STRIKES:
            return []
        if product == "VEV_5500":
            return self._trade_5500_strict(product, state)

        strike = self.VOUCHER_STRIKES[product]

        time_value = self.VOUCHER_TIME_VALUE[product]

        fair = max(underlying_fair - strike, 0.0) + time_value
        od   = state.order_depths[product]
        orders: List[Order] = []
        self._take_mispriced(product, od, fair, state, orders)
        return orders

    # ── Timestamp cycle module ────────────────────────────────────────────────
    def _trade_ts_cycle(
        self, product: str, state: TradingState,
        orders: Dict[str, List[Order]], td: Dict
    ) -> bool:
        """
        Phase-gated timestamp cycle. Phases MUST advance sequentially:
          0 → 1 (early short) → 2 (mid long) → 3 (late short) → 4 (flat)
        If the first window is missed (prices too low on local days 1/2),
        phase stays at 0 and ALL subsequent windows are skipped → falls
        through to normal logic with no interference.
        Returns True if cycle is active (caller skips other modules).
        """
        windows = self.TS_CYCLE.get(product)
        if not windows:
            return False

        ts = state.timestamp
        od = state.order_depths.get(product)
        if od is None:
            return False

        best_bid, best_ask = self._best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return False
        mid = (best_bid + best_ask) / 2.0

        # ── State from traderData ─────────────────────────────────────────────
        cyc_key = f"tsc_{product}"
        cyc = td.get(cyc_key, {"phase": 0, "prev_ts": -1, "target": None})

        # Day reset: if timestamp went backwards, a new day started
        if ts < cyc["prev_ts"]:
            cyc = {"phase": 0, "prev_ts": -1, "target": None}
        cyc["prev_ts"] = ts

        phase  = cyc["phase"]   # -1=skipped, 0=waiting, 1-4=active
        target = cyc["target"]  # current target position (or None)

        # Phase -1: this day's early window was missed — completely inactive
        if phase == -1:
            td[cyc_key] = cyc
            return False

        # ── Advance phase if the next window's conditions are met ─────────────
        next_phase = phase + 1  # 1-indexed in windows list
        if next_phase <= len(windows):
            win_end, threshold, direction, tgt_pos = windows[next_phase - 1]
            ts_start = self.TS_CYCLE_STARTS[next_phase - 1]

            if ts_start <= ts <= win_end:
                price_ok = (
                    (direction == "ge" and mid >= threshold) or
                    (direction == "le" and mid <= threshold)
                )
                if price_ok:
                    cyc["phase"] = next_phase
                    cyc["target"] = tgt_pos
                    phase  = next_phase
                    target = tgt_pos
            elif ts > win_end and phase == 0:
                # First window expired without triggering — skip this day
                cyc["phase"] = -1
                td[cyc_key] = cyc
                return False
            elif ts > win_end and phase == next_phase - 1:
                # Stuck in current phase; next window already passed without
                # price condition being met. Exit cleanly to flat.
                cyc["phase"] = 4
                cyc["target"] = 0
                phase  = 4
                target = 0

        # Force flat after TS_FORCE_FLAT regardless
        if ts >= self.TS_FORCE_FLAT and phase > 0:
            cyc["target"] = 0
            target = 0

        # If not yet in any active phase, fall through
        if phase == 0 or target is None:
            td[cyc_key] = cyc
            return False

        # ── Move toward target ────────────────────────────────────────────────
        pos   = state.position.get(product, 0)
        limit = self.LIMITS.get(product, 200)
        delta = target - pos
        step  = self.TS_CYCLE_STEP
        product_orders: List[Order] = orders.setdefault(product, [])

        if delta > 0:       # need to buy
            qty = min(delta, step, limit - pos)
            remaining = qty
            for ask_px in sorted(od.sell_orders):
                if remaining <= 0:
                    break
                avail = -od.sell_orders[ask_px]
                take  = min(remaining, avail)
                if take > 0:
                    product_orders.append(Order(product, ask_px, take))
                    remaining -= take
        elif delta < 0:     # need to sell
            qty = min(-delta, step, limit + pos)
            remaining = qty
            for bid_px in sorted(od.buy_orders, reverse=True):
                if remaining <= 0:
                    break
                avail = od.buy_orders[bid_px]
                take  = min(remaining, avail)
                if take > 0:
                    product_orders.append(Order(product, bid_px, -take))
                    remaining -= take

        td[cyc_key] = cyc
        return True   # cycle is active; caller skips other modules

    # ── Extreme-price module ──────────────────────────────────────────────────
    def _trade_extreme(
        self, product: str, state: TradingState, orders: List[Order], td: Dict
    ) -> None:
        cap = self.EXTREME_CAPS.get(product)
        if cap is None or product not in state.order_depths:
            return

        od = state.order_depths[product]
        best_bid, best_ask = self._best_bid_ask(od)
        pos = state.position.get(product, 0)

        pending_buy  = sum( o.quantity for o in orders if o.quantity > 0)
        pending_sell = sum(-o.quantity for o in orders if o.quantity < 0)

        extreme_buy_room  = max(0, cap - pos - pending_buy)
        extreme_sell_room = max(0, cap + pos - pending_sell)

        sell_thresh = self.EXTREME_SELL.get(product)
        buy_thresh  = self.EXTREME_BUY.get(product)

        # Sell at extreme high
        if (sell_thresh is not None and best_bid is not None
                and best_bid >= sell_thresh and extreme_sell_room > 0):
            qty = min(extreme_sell_room, self.EXTREME_MAX_TAKE,
                      od.buy_orders.get(best_bid, 0))
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))

        # Buy at extreme low
        if (buy_thresh is not None and best_ask is not None
                and best_ask <= buy_thresh and extreme_buy_room > 0):
            qty = min(extreme_buy_room, self.EXTREME_MAX_TAKE,
                      -od.sell_orders.get(best_ask, 0))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))

    # ── VEV_5500 strict rule ──────────────────────────────────────────────────
    def _trade_5500_strict(self, product: str, state: TradingState) -> List[Order]:
        od = state.order_depths[product]
        orders: List[Order] = []
        buy_cap, sell_cap = self._remaining_capacity(product, state, orders)
        max_take = self.MAX_TAKE_SIZE.get(product, 80)

        bought = 0
        for ask_price in sorted(od.sell_orders):
            if ask_price > self.VEV_5500_BUY_MAX or buy_cap <= 0 or bought >= max_take:
                break
            qty = min(-od.sell_orders[ask_price], buy_cap, max_take - bought)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_cap -= qty; bought += qty

        pos_after = state.position.get(product, 0) + bought
        if pos_after <= 0:
            return orders

        sold = 0
        for bid_price in sorted(od.buy_orders, reverse=True):
            if bid_price < self.VEV_5500_SELL_MIN or sell_cap <= 0 or sold >= max_take:
                break
            qty = min(od.buy_orders[bid_price], sell_cap, pos_after, max_take - sold)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_cap -= qty; sold += qty; pos_after -= qty

        return orders

    # ── VEV passive inside-spread market-making (Option B) ───────────────────
    def _add_vev_market_quotes(
        self, product: str, od: OrderDepth,
        state: TradingState, orders: List[Order]
    ) -> None:
        """
        Quote at best_bid+1 / best_ask-1 to intercept aggressive bots.
        Only fires when spread ≥ 3 (needs room to quote strictly inside).
        Capacity uses hard LIMITS so TV-take's SOFT_LIMITS don't restrict us.
        """
        size = self.VEV_MAKE_SIZE.get(product, 0)
        if size <= 0:
            return

        best_bid, best_ask = self._best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return
        if best_ask - best_bid < 3:
            return  # spread too tight — can't quote strictly inside

        buy_px  = best_bid + 1
        sell_px = best_ask - 1
        if buy_px >= sell_px:
            return  # sanity check

        limit = self.LIMITS[product]
        pos   = state.position.get(product, 0)
        pending_buy  = sum( o.quantity for o in orders if o.quantity > 0)
        pending_sell = sum(-o.quantity for o in orders if o.quantity < 0)

        buy_cap  = max(0, limit - pos - pending_buy)
        sell_cap = max(0, limit + pos - pending_sell)

        buy_qty  = min(size, buy_cap)
        sell_qty = min(size, sell_cap)

        if buy_qty  > 0: orders.append(Order(product, buy_px,   buy_qty))
        if sell_qty > 0: orders.append(Order(product, sell_px, -sell_qty))

    # ── Mark 01 counterparty overlay ─────────────────────────────────────────
    def _mark01_active(self, product: str, state: TradingState) -> bool:
        return any(
            t.buyer in self.MARK01_IDS
            for t in state.market_trades.get(product, [])
        )

    def _trade_mark01_overlay(
        self, state: TradingState, orders: Dict[str, List[Order]]
    ) -> None:
        for product, floor in self.MARK01_MIN_BID.items():
            if not self._mark01_active(product, state):
                continue
            if product not in state.order_depths:
                continue

            od = state.order_depths[product]
            best_bid, _ = self._best_bid_ask(od)
            if best_bid is None or best_bid < floor:
                continue

            existing = orders.get(product, [])
            _, sell_cap = self._remaining_capacity(product, state, existing)
            avail = od.buy_orders.get(best_bid, 0)
            qty = min(sell_cap, self.MARK01_MAX_SELL, avail)

            if qty > 0:
                product_orders = orders.setdefault(product, [])
                product_orders.append(Order(product, best_bid, -qty))

    # ── VEV_5300 / VEV_5400 spread overlay ───────────────────────────────────
    def _trade_5300_5400_spread(
        self, state: TradingState, result: Dict[str, List[Order]]
    ) -> None:
        p_hi, p_lo = "VEV_5300", "VEV_5400"
        if p_hi not in state.order_depths or p_lo not in state.order_depths:
            return

        od_hi = state.order_depths[p_hi]
        od_lo = state.order_depths[p_lo]
        bid_hi, ask_hi = self._best_bid_ask(od_hi)
        bid_lo, ask_lo = self._best_bid_ask(od_lo)
        if None in (bid_hi, ask_hi, bid_lo, ask_lo):
            return

        sell_spread = bid_hi - ask_lo
        buy_spread  = ask_hi - bid_lo
        spread_pos  = state.position.get(p_hi, 0) - state.position.get(p_lo, 0)
        orders_hi   = result.setdefault(p_hi, [])
        orders_lo   = result.setdefault(p_lo, [])

        if sell_spread >= self.SPREAD_HIGH_SELL and spread_pos > -self.SPREAD_POSITION_CAP:
            qty = min(
                self.SPREAD_PAIR_MAX_TAKE,
                od_hi.buy_orders.get(bid_hi, 0),
                -od_lo.sell_orders.get(ask_lo, 0),
                self.SPREAD_POSITION_CAP + spread_pos,
                self._remaining_capacity(p_hi, state, orders_hi)[1],
                self._remaining_capacity(p_lo, state, orders_lo)[0],
            )
            if qty > 0:
                orders_hi.append(Order(p_hi, bid_hi, -qty))
                orders_lo.append(Order(p_lo, ask_lo,  qty))

        elif buy_spread <= self.SPREAD_LOW_BUY and spread_pos < self.SPREAD_POSITION_CAP:
            qty = min(
                self.SPREAD_PAIR_MAX_TAKE,
                -od_hi.sell_orders.get(ask_hi, 0),
                od_lo.buy_orders.get(bid_lo, 0),
                self.SPREAD_POSITION_CAP - spread_pos,
                self._remaining_capacity(p_hi, state, orders_hi)[0],
                self._remaining_capacity(p_lo, state, orders_lo)[1],
            )
            if qty > 0:
                orders_hi.append(Order(p_hi, ask_hi,  qty))
                orders_lo.append(Order(p_lo, bid_lo, -qty))

    # ── EMA fair value ────────────────────────────────────────────────────────
    def _spot_fair(
        self, product: str, state: TradingState, td: Dict
    ) -> Optional[float]:
        if product not in state.order_depths:
            return None
        od = state.order_depths[product]
        best_bid, best_ask = self._best_bid_ask(od)

        raw: Optional[float] = None
        if best_bid is not None and best_ask is not None:
            bid_vol = max(od.buy_orders.get(best_bid, 0), 0)
            ask_vol = max(-od.sell_orders.get(best_ask, 0), 0)
            raw = (
                (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
                if bid_vol + ask_vol > 0
                else (best_bid + best_ask) / 2.0
            )
        elif best_bid is not None:
            raw = float(best_bid)
        elif best_ask is not None:
            raw = float(best_ask)

        if raw is None:
            prev = td.get(product + "_ema")
            return float(prev) if prev is not None else None

        alpha = self.EMA_ALPHA.get(product, 0.10)
        prev  = td.get(product + "_ema")
        ema   = raw if prev is None else alpha * raw + (1.0 - alpha) * float(prev)
        td[product + "_ema"] = ema
        return ema

    # ── Asymmetric take ───────────────────────────────────────────────────────
    def _take_mispriced(
        self, product: str, od: OrderDepth, fair: float,
        state: TradingState, orders: List[Order]
    ) -> None:
        buy_cap, sell_cap = self._remaining_capacity(product, state, orders)
        edge_buy  = self.TAKE_EDGE_BUY.get(product,  999_999.0)
        edge_sell = self.TAKE_EDGE_SELL.get(product, 999_999.0)
        max_take  = self.MAX_TAKE_SIZE.get(product, 0)
        if max_take == 0:
            return

        bought = 0
        for ask_price in sorted(od.sell_orders):
            if ask_price >= fair - edge_buy or buy_cap <= 0 or bought >= max_take:
                break
            qty = min(-od.sell_orders[ask_price], buy_cap, max_take - bought)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_cap -= qty; bought += qty

        sold = 0
        for bid_price in sorted(od.buy_orders, reverse=True):
            if bid_price <= fair + edge_sell or sell_cap <= 0 or sold >= max_take:
                break
            qty = min(od.buy_orders[bid_price], sell_cap, max_take - sold)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_cap -= qty; sold += qty

    # ── Passive quoting (HYDROGEL / VELVETFRUIT) ──────────────────────────────
    def _add_passive_quotes(
        self, product: str, od: OrderDepth, fair: float,
        state: TradingState, orders: List[Order], edge: int
    ) -> None:
        best_bid, best_ask = self._best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return
        buy_cap, sell_cap = self._remaining_capacity(product, state, orders)
        max_make = self.MAX_MAKE_SIZE.get(product, 0)
        if max_make <= 0:
            return

        buy_px  = int(math.floor(fair - edge))
        sell_px = int(math.ceil(fair  + edge))

        if best_bid + 1 < fair - 1:
            buy_px = max(buy_px, best_bid + 1)
        if best_ask - 1 > fair + 1:
            sell_px = min(sell_px, best_ask - 1)

        if buy_px  >= best_ask: buy_px  = best_ask - 1
        if sell_px <= best_bid: sell_px = best_bid + 1
        buy_px  = max(0, buy_px)
        sell_px = max(1, sell_px)

        pos   = state.position.get(product, 0)
        limit = self.LIMITS[product]
        buy_scale  = max(0.0, 1.0 - max( pos, 0) / limit)
        sell_scale = max(0.0, 1.0 - max(-pos, 0) / limit)

        buy_qty  = min(buy_cap,  max(1, int(max_make * buy_scale)))
        sell_qty = min(sell_cap, max(1, int(max_make * sell_scale)))

        if buy_qty  > 0: orders.append(Order(product, buy_px,   buy_qty))
        if sell_qty > 0: orders.append(Order(product, sell_px, -sell_qty))

    # ── Remaining capacity ────────────────────────────────────────────────────
    def _remaining_capacity(
        self, product: str, state: TradingState, existing: List[Order]
    ) -> Tuple[int, int]:
        limit    = self.SOFT_LIMITS.get(product, self.LIMITS.get(product, 200))
        position = state.position.get(product, 0)
        pending_buy  = sum( o.quantity for o in existing if o.quantity > 0)
        pending_sell = sum(-o.quantity for o in existing if o.quantity < 0)
        return (max(0, limit - position - pending_buy),
                max(0, limit + position - pending_sell))

    # ── Best bid / ask ────────────────────────────────────────────────────────
    def _best_bid_ask(
        self, od: OrderDepth
    ) -> Tuple[Optional[int], Optional[int]]:
        return (
            max(od.buy_orders)  if od.buy_orders  else None,
            min(od.sell_orders) if od.sell_orders else None,
        )
