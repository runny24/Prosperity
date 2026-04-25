import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState


class Trader:
    """
    Round 3 v4 platform-calibrated candidate.

    Built from r3_v3 platform logs:
    - Disable weak vouchers: VEV_4000, VEV_4500, VEV_5100, VEV_5200, VEV_6000, VEV_6500.
    - Focus visible-liquidity trading on VEV_5300 and VEV_5400.
    - Use short-bias no-flip logic for VEV_5300 / VEV_5400 so profitable shorts
      are not bought back and flipped into long inventory late in the run.
    - Disable VEV_5500 after negative platform v3 result.
    - Keep spot strategies from v2.
    """

    LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300,
        "VEV_4500": 300,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
        "VEV_6000": 300,
        "VEV_6500": 300,
    }

    SOFT_LIMITS: Dict[str, int] = {
        "VEV_5000": 80,
        "VEV_5300": 300,
        "VEV_5400": 300,
    }

    VOUCHER_STRIKES: Dict[str, int] = {
        "VEV_5000": 5000,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
    }

    VOUCHER_TIME_VALUE: Dict[str, float] = {
        "VEV_5000": 5.0,
        "VEV_5300": 50.0,
        "VEV_5400": 16.0,
    }

    DISABLED_VOUCHERS = {
        "VEV_4000",
        "VEV_4500",
        "VEV_5100",
        "VEV_5200",
        "VEV_5500",
        "VEV_6000",
        "VEV_6500",
    }

    SHORT_BIAS_VOUCHERS = {"VEV_5300", "VEV_5400"}

    TAKE_EDGE: Dict[str, float] = {
        "HYDROGEL_PACK": 8.0,
        "VELVETFRUIT_EXTRACT": 3.0,
        "VEV_5000": 6.0,
        "VEV_5300": 3.0,
        "VEV_5400": 1.0,
    }

    MAKE_EDGE: Dict[str, int] = {
        "HYDROGEL_PACK": 10,
        "VELVETFRUIT_EXTRACT": 4,
    }

    MAX_TAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK": 24,
        "VELVETFRUIT_EXTRACT": 30,
        "VEV_5000": 12,
        "VEV_5300": 80,
        "VEV_5400": 80,
    }

    MAX_MAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK": 18,
        "VELVETFRUIT_EXTRACT": 24,
    }

    EMA_ALPHA: Dict[str, float] = {
        "HYDROGEL_PACK": 0.08,
        "VELVETFRUIT_EXTRACT": 0.12,
    }

    def bid(self):
        return 15

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_data = {}

        velvet_fair = self._spot_fair("VELVETFRUIT_EXTRACT", state, trader_data)

        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = self._trade_spot("HYDROGEL_PACK", state, trader_data)

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            result["VELVETFRUIT_EXTRACT"] = self._trade_spot(
                "VELVETFRUIT_EXTRACT", state, trader_data, velvet_fair
            )

        if velvet_fair is not None:
            for product in state.order_depths:
                if product.startswith("VEV_"):
                    result[product] = self._trade_voucher(product, velvet_fair, state)

        trader_data_out = json.dumps(trader_data, separators=(",", ":"))
        return result, conversions, trader_data_out

    def _trade_spot(
        self,
        product: str,
        state: TradingState,
        trader_data: Dict,
        known_fair: Optional[float] = None,
    ) -> List[Order]:
        fair = known_fair if known_fair is not None else self._spot_fair(product, state, trader_data)
        if fair is None:
            return []

        order_depth = state.order_depths[product]
        orders: List[Order] = []
        self._take_mispriced_liquidity(product, order_depth, fair, state, orders)

        make_edge = self.MAKE_EDGE.get(product)
        if make_edge is not None:
            self._add_passive_quotes(product, order_depth, fair, state, orders, make_edge)

        return orders

    def _trade_voucher(self, product: str, underlying_fair: float, state: TradingState) -> List[Order]:
        if product in self.DISABLED_VOUCHERS or product not in self.VOUCHER_STRIKES:
            return []

        strike = self.VOUCHER_STRIKES[product]
        time_value = self.VOUCHER_TIME_VALUE[product]
        fair = max(underlying_fair - strike, 0.0) + time_value

        order_depth = state.order_depths[product]
        orders: List[Order] = []
        if product in self.SHORT_BIAS_VOUCHERS:
            self._take_short_bias_liquidity(product, order_depth, fair, state, orders)
        else:
            self._take_mispriced_liquidity(product, order_depth, fair, state, orders)
        return orders

    def _take_short_bias_liquidity(
        self,
        product: str,
        order_depth: OrderDepth,
        fair: float,
        state: TradingState,
        orders: List[Order],
    ) -> None:
        position = state.position.get(product, 0)
        soft_limit = self.SOFT_LIMITS.get(product, self.LIMITS[product])
        take_edge = self.TAKE_EDGE.get(product, 999999.0)
        max_take = self.MAX_TAKE_SIZE.get(product, 0)

        # If bids are rich, sell into them down to the short soft limit.
        sell_capacity = soft_limit + position
        sold = 0
        for bid_price in sorted(order_depth.buy_orders, reverse=True):
            if bid_price <= fair + take_edge or sell_capacity <= 0 or sold >= max_take:
                break
            available = order_depth.buy_orders[bid_price]
            quantity = min(available, sell_capacity, max_take - sold)
            if quantity > 0:
                orders.append(Order(product, bid_price, -quantity))
                sell_capacity -= quantity
                sold += quantity

        # Buy only to reduce an existing short. Never flip long.
        projected_position = position - sold
        if projected_position >= 0:
            return

        buy_to_flat_capacity = -projected_position
        bought = 0
        for ask_price in sorted(order_depth.sell_orders):
            if ask_price >= fair - take_edge or buy_to_flat_capacity <= 0 or bought >= max_take:
                break
            available = -order_depth.sell_orders[ask_price]
            quantity = min(available, buy_to_flat_capacity, max_take - bought)
            if quantity > 0:
                orders.append(Order(product, ask_price, quantity))
                buy_to_flat_capacity -= quantity
                bought += quantity

    def _spot_fair(self, product: str, state: TradingState, trader_data: Dict) -> Optional[float]:
        if product not in state.order_depths:
            return None

        order_depth = state.order_depths[product]
        best_bid, best_ask = self._best_bid_ask(order_depth)

        raw_fair = None
        if best_bid is not None and best_ask is not None:
            bid_volume = max(order_depth.buy_orders.get(best_bid, 0), 0)
            ask_volume = max(-order_depth.sell_orders.get(best_ask, 0), 0)
            if bid_volume + ask_volume > 0:
                raw_fair = (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)
            else:
                raw_fair = (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            raw_fair = float(best_bid)
        elif best_ask is not None:
            raw_fair = float(best_ask)

        if raw_fair is None:
            key = product + "_ema"
            previous = trader_data.get(key)
            return float(previous) if previous is not None else None

        key = product + "_ema"
        alpha = self.EMA_ALPHA.get(product, 0.10)
        previous = trader_data.get(key)
        ema = raw_fair if previous is None else alpha * raw_fair + (1.0 - alpha) * float(previous)
        trader_data[key] = ema
        return ema

    def _take_mispriced_liquidity(
        self,
        product: str,
        order_depth: OrderDepth,
        fair: float,
        state: TradingState,
        orders: List[Order],
    ) -> None:
        buy_capacity, sell_capacity = self._remaining_capacity(product, state, orders)
        take_edge = self.TAKE_EDGE.get(product, 999999.0)
        max_take = self.MAX_TAKE_SIZE.get(product, 0)

        bought = 0
        for ask_price in sorted(order_depth.sell_orders):
            if ask_price >= fair - take_edge or buy_capacity <= 0 or bought >= max_take:
                break
            available = -order_depth.sell_orders[ask_price]
            quantity = min(available, buy_capacity, max_take - bought)
            if quantity > 0:
                orders.append(Order(product, ask_price, quantity))
                buy_capacity -= quantity
                bought += quantity

        sold = 0
        for bid_price in sorted(order_depth.buy_orders, reverse=True):
            if bid_price <= fair + take_edge or sell_capacity <= 0 or sold >= max_take:
                break
            available = order_depth.buy_orders[bid_price]
            quantity = min(available, sell_capacity, max_take - sold)
            if quantity > 0:
                orders.append(Order(product, bid_price, -quantity))
                sell_capacity -= quantity
                sold += quantity

    def _add_passive_quotes(
        self,
        product: str,
        order_depth: OrderDepth,
        fair: float,
        state: TradingState,
        orders: List[Order],
        edge: int,
    ) -> None:
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return

        buy_capacity, sell_capacity = self._remaining_capacity(product, state, orders)
        max_make = self.MAX_MAKE_SIZE.get(product, 0)
        if max_make <= 0:
            return

        buy_price = int(math.floor(fair - edge))
        sell_price = int(math.ceil(fair + edge))

        if best_bid + 1 < fair - 1:
            buy_price = max(buy_price, best_bid + 1)
        if best_ask - 1 > fair + 1:
            sell_price = min(sell_price, best_ask - 1)

        if buy_price >= best_ask:
            buy_price = best_ask - 1
        if sell_price <= best_bid:
            sell_price = best_bid + 1

        pos = state.position.get(product, 0)
        limit = self.LIMITS[product]
        buy_scale = max(0.0, 1.0 - max(pos, 0) / limit)
        sell_scale = max(0.0, 1.0 - max(-pos, 0) / limit)

        buy_quantity = min(buy_capacity, max(1, int(max_make * buy_scale)))
        sell_quantity = min(sell_capacity, max(1, int(max_make * sell_scale)))

        if buy_quantity > 0:
            orders.append(Order(product, buy_price, buy_quantity))
        if sell_quantity > 0:
            orders.append(Order(product, sell_price, -sell_quantity))

    def _remaining_capacity(
        self,
        product: str,
        state: TradingState,
        existing_orders: List[Order],
    ) -> Tuple[int, int]:
        limit = self.SOFT_LIMITS.get(product, self.LIMITS[product])
        position = state.position.get(product, 0)
        pending_buy = 0
        pending_sell = 0
        for order in existing_orders:
            if order.quantity > 0:
                pending_buy += order.quantity
            else:
                pending_sell += -order.quantity

        buy_capacity = limit - position - pending_buy
        sell_capacity = limit + position - pending_sell
        return max(0, buy_capacity), max(0, sell_capacity)

    def _best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        return best_bid, best_ask