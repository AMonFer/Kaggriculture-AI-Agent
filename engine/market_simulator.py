"""
Market Analytics Engine (Layer 1) for Kaggriculture.
Replicates the way the kaggle logic of how the market works
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from models.constants import (
    BASE_MARKET_INVENTORY,
    PRICE_FLOOR,
    TURNS_PER_DAY,
    TOWN_SHOP_SELL_INTERVAL_TURNS,
    TOWN_CENTER_SELL_INTERVAL_TURNS,
    MARKET_PARAMS,
    TOWN_SHOPS,
    MarketParamSpec,
    ProductType,
)


@dataclass(frozen=True)
class BatchSellResult:
    resource: str
    units_sold: int
    total_revenue: int
    avg_price: float
    marginal_price: int
    final_inventory: int
    price_trajectory: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class BatchBuyResult:
    resource: str
    units_bought: int
    total_cost: int
    avg_price: float
    marginal_price: int
    final_inventory: int
    price_trajectory: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class PriceProjection:
    day: int
    inventory: int
    price: int
    daily_drain: int
    daily_sales: int


class MarketSimulator:
    """
    Market analytics and price forecasting engine.
    """

    def __init__(self, market_params_override: Optional[Dict[str, Dict]] = None):
        """
        Initializes the market simulator with default or overridden market parameters.
        """
        self.params: Dict[str, MarketParamSpec] = dict(MARKET_PARAMS)
        if market_params_override:
            for res_name, custom_cfg in market_params_override.items():
                if res_name in self.params:
                    orig = self.params[res_name]
                    self.params[res_name] = MarketParamSpec(
                        base=custom_cfg.get("base", orig.base),
                        I0=custom_cfg.get("I0", orig.I0),
                        T=custom_cfg.get("T", orig.T),
                        below_func=custom_cfg.get("below_func", orig.below_func),
                        below_target=custom_cfg.get("below_target", orig.below_target),
                        above_func=custom_cfg.get("above_func", orig.above_func),
                        above_target=custom_cfg.get("above_target", orig.above_target),
                    )

    @staticmethod
    def _eval_f(func_name: str, x: float) -> float:
        #Evaluates a given math function
        if x <= 0:
            return 0.0
        if func_name == "linear":
            return float(x)
        elif func_name == "sq":
            return float(x * x)
        elif func_name == "sqrt":
            return math.sqrt(x)
        elif func_name == "log":
            return math.log(1.0 + x)
        elif func_name == "log10":
            return math.log10(1.0 + x)
        else:
            raise ValueError(f"Unknown shape function: {func_name}")

    def compute_price(self, resource: str, inventory: int) -> int:
        """
        Computes the instantaneous unit market price for a given resource and inventory.
        Matches the exact formula of the Kaggriculture simulation engine.
        """
        spec = self.params.get(resource)
        if not spec:
            raise KeyError(f"Unknown resource: {resource}")

        base = spec.base
        i0 = spec.I0
        t = spec.T

        if inventory == i0:
            return base

        diff = abs(inventory - i0)

        if inventory < i0:
            # Scarcity -> Price increases
            sign = 1
            func_name = spec.below_func
            target = spec.below_target
        else:
            # Glut -> Price collapses
            sign = -1
            func_name = spec.above_func
            target = spec.above_target

        f_t = self._eval_f(func_name, t)
        if f_t == 0:
            amp = 0.0
        else:
            amp = (target * base) / f_t

        f_val = self._eval_f(func_name, diff)
        raw_price = base + sign * amp * f_val
        rounded_price = round(raw_price)

        return max(PRICE_FLOOR, rounded_price)

    def simulate_batch_sell(self, resource: str, current_inventory: int, quantity: int) -> BatchSellResult:
        """
        Simulates selling N units one-by-one according to the Kaggle engine rules.
        """
        if quantity <= 0:
            p = self.compute_price(resource, current_inventory)
            return BatchSellResult(
                resource=resource,
                units_sold=0,
                total_revenue=0,
                avg_price=float(p),
                marginal_price=p,
                final_inventory=current_inventory,
                price_trajectory=[],
            )

        inv = current_inventory
        total_rev = 0
        trajectory: List[int] = []
        last_price = 1

        for _ in range(quantity):
            p = self.compute_price(resource, inv)
            trajectory.append(p)
            total_rev += p
            last_price = p
            # If price is at the floor ($1), the unit is purchased but NOT added to inventory
            if p > PRICE_FLOOR:
                inv += 1

        avg_p = total_rev / float(quantity)

        return BatchSellResult(
            resource=resource,
            units_sold=quantity,
            total_revenue=total_rev,
            avg_price=avg_p,
            marginal_price=last_price,
            final_inventory=inv,
            price_trajectory=trajectory,
        )

    def simulate_batch_buy(self, resource: str, current_inventory: int, quantity: int) -> BatchBuyResult:
        """
        Simulates buying N units (Wheat or Fertilizer) one-by-one.
        Buy price is quoted at the post-buy inventory.
        """
        if quantity <= 0:
            p = self.compute_price(resource, current_inventory)
            return BatchBuyResult(
                resource=resource,
                units_bought=0,
                total_cost=0,
                avg_price=float(p),
                marginal_price=p,
                final_inventory=current_inventory,
                price_trajectory=[],
            )

        inv = current_inventory
        total_cost = 0
        trajectory: List[int] = []
        last_price = 1

        for _ in range(quantity):
            # Post-buy inventory shift
            inv = max(0, inv - 1)
            p = self.compute_price(resource, inv)
            trajectory.append(p)
            total_cost += p
            last_price = p

        avg_p = total_cost / float(quantity)

        return BatchBuyResult(
            resource=resource,
            units_bought=quantity,
            total_cost=total_cost,
            avg_price=avg_p,
            marginal_price=last_price,
            final_inventory=inv,
            price_trajectory=trajectory,
        )

    def find_optimal_sell_batch(
        self,
        resource: str,
        current_inventory: int,
        available_quantity: int,
        min_acceptable_price: int,
    ) -> int:
        """
        Finds the maximum quantity to sell such that every sold unit satisfies
        price >= min_acceptable_price.
        """
        if available_quantity <= 0 or min_acceptable_price <= PRICE_FLOOR:
            return available_quantity

        # Fast scan: check price step by step
        inv = current_inventory
        units_to_sell = 0
        for _ in range(available_quantity):
            p = self.compute_price(resource, inv)
            if p < min_acceptable_price:
                break
            units_to_sell += 1
            if p > PRICE_FLOOR:
                inv += 1

        return units_to_sell

    @staticmethod
    def get_daily_town_drain(unlocked_shops: List[str]) -> Dict[str, int]:
        """
        Calculates exact units consumed per day by the town center and all unlocked shops.
        - Town center: 1 of every product (except Fertilizer) per 24 turns (1 unit/day).
        - Each shop: consumes every 4 turns = 6 times per day.
        """
        daily_drain: Dict[str, int] = {p.value: 0 for p in ProductType}

        # Town center drain (1/day for all products except fertilizer)
        for p in ProductType:
            if p != ProductType.FERTILIZER:
                daily_drain[p.value] += 1

        # Unlocked shops drain
        ticks_per_day = TURNS_PER_DAY // TOWN_SHOP_SELL_INTERVAL_TURNS  # 24 // 4 = 6
        for shop_name in unlocked_shops:
            demands = TOWN_SHOPS.get(shop_name, {})
            for product, units_per_tick in demands.items():
                daily_drain[product] += units_per_tick * ticks_per_day

        return daily_drain

    def project_market_trajectory(
        self,
        initial_inventory: Dict[str, int],
        unlocked_shops: List[str],
        days_ahead: int,
        planned_sales_per_day: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, List[PriceProjection]]:
        """
        Deterministically projects market inventory and prices for each product
        over future days given current town consumption and planned sales.
        """
        daily_drain = self.get_daily_town_drain(unlocked_shops)
        projections: Dict[str, List[PriceProjection]] = {p.value: [] for p in ProductType}

        current_inv = dict(initial_inventory)

        for day_idx in range(days_ahead):
            for p_enum in ProductType:
                res = p_enum.value
                inv = current_inv.get(res, BASE_MARKET_INVENTORY)
                drain = daily_drain.get(res, 0)
                sales = 0
                if planned_sales_per_day and res in planned_sales_per_day:
                    sales_list = planned_sales_per_day[res]
                    if day_idx < len(sales_list):
                        sales = sales_list[day_idx]

                # First simulate sales for today
                sell_res = self.simulate_batch_sell(res, inv, sales)
                post_sell_inv = sell_res.final_inventory

                # Then town absorbs inventory at end of day
                new_inv = max(0, post_sell_inv - drain)
                price_end = self.compute_price(res, new_inv)

                projections[res].append(
                    PriceProjection(
                        day=day_idx,
                        inventory=new_inv,
                        price=price_end,
                        daily_drain=drain,
                        daily_sales=sales,
                    )
                )
                current_inv[res] = new_inv

        return projections

    def optimize_liquidation_schedule(
        self,
        resource: str,
        current_inventory: int,
        total_units_to_sell: int,
        days_remaining: int,
        unlocked_shops: List[str],
        min_price_threshold: int = 5,
    ) -> List[int]:
        """
        Calculates an optimal multi-day liquidation schedule.
        Splits total_units_to_sell across remaining days to maximize total revenue
        by taking advantage of daily town demand recovery.
        """
        if total_units_to_sell <= 0:
            return [0] * max(1, days_remaining)

        if days_remaining <= 1:
            return [total_units_to_sell]

        daily_drain = self.get_daily_town_drain(unlocked_shops).get(resource, 1)

        schedule = [0] * days_remaining
        remaining_units = total_units_to_sell
        sim_inv = current_inventory

        for day in range(days_remaining):
            is_last_day = (day == days_remaining - 1)
            if is_last_day:
                # Must liquidate everything on the last day
                schedule[day] = remaining_units
                break

            # Find how many we can sell today without crashing price below min_price_threshold
            # or exceeding daily absorption rate significantly
            qty_today = self.find_optimal_sell_batch(
                resource=resource,
                current_inventory=sim_inv,
                available_quantity=min(remaining_units, daily_drain * 2),
                min_acceptable_price=min_price_threshold,
            )

            # Ensure we make progress towards selling all units
            min_progress = remaining_units // (days_remaining - day)
            qty_today = max(qty_today, min_progress)
            qty_today = min(qty_today, remaining_units)

            schedule[day] = qty_today
            remaining_units -= qty_today

            # Advance simulated inventory
            sell_res = self.simulate_batch_sell(resource, sim_inv, qty_today)
            sim_inv = max(0, sell_res.final_inventory - daily_drain)

            if remaining_units <= 0:
                break

        return schedule
