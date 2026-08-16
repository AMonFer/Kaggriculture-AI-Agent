"""
Macro-Planner & Resource Allocation Engine (Capa 2) for Kaggriculture.
Optimizes dynamic crop portfolio ROI, Fibonacci labor scaling, land expansion,
and terminal horizon harvest cutoffs.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models.constants import (
    BOARD_SIZE,
    QUADRANT_SIZE,
    QUADRANT_COSTS,
    CROP_SPECS,
    CropType,
    ProductType,
    get_fibonacci_cost,
    FIBONACCI_COSTS,
)
from models.state_representation import (
    EmptyTile,
    FarmState,
    GameState,
    PlantTile,
    StructureTile,
    WeedTile,
)
from engine.market_simulator import MarketSimulator


@dataclass
class MacroPlan:
    day: int
    seed_orders: Dict[str, int] = field(default_factory=dict)
    hands_to_hire: int = 0
    buy_land_quadrant: Optional[str] = None
    target_crop_distribution: Dict[str, float] = field(default_factory=dict)
    preferred_seed_order: List[str] = field(default_factory=list)
    total_budget_allocated: float = 0.0
    liquidation_orders: Dict[str, int] = field(default_factory=dict)


class MacroPlanner:
    """
    Evaluates daily macroeconomic opportunities and generates resource allocation plans.
    """

    def __init__(self) -> None:
        self.quadrant_priority_order = ["NE", "SW", "SE"]

    def compute_crop_rois(
        self,
        game_state: GameState,
        market_sim: MarketSimulator,
    ) -> Dict[str, float]:
        """
        Calculates dynamic ROI per tile per day for each crop.
        Applies terminal horizon cutoffs: returns -inf for crops that cannot mature before day 30.
        """
        current_day = game_state.day
        days_left = 30 - current_day  # Days remaining in season (1..30)
        rois: Dict[str, float] = {}

        # On the final 2 days no new planting allowed
        if days_left <= 2:
            for crop_name in CROP_SPECS:
                rois[crop_name] = -float("inf")
            return rois

        for crop_name, spec in CROP_SPECS.items():
            first_yield = spec.time_to_first_yield
            max_yield_day = spec.time_to_max_yield
            seed_cost = spec.seed_cost
            current_market_inv = game_state.market.get_inventory(crop_name)
            est_unit_price = market_sim.compute_price(crop_name, current_market_inv)

            # 1. Melon (One-time, peak at day 10, yield 6)
            if crop_name == CropType.MELON.value:
                if days_left < 10:
                    rois[crop_name] = -float("inf")
                else:
                    expected_yield = 6
                    effective_days = 10
                    # Estimate batch sell price for 6 units
                    batch_res = market_sim.simulate_batch_sell(crop_name, current_market_inv, expected_yield)
                    revenue = batch_res.total_revenue
                    net_profit = revenue - seed_cost
                    rois[crop_name] = net_profit / float(effective_days)

            # 2. Carrot (One-time, peak at day 3 with yield 3, first yield at day 2 with yield 2)
            elif crop_name == CropType.CARROT.value:
                if days_left < 2:
                    rois[crop_name] = -float("inf")
                elif days_left == 2:
                    # Quick short cycle: harvest at day 2
                    expected_yield = 2
                    effective_days = 2
                    batch_res = market_sim.simulate_batch_sell(crop_name, current_market_inv, expected_yield)
                    net_profit = batch_res.total_revenue - seed_cost
                    rois[crop_name] = net_profit / float(effective_days)
                else:
                    # Standard 3-day cycle
                    expected_yield = 3
                    effective_days = 3
                    batch_res = market_sim.simulate_batch_sell(crop_name, current_market_inv, expected_yield)
                    net_profit = batch_res.total_revenue - seed_cost
                    rois[crop_name] = net_profit / float(effective_days)

            # 3. Wheat (One-time, peak at day 4 with yield 4, first yield at day 2 with yield 2)
            elif crop_name == CropType.WHEAT.value:
                if days_left < 2:
                    rois[crop_name] = -float("inf")
                elif days_left < 4:
                    expected_yield = 2
                    effective_days = 2
                    batch_res = market_sim.simulate_batch_sell(crop_name, current_market_inv, expected_yield)
                    net_profit = batch_res.total_revenue - seed_cost
                    rois[crop_name] = net_profit / float(effective_days)
                else:
                    expected_yield = 4
                    effective_days = 4
                    batch_res = market_sim.simulate_batch_sell(crop_name, current_market_inv, expected_yield)
                    net_profit = batch_res.total_revenue - seed_cost
                    rois[crop_name] = net_profit / float(effective_days)

            # 4. Tomato (Ongoing, yields at ages 8, 9, 10, 11)
            elif crop_name == CropType.TOMATO.value:
                if days_left < 8:
                    rois[crop_name] = -float("inf")
                else:
                    harvests_possible = min(4, days_left - 7)
                    effective_days = min(11, days_left)
                    batch_res = market_sim.simulate_batch_sell(crop_name, current_market_inv, harvests_possible)
                    net_profit = batch_res.total_revenue - seed_cost
                    rois[crop_name] = net_profit / float(effective_days)

            # 5. Strawberry (Ongoing, yields at ages 10, 12, 14, 16)
            elif crop_name == CropType.STRAWBERRY.value:
                if days_left < 10:
                    rois[crop_name] = -float("inf")
                else:
                    harvests_possible = 1
                    if days_left >= 12: harvests_possible += 1
                    if days_left >= 14: harvests_possible += 1
                    if days_left >= 16: harvests_possible += 1
                    effective_days = min(16, days_left)
                    batch_res = market_sim.simulate_batch_sell(crop_name, current_market_inv, harvests_possible)
                    net_profit = batch_res.total_revenue - seed_cost
                    rois[crop_name] = net_profit / float(effective_days)

        return rois

    def evaluate_land_expansion(
        self,
        game_state: GameState,
        best_daily_roi: float,
    ) -> Optional[str]:
        """
        Evaluates whether buying the next quadrant (NE, SW, SE) is financially amortizable.
        """
        my_farm = game_state.my_farm
        unlocked = my_farm.unlocked_quadrants
        current_day = game_state.day
        days_left = 30 - current_day

        # Find next available quadrant to buy
        next_quad: Optional[str] = None
        for q in self.quadrant_priority_order:
            if q not in unlocked:
                next_quad = q
                break

        if next_quad is None:
            return None  # All quadrants already unlocked

        quad_cost = QUADRANT_COSTS.get(next_quad, 99999)

        # Minimum required days to amortize: NE=8 days, SW=10 days, SE=12 days
        min_days_required = {"NE": 8, "SW": 10, "SE": 12}.get(next_quad, 10)
        if days_left < min_days_required:
            return None

        if best_daily_roi <= 2.0:
            return None

        # Cash buffer requirement: must have cost + $300 reserve for seeds and labor
        cash_reserve_needed = quad_cost + 300.0
        if my_farm.money < cash_reserve_needed:
            return None

        # Check current farm utilization: at least 60% of current tiles should be occupied/cultivated
        total_unlocked_tiles = len(unlocked) * 25
        active_plants = my_farm.get_plant_count()
        if (active_plants / float(total_unlocked_tiles)) < 0.50 and total_unlocked_tiles >= 25:
            # Don't expand if current land is underutilized
            return None

        # Projected net return on 25 new tiles over remaining days
        projected_new_revenue = 25 * (days_left - 1) * best_daily_roi * 0.75
        if projected_new_revenue > (quad_cost + 200):
            return next_quad

        return None

    def evaluate_labor_needs(
        self,
        game_state: GameState,
        target_plantings: int,
        available_budget: float,
    ) -> int:
        """
        Calculates optimal number of farm hands to hire today by comparing
        estimated daily action workload against Fibonacci marginal hiring costs.
        """
        my_farm = game_state.my_farm
        current_day = game_state.day
        days_left = 30 - current_day

        # Count active farm workload
        plant_count = my_farm.get_plant_count()
        harvests_ready = 0
        weeds_count = 0

        for row in my_farm.tiles:
            for tile in row:
                if isinstance(tile, PlantTile) and tile.is_ready_to_harvest:
                    harvests_ready += 1
                elif isinstance(tile, WeedTile):
                    weeds_count += 1

        # Total estimated daily work actions needed
        # (Watering + Harvesting + Weeding + Planting + Transit Overhead)
        total_actions = plant_count + harvests_ready + weeds_count + target_plantings
        # Estimated transit and shed drop overhead
        transit_overhead = int(total_actions * 0.35)
        workload = total_actions + transit_overhead

        # Farmer alone provides 24 actions at $0
        extra_work_needed = max(0, workload - 24)

        if extra_work_needed == 0:
            return 0

        # Each hired hand provides ~20 effective productive actions
        desired_hands = int(math.ceil(extra_work_needed / 20.0))

        # In final closing days (days 28-29), only hire if heavy harvest backlog exists
        if days_left <= 2:
            desired_hands = min(desired_hands, int(math.ceil(harvests_ready / 18.0)))

        # Cap max hands
        desired_hands = min(desired_hands, 8)

        # Budget constraint: calculate cumulative Fibonacci cost
        affordable_hands = 0
        cumulative_cost = 0

        for h in range(desired_hands):
            hand_cost = get_fibonacci_cost(h)
            if cumulative_cost + hand_cost <= available_budget:
                cumulative_cost += hand_cost
                affordable_hands += 1
            else:
                break

        return affordable_hands

    def allocate_seed_budget(
        self,
        game_state: GameState,
        rois: Dict[str, float],
        available_money: float,
    ) -> Tuple[Dict[str, int], List[str]]:
        """
        Allocates seed purchase orders to the highest-ROI crops within space and budget limits.
        """
        my_farm = game_state.my_farm
        current_day = game_state.day
        days_left = 30 - current_day

        if days_left <= 2 or available_money < 10:
            return {}, []

        # Filter viable crops with ROI > 0
        viable_crops = [(c, roi) for c, roi in rois.items() if roi > 0 and not math.isinf(roi)]
        viable_crops.sort(key=lambda x: x[1], reverse=True)

        if not viable_crops:
            return {}, []

        preferred_order = [c for c, _ in viable_crops]

        # Calculate empty tiles in unlocked quadrants
        empty_unlocked_tiles = 0
        for r_idx in range(BOARD_SIZE):
            for c_idx in range(BOARD_SIZE):
                tile = my_farm.tiles[r_idx][c_idx]
                if isinstance(tile, EmptyTile):
                    empty_unlocked_tiles += 1

        existing_seeds = sum(my_farm.seeds.values())
        seeds_needed = max(0, empty_unlocked_tiles - existing_seeds)

        # Cap batch purchase to what can be managed and stored
        max_seeds_to_buy = min(empty_unlocked_tiles + 4, 30)
        if seeds_needed <= 0 and existing_seeds >= 10:
            return {}, preferred_order

        total_to_buy = min(seeds_needed if seeds_needed > 0 else 5, max_seeds_to_buy)

        seed_orders: Dict[str, int] = {}
        remaining_budget = available_money

        # Portfolio allocation: 75% to best crop, 25% to second best (if available and positive)
        top_crop = viable_crops[0][0]
        top_spec = CROP_SPECS[top_crop]

        if len(viable_crops) > 1 and viable_crops[1][1] > (viable_crops[0][1] * 0.7):
            second_crop = viable_crops[1][0]
            second_spec = CROP_SPECS[second_crop]

            qty_top = int(math.ceil(total_to_buy * 0.70))
            qty_second = total_to_buy - qty_top

            # Budget check
            cost_top = qty_top * top_spec.seed_cost
            if cost_top > remaining_budget:
                qty_top = int(remaining_budget // top_spec.seed_cost)
            remaining_budget -= qty_top * top_spec.seed_cost

            cost_second = qty_second * second_spec.seed_cost
            if cost_second > remaining_budget:
                qty_second = int(remaining_budget // second_spec.seed_cost)

            if qty_top > 0:
                seed_orders[top_crop] = qty_top
            if qty_second > 0:
                seed_orders[second_crop] = qty_second
        else:
            qty_top = min(total_to_buy, int(remaining_budget // top_spec.seed_cost))
            if qty_top > 0:
                seed_orders[top_crop] = qty_top

        return seed_orders, preferred_order

    def generate_daily_macro_plan(
        self,
        game_state: GameState,
        market_sim: MarketSimulator,
    ) -> MacroPlan:
        """
        Consolidates all macro-level financial and operational decisions for the day.
        """
        current_day = game_state.day
        days_left = 30 - current_day
        my_farm = game_state.my_farm
        starting_cash = my_farm.money

        # 1. Compute dynamic ROIs
        rois = self.compute_crop_rois(game_state, market_sim)
        best_roi = max(rois.values()) if rois else 0.0

        # 2. Evaluate land expansion
        buy_land = self.evaluate_land_expansion(game_state, best_roi)
        spent_on_land = QUADRANT_COSTS.get(buy_land, 0) if buy_land else 0.0
        cash_after_land = max(0.0, starting_cash - spent_on_land)

        # 3. Seed budget allocation
        # Keep a 15% cash buffer for labor
        seed_budget = cash_after_land * 0.85
        seed_orders, preferred_seed_order = self.allocate_seed_budget(game_state, rois, seed_budget)

        seed_spend = sum(seed_orders.get(c, 0) * CROP_SPECS[c].seed_cost for c in seed_orders)
        cash_after_seeds = max(0.0, cash_after_land - seed_spend)

        # 4. Evaluate labor hiring (Fibonacci scaling)
        target_plantings = sum(seed_orders.values()) + sum(my_farm.seeds.values())
        hands_to_hire = self.evaluate_labor_needs(game_state, target_plantings, cash_after_seeds)

        # 5. Liquidation planning for closing season (days 27-29)
        liquidation_orders: Dict[str, int] = {}
        if days_left <= 3:
            for prod_name, qty in my_farm.shed.items():
                if qty > 0:
                    current_inv = game_state.market.get_inventory(prod_name)
                    # Sell down to floor price 2 if days_left == 1, or 4 if days_left > 1
                    min_p = 2 if days_left <= 1 else 4
                    sell_batch = market_sim.find_optimal_sell_batch(
                        resource=prod_name,
                        current_inventory=current_inv,
                        available_quantity=qty,
                        min_acceptable_price=min_p,
                    )
                    if sell_batch > 0:
                        liquidation_orders[prod_name] = sell_batch

        return MacroPlan(
            day=current_day,
            seed_orders=seed_orders,
            hands_to_hire=hands_to_hire,
            buy_land_quadrant=buy_land,
            preferred_seed_order=preferred_seed_order,
            total_budget_allocated=spent_on_land + seed_spend,
            liquidation_orders=liquidation_orders,
        )
