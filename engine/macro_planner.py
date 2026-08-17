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
    ANIMAL_SPECS,
    ANIMAL_CUTOFF_DAYS,
    WHEAT_RESERVE_DAYS,
    MAX_WHEAT_RESERVE_IN_SHED,
    FERTILIZER_EFFECT_DURATION_DAYS,
    STRUCTURE_FOR_ANIMAL,
    BUILD_ACTION_FOR_STRUCTURE,
    CropType,
    AnimalType,
    ProductType,
    StructureType,
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
    animal_orders: Dict[str, int] = field(default_factory=dict)
    wheat_buy_orders: int = 0
    structure_build_orders: List[Tuple[str, Tuple[int, int]]] = field(default_factory=list)
    fertilizer_target_tiles: List[Tuple[int, int]] = field(default_factory=list)
    hands_to_hire: int = 0
    buy_land_quadrant: Optional[str] = None
    target_crop_distribution: Dict[str, float] = field(default_factory=dict)
    preferred_seed_order: List[str] = field(default_factory=list)
    total_budget_allocated: float = 0.0
    liquidation_orders: Dict[str, int] = field(default_factory=dict)
    animal_rois: Dict[str, float] = field(default_factory=dict)


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

    def compute_animal_rois(
        self,
        game_state: GameState,
        market_sim: MarketSimulator,
    ) -> Dict[str, float]:
        """
        Calculates dynamic ROI per tile per day for each animal type.
        Applies terminal cutoffs (Goose <= 14, Cow/Sheep <= 10).
        Factors in base production, daily CARE bonuses, daily fertilizer generation,
        purchase cost, and daily WHEAT feeding costs.
        """
        current_day = game_state.day
        days_left = 30 - current_day
        rois: Dict[str, float] = {}

        for animal_name, spec in ANIMAL_SPECS.items():
            cutoff = ANIMAL_CUTOFF_DAYS.get(animal_name, 10)
            if current_day > cutoff or days_left < spec.time_to_first_yield:
                rois[animal_name] = -float("inf")
                continue

            t1 = spec.time_to_first_yield
            interval = spec.yield_interval
            scheduled_yield_events = 1 + (days_left - t1) // interval

            # Daily CARE bonus: with feeding & daily care, pending_care_bonus increments by 1 per day.
            # On scheduled yield days, yield = 1 + pending_care_bonus = 1 + interval.
            units_per_yield = 1 + interval
            total_product_units = scheduled_yield_events * units_per_yield

            # Product revenue
            prod_res_name = spec.product_type.value
            prod_market_inv = game_state.market.get_inventory(prod_res_name)
            batch_sell = market_sim.simulate_batch_sell(prod_res_name, prod_market_inv, total_product_units)
            rev_prod = batch_sell.total_revenue

            # Fertilizer revenue (1 unit generated every day on farm: days_left - 1 days)
            fert_days = max(0, days_left - 1)
            fert_market_inv = game_state.market.get_inventory(ProductType.FERTILIZER.value)
            fert_sell = market_sim.simulate_batch_sell(ProductType.FERTILIZER.value, fert_market_inv, fert_days)
            rev_fert = fert_sell.total_revenue

            # Purchase cost
            purchase_cost = spec.purchase_cost

            # Feed cost: 1 wheat per day for fert_days
            wheat_market_inv = game_state.market.get_inventory(ProductType.WHEAT.value)
            feed_buy = market_sim.simulate_batch_buy(ProductType.WHEAT.value, wheat_market_inv, fert_days)
            feed_cost = feed_buy.total_cost

            net_profit = rev_prod + rev_fert - purchase_cost - feed_cost
            rois[animal_name] = net_profit / float(days_left)

        return rois

    def evaluate_wheat_reserve(
        self,
        game_state: GameState,
        market_sim: MarketSimulator,
        prospective_animals: int = 0,
    ) -> Tuple[int, float]:
        """
        Calculates needed wheat purchase to maintain a safety buffer of 4 days of food per animal,
        strictly capped at MAX_WHEAT_RESERVE_IN_SHED (16 units) to protect shed capacity.
        Returns (units_to_buy, cost).
        """
        my_farm = game_state.my_farm
        total_animals = my_farm.get_animal_count() + prospective_animals
        if total_animals <= 0:
            return 0, 0.0

        required_wheat = min(total_animals * WHEAT_RESERVE_DAYS, MAX_WHEAT_RESERVE_IN_SHED)
        current_wheat_in_shed = my_farm.shed.get(ProductType.WHEAT.value, 0)
        deficit = max(0, required_wheat - current_wheat_in_shed)

        if deficit <= 0:
            return 0, 0.0

        wheat_inv = game_state.market.get_inventory(ProductType.WHEAT.value)
        buy_res = market_sim.simulate_batch_buy(ProductType.WHEAT.value, wheat_inv, deficit)
        return deficit, float(buy_res.total_cost)

    def evaluate_fertilizer_shadow_matrix(
        self,
        game_state: GameState,
        market_sim: MarketSimulator,
    ) -> List[Tuple[int, int]]:
        """
        Evaluates marginal shadow value of applying fertilizer to unfertilized active crops.
        Priority order: STRAWBERRY (+4 units net) >= MELON (cycle reduction from 10 to 8 days) > TOMATO (+4 units) > Market Spot.
        Returns sorted list of tile coordinates (x, y) with highest shadow return.
        """
        my_farm = game_state.my_farm
        current_day = game_state.day
        fert_inv = game_state.market.get_inventory(ProductType.FERTILIZER.value)
        spot_fert_price = market_sim.compute_price(ProductType.FERTILIZER.value, fert_inv)

        ranked_targets: List[Tuple[float, Tuple[int, int]]] = []

        for y, row in enumerate(my_farm.tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, PlantTile):
                    if tile.fertilized_until_day >= current_day:
                        continue  # Already fertilized!

                    crop = tile.crop
                    crop_inv = game_state.market.get_inventory(crop)
                    crop_price = market_sim.compute_price(crop, crop_inv)

                    shadow_value = 0.0

                    if crop == CropType.STRAWBERRY.value:
                        # Yields double: +4 strawberries
                        shadow_value = 4.0 * crop_price
                    elif crop == CropType.MELON.value:
                        # Accelerates harvest from 10 days to 8 days (saving 2 days of land)
                        shadow_value = (6.0 * crop_price) * (2.0 / 8.0) * 1.5
                    elif crop == CropType.TOMATO.value:
                        # Yields double: +4 tomatoes
                        shadow_value = 4.0 * crop_price
                    elif crop in (CropType.CARROT.value, CropType.WHEAT.value):
                        # Standard single harvest +1 unit
                        shadow_value = 1.0 * crop_price

                    # Only prioritize applying fertilizer if shadow value exceeds spot selling value
                    if shadow_value >= spot_fert_price:
                        ranked_targets.append((shadow_value, (x, y)))

        # Sort descending by shadow value
        ranked_targets.sort(key=lambda item: item[0], reverse=True)
        return [pos for _, pos in ranked_targets]


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
        animal_count = my_farm.get_animal_count()
        harvests_ready = 0
        weeds_count = 0

        for row in my_farm.tiles:
            for tile in row:
                if isinstance(tile, PlantTile) and tile.is_ready_to_harvest:
                    harvests_ready += 1
                elif isinstance(tile, StructureTile) and tile.is_occupied:
                    if tile.yield_units > 0:
                        harvests_ready += 1
                    if tile.fertilizer_available:
                        harvests_ready += 1
                elif isinstance(tile, WeedTile):
                    weeds_count += 1

        # Total estimated daily work actions needed
        # (Watering + Harvesting + Weeding + Planting + Feeding/Caring + Transit Overhead)
        animal_actions = animal_count * 2  # Feed + Care
        total_actions = plant_count + animal_actions + harvests_ready + weeds_count + target_plantings
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

    def evaluate_livestock_plan(
        self,
        game_state: GameState,
        animal_rois: Dict[str, float],
        crop_rois: Dict[str, float],
        available_budget: float,
        market_sim: MarketSimulator,
    ) -> Tuple[Dict[str, int], List[Tuple[str, Tuple[int, int]]], float]:
        """
        Evaluates animal purchases and structure construction:
        - Only emits BUY_ANIMAL if an empty structure exists or BUILD_COOP/BUILD_PASTURE can be scheduled today.
        - Prioritizes GOOSE due to EGG price resilience, limiting COW/SHEEP to 1-2 initial units.
        - Verifies wheat safety reserve budget before approving animal purchase.
        """
        my_farm = game_state.my_farm
        current_day = game_state.day

        # Find empty structures and empty unlocked tiles for potential construction
        empty_coops = my_farm.get_empty_structures(StructureType.COOP.value)
        empty_pastures = my_farm.get_empty_structures(StructureType.PASTURE.value)

        # Unplaced animals in shed awaiting placement
        animals_in_shed = my_farm.get_animals_in_shed()
        pending_goose_in_shed = animals_in_shed.get(AnimalType.GOOSE.value, 0)
        pending_cow_in_shed = animals_in_shed.get(AnimalType.COW.value, 0)
        pending_sheep_in_shed = animals_in_shed.get(AnimalType.SHEEP.value, 0)

        # Usable empty structures accounting for shed animals
        net_empty_coops = max(0, len(empty_coops) - pending_goose_in_shed)
        net_empty_pastures = max(0, len(empty_pastures) - (pending_cow_in_shed + pending_sheep_in_shed))

        # Find empty tiles for building new structures
        empty_tiles: List[Tuple[int, int]] = []
        for y, row in enumerate(my_farm.tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, EmptyTile):
                    empty_tiles.append((x, y))

        animal_orders: Dict[str, int] = {}
        build_orders: List[Tuple[str, Tuple[int, int]]] = []
        spent_budget = 0.0

        # Don't buy animals in late season
        if current_day > 14 or available_budget < 350:
            return animal_orders, build_orders, spent_budget

        # Animal counts currently on farm
        goose_count = my_farm.get_animal_count(AnimalType.GOOSE.value) + pending_goose_in_shed
        cow_count = my_farm.get_animal_count(AnimalType.COW.value) + pending_cow_in_shed
        sheep_count = my_farm.get_animal_count(AnimalType.SHEEP.value) + pending_sheep_in_shed

        # Best crop ROI to compare with
        valid_crop_rois = [r for r in crop_rois.values() if r > 0 and not math.isinf(r)]
        best_crop_roi = max(valid_crop_rois) if valid_crop_rois else 5.0

        # Preference order: GOOSE first, then COW/SHEEP (capped at 1-2 units)
        candidate_animals = [
            (AnimalType.GOOSE.value, ANIMAL_SPECS[AnimalType.GOOSE.value], net_empty_coops, StructureType.COOP.value, goose_count, 3),
            (AnimalType.COW.value, ANIMAL_SPECS[AnimalType.COW.value], net_empty_pastures, StructureType.PASTURE.value, cow_count, 1),
            (AnimalType.SHEEP.value, ANIMAL_SPECS[AnimalType.SHEEP.value], net_empty_pastures, StructureType.PASTURE.value, sheep_count, 1),
        ]

        tile_idx = 0
        rem_budget = available_budget

        for animal_name, spec, net_empty_structs, struct_kind, cur_count, max_limit in candidate_animals:
            roi = animal_rois.get(animal_name, -float("inf"))
            if math.isinf(roi) or roi < 0:
                continue

            if cur_count >= max_limit:
                continue

            # Need at least competitive ROI vs crops or high early return
            if roi < (best_crop_roi * 0.8) and current_day > 6:
                continue

            # Check structure availability or ability to build
            can_house = False
            assigned_pos: Optional[Tuple[int, int]] = None

            if net_empty_structs > 0:
                can_house = True
                # Structure already exists
            elif tile_idx < len(empty_tiles):
                # We can schedule building structure today!
                assigned_pos = empty_tiles[tile_idx]
                tile_idx += 1
                can_house = True

            if not can_house:
                continue

            # Check wheat reserve cost
            wheat_deficit, wheat_cost = self.evaluate_wheat_reserve(
                game_state, market_sim, prospective_animals=sum(animal_orders.values()) + 1
            )

            total_needed = spec.purchase_cost + wheat_cost + 100.0  # $100 safety margin
            if rem_budget >= total_needed:
                animal_orders[animal_name] = animal_orders.get(animal_name, 0) + 1
                rem_budget -= (spec.purchase_cost + wheat_cost)
                spent_budget += (spec.purchase_cost + wheat_cost)

                if assigned_pos is not None:
                    build_orders.append((struct_kind, assigned_pos))

                # Limit to 1 animal purchase per day to avoid capital overextension
                break

        return animal_orders, build_orders, spent_budget

    def allocate_seed_budget(
        self,
        game_state: GameState,
        rois: Dict[str, float],
        available_money: float,
        planned_structures_count: int = 0,
    ) -> Tuple[Dict[str, int], List[str]]:
        """
        Allocates seed purchase orders to the highest-ROI crops, strictly bounded by:
        Max_Seeds = max(0, Empty_Unlocked_Tiles - Seeds_In_Stock - Planned_Structures)
        Prevents capital lock from buying excess seeds.
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
        # Strictly bound to real plantable capacity
        total_to_buy = max(0, empty_unlocked_tiles - existing_seeds - planned_structures_count)

        if total_to_buy <= 0:
            return {}, preferred_order

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
        Consolidates all macro-level financial and operational decisions for the day:
        - Crop & Animal dynamic ROIs
        - Land Expansion
        - Livestock purchases, structure construction, and wheat food safety buffer
        - Seed budget allocation & preferred planting order (strictly bounded)
        - Fertilizer target allocation matrix
        - Fibonacci labor scaling
        - Terminal harvest liquidation
        """
        current_day = game_state.day
        days_left = 30 - current_day
        my_farm = game_state.my_farm
        starting_cash = my_farm.money

        # 1. Compute dynamic ROIs
        crop_rois = self.compute_crop_rois(game_state, market_sim)
        animal_rois = self.compute_animal_rois(game_state, market_sim)
        best_crop_roi = max(crop_rois.values()) if crop_rois else 0.0

        # 2. Evaluate land expansion
        buy_land = self.evaluate_land_expansion(game_state, best_crop_roi)
        spent_on_land = QUADRANT_COSTS.get(buy_land, 0) if buy_land else 0.0
        cash_after_land = max(0.0, starting_cash - spent_on_land)

        # 3. Evaluate Livestock purchases & wheat reserve
        animal_orders, build_orders, spent_livestock = self.evaluate_livestock_plan(
            game_state, animal_rois, crop_rois, cash_after_land * 0.70, market_sim
        )
        prospective_animals = sum(animal_orders.values())
        wheat_deficit, wheat_cost = self.evaluate_wheat_reserve(
            game_state, market_sim, prospective_animals=prospective_animals
        )
        cash_after_livestock = max(0.0, cash_after_land - spent_livestock)

        # 4. Seed budget allocation (bounded by empty tiles minus planned structures)
        seed_budget = cash_after_livestock * 0.85
        seed_orders, preferred_seed_order = self.allocate_seed_budget(
            game_state, crop_rois, seed_budget, planned_structures_count=len(build_orders)
        )
        seed_spend = sum(seed_orders.get(c, 0) * CROP_SPECS[c].seed_cost for c in seed_orders)
        cash_after_seeds = max(0.0, cash_after_livestock - seed_spend)


        # 5. Evaluate labor hiring (Fibonacci scaling)
        target_plantings = sum(seed_orders.values()) + sum(my_farm.seeds.values())
        hands_to_hire = self.evaluate_labor_needs(game_state, target_plantings, cash_after_seeds)

        # 6. Fertilizer target prioritization
        fertilizer_targets = self.evaluate_fertilizer_shadow_matrix(game_state, market_sim)

        # 7. Liquidation planning for closing season (days 27-29)
        liquidation_orders: Dict[str, int] = {}
        if days_left <= 3:
            for prod_name, qty in my_farm.shed.items():
                if qty > 0 and prod_name not in ("GOOSE", "COW", "SHEEP"):
                    current_inv = game_state.market.get_inventory(prod_name)
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
            animal_orders=animal_orders,
            wheat_buy_orders=wheat_deficit,
            structure_build_orders=build_orders,
            fertilizer_target_tiles=fertilizer_targets,
            hands_to_hire=hands_to_hire,
            buy_land_quadrant=buy_land,
            preferred_seed_order=preferred_seed_order,
            total_budget_allocated=spent_on_land + spent_livestock + seed_spend,
            liquidation_orders=liquidation_orders,
            animal_rois=animal_rois,
        )

