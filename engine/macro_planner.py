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
        Evaluates whether buying the next quadrant (NE, SW, SE) is financially amortizable:
        - NE ($1,000): Early Rush in Days 3-6 (or earlier if solvent). Seed buffer calculated with base crop (Carrot $20 * 25 = $500).
        - SW ($2,000): Days 8-12 when solvent.
        - SE ($4,000): Days 13-18 when amortizable expected return exceeds hurdle rate ($500).
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
        existing_seeds = sum(my_farm.seeds.values())
        seeds_needed = max(0, 25 - existing_seeds)

        # Early NE Rush: base seed cost is Carrot ($20) -> $500 cushion for 25 tiles
        seed_cushion = seeds_needed * 20.0

        if next_quad == "NE":
            # Early NE Rush window (Days 2-8 or whenever solvent)
            if days_left < 6 or best_daily_roi <= 1.0:
                return None
            # Need quadrant cost ($1,000) + seed cushion + buffer
            required_money = quad_cost + seed_cushion + 50.0
            if my_farm.money < required_money:
                return None

            # Check utilization of initial NW quadrant (at least 8 occupied/planted tiles or seeds)
            active_utilization = my_farm.get_plant_count() + len(my_farm.get_occupied_structures()) + existing_seeds
            if active_utilization < 6 and len(unlocked) == 1:
                return None

            return "NE"

        elif next_quad == "SW":
            # SW Window (Days 6-15 or whenever solvent)
            if days_left < 8 or best_daily_roi <= 1.0:
                return None
            required_money = quad_cost + seed_cushion + 50.0
            if my_farm.money < required_money:
                return None

            total_unlocked_tiles = len(unlocked) * 25
            active_utilization = my_farm.get_plant_count() + len(my_farm.get_occupied_structures()) + existing_seeds
            if (active_utilization / float(total_unlocked_tiles)) < 0.30:
                return None

            return "SW"

        elif next_quad == "SE":
            # SE Window (Days 10-20)
            if days_left < 10 or best_daily_roi <= 1.0:
                return None
            required_money = quad_cost + seed_cushion + 50.0
            if my_farm.money < required_money:
                return None

            # Hurdle rate: projected net profit on 25 tiles over remaining days > $300
            expected_profit_se = 25 * (days_left - 1) * best_daily_roi * 0.85 - 4000.0
            if expected_profit_se > 300.0:
                return "SE"

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
        Workload: In 2D grid, each hand realistically achieves ~8-10 effective field actions/day.
        """
        my_farm = game_state.my_farm
        current_day = game_state.day
        days_left = 30 - current_day

        # Count active farm workload
        n_water = 0
        n_harvest = 0
        n_weeds = 0
        for row in my_farm.tiles:
            for tile in row:
                if isinstance(tile, PlantTile):
                    if tile.is_in_danger or not tile.watered_today:
                        n_water += 1
                    if tile.is_mature(current_day):
                        n_harvest += 1
                elif isinstance(tile, StructureTile) and tile.is_occupied:
                    if tile.yield_units > 0:
                        n_harvest += 1
                    if tile.fertilizer_available:
                        n_harvest += 1
                elif isinstance(tile, WeedTile):
                    n_weeds += 1

        animal_count = my_farm.get_animal_count()
        animal_workload = animal_count * 4  # Feed + Care + Collect + Transit

        # Transit overhead: in 10x10 board, workers take ~2.5 moves per task
        action_units = n_water + (n_harvest * 1.5) + target_plantings + (n_weeds * 1.2) + animal_workload
        l_total = action_units * 2.2  # Total physical turns needed

        # Farmer provides 24 turns
        extra_turns_needed = max(0.0, l_total - 24.0)

        # Each hand provides 24 turns
        desired_hands = int(math.ceil(extra_turns_needed / 24.0))

        # Dynamic baseline scaling proportional to unlocked land
        n_quads = len(my_farm.unlocked_quadrants)
        # Baseline minimum hands if farm has crops: 1Q: 2, 2Q: 4, 3Q: 6, 4Q: 8
        if (n_water + target_plantings + n_harvest) >= 6:
            min_hands = min(n_quads * 2, 8)
            desired_hands = max(desired_hands, min_hands)

        max_hands_by_quads = {1: 3, 2: 6, 3: 8, 4: 12}
        desired_hands = min(desired_hands, max_hands_by_quads.get(n_quads, 8))

        # In final closing days (days 28-29), focus on harvest
        if days_left <= 2:
            desired_hands = min(desired_hands, int(math.ceil(n_harvest / 6.0)))

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
        - Early Day 0 synergy: allows 1-2 Cows/Geese to establish daily fertilizer & dairy cashflow.
        - Day 4+: scales livestock when solvent (money >= $1,500).
        - Prioritizes COW (Milk + Fertilizer) and GOOSE (Egg + Fertilizer).
        - Verifies wheat safety reserve budget before approving animal purchase.
        """
        my_farm = game_state.my_farm
        current_day = game_state.day

        animal_orders: Dict[str, int] = {}
        build_orders: List[Tuple[str, Tuple[int, int]]] = []
        spent_budget = 0.0

        # Capital gating for livestock:
        if current_day == 0:
            # On Day 0, allow early livestock if starting cash >= $2,000
            if my_farm.money < 2000.0 or available_budget < 350:
                return animal_orders, build_orders, spent_budget
        elif current_day in (1, 2, 3):
            # Days 1-3: allow animal if cash is healthy (>= $1,800)
            if my_farm.money < 1800.0 or available_budget < 350:
                return animal_orders, build_orders, spent_budget
        else:
            # Day 4+: Require bank money >= $1,500
            if my_farm.money < 1500.0 or available_budget < 350:
                return animal_orders, build_orders, spent_budget

        # Don't buy new animals in closing season
        if current_day > 18:
            return animal_orders, build_orders, spent_budget

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

        # Find empty tiles for building new structures (sorted near Shed for zero-overhead logistics)
        empty_tiles: List[Tuple[int, int]] = []
        for y, row in enumerate(my_farm.tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, EmptyTile):
                    empty_tiles.append((x, y))
        empty_tiles.sort(key=lambda pos: abs(pos[0] - 4) + abs(pos[1] - 4))

        # Animal counts currently on farm
        goose_count = my_farm.get_animal_count(AnimalType.GOOSE.value) + pending_goose_in_shed
        cow_count = my_farm.get_animal_count(AnimalType.COW.value) + pending_cow_in_shed
        sheep_count = my_farm.get_animal_count(AnimalType.SHEEP.value) + pending_sheep_in_shed

        # Max animal caps scaled by land
        n_quads = len(my_farm.unlocked_quadrants)
        max_cows = 3 if n_quads == 1 else (6 if n_quads == 2 else 12)
        max_geese = 2 if n_quads == 1 else (4 if n_quads == 2 else 6)
        max_sheep = 0 if n_quads == 1 else (2 if n_quads == 2 else 4)

        # Best crop ROI to compare with
        valid_crop_rois = [r for r in crop_rois.values() if r > 0 and not math.isinf(r)]
        best_crop_roi = max(valid_crop_rois) if valid_crop_rois else 5.0

        # Preference order: COW first (Milk $160 + Fertilizer $100 daily), then GOOSE, then SHEEP
        candidate_animals = [
            (AnimalType.COW.value, ANIMAL_SPECS[AnimalType.COW.value], net_empty_pastures, StructureType.PASTURE.value, cow_count, max_cows),
            (AnimalType.GOOSE.value, ANIMAL_SPECS[AnimalType.GOOSE.value], net_empty_coops, StructureType.COOP.value, goose_count, max_geese),
            (AnimalType.SHEEP.value, ANIMAL_SPECS[AnimalType.SHEEP.value], net_empty_pastures, StructureType.PASTURE.value, sheep_count, max_sheep),
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
        # Reserve a modest $30 buffer for Fibonacci labor hiring
        labor_reserve = 30.0 if cash_after_livestock > 50 else 0.0
        seed_budget = max(0.0, cash_after_livestock - labor_reserve)
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

