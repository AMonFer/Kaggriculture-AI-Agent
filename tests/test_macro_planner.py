"""
Unit tests for Capa 2 MacroPlanner: Dynamic Crop ROI, Fibonacci Labor Scaling,
Land Expansion Amortization, and Terminal Horizon Harvest Cutoffs.
"""

import math
import pytest
from models.constants import (
    BOARD_SIZE,
    CROP_SPECS,
    ANIMAL_SPECS,
    CropType,
    AnimalType,
    ProductType,
    StructureType,
)
from models.state_representation import (
    EmptyTile,
    FarmState,
    GameState,
    MarketState,
    PlantTile,
    StructureTile,
    TownState,
    UnitState,
    WeedTile,
)
from engine.market_simulator import MarketSimulator
from engine.macro_planner import MacroPlanner, MacroPlan



@pytest.fixture
def planner() -> MacroPlanner:
    return MacroPlanner()


@pytest.fixture
def sim() -> MarketSimulator:
    return MarketSimulator()


def create_mock_game_state(
    day: int = 0,
    money: float = 3000.0,
    unlocked_quadrants: set = None,
    plant_count: int = 0,
    melon_market_inv: int = 10000,
) -> GameState:
    """Helper to construct configurable GameState for unit testing."""
    unlocked = unlocked_quadrants or {"NW"}
    tiles = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            in_nw = (r < 5 and c < 5)
            in_ne = (r < 5 and c >= 5)
            in_sw = (r >= 5 and c < 5)
            in_se = (r >= 5 and c >= 5)

            is_unlocked = (
                ("NW" in unlocked and in_nw) or
                ("NE" in unlocked and in_ne) or
                ("SW" in unlocked and in_sw) or
                ("SE" in unlocked and in_se)
            )
            if not is_unlocked:
                tiles[r][c] = "LOCKED"

    # Populate plants in NW
    placed = 0
    for r in range(5):
        for c in range(5):
            if placed < plant_count:
                tiles[r][c] = {
                    "kind": "PLANT",
                    "crop": "WHEAT",
                    "planted_day": day,
                    "watered_today": False,
                    "consecutive_unwatered": 0,
                    "yield_units": 0,
                    "max_lifespan_step": -1,
                    "fertilized_until_day": -1,
                }
                placed += 1

    market_inventories = {p.value: 10000 for p in ProductType}
    market_inventories["MELON"] = melon_market_inv

    raw_obs = {
        "player": 0,
        "day": day,
        "hour": 0,
        "market": {
            "inventory": market_inventories,
            "prices": {p.value: 50 for p in ProductType},
        },
        "town": {"unlocked_shops": []},
        "farms": [
            {
                "money": money,
                "tiles": tiles,
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": list(unlocked),
                "hires_today": 0,
            },
            {
                "money": money,
                "tiles": tiles,
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": list(unlocked),
                "hires_today": 0,
            },
        ],
        "private": {
            "shed": {},
            "seeds": {},
            "inventories": [{}],
        },
    }
    return GameState.from_raw_obs(raw_obs)


class TestCropROIsAndTerminalCutoffs:
    """Validates dynamic crop ROI computation and strict terminal horizon vetos."""

    def test_day_0_all_crops_viable(self, planner: MacroPlanner, sim: MarketSimulator):
        state = create_mock_game_state(day=0)
        rois = planner.compute_crop_rois(state, sim)

        # On day 0 (30 days left), all crops should have positive ROI
        assert rois["MELON"] > 0
        assert rois["CARROT"] > 0
        assert rois["WHEAT"] > 0
        assert rois["TOMATO"] > 0
        assert rois["STRAWBERRY"] > 0

    def test_day_21_melon_cutoff(self, planner: MacroPlanner, sim: MarketSimulator):
        state = create_mock_game_state(day=21)  # days_left = 9 < 10
        rois = planner.compute_crop_rois(state, sim)

        # Melon needs 10 days -> must be -inf at day 21
        assert math.isinf(rois["MELON"]) and rois["MELON"] < 0
        # Tomato and Strawberry also need >= 8 and >= 10 days -> Tomato has 1 harvest, Strawberry -inf
        assert math.isinf(rois["STRAWBERRY"]) and rois["STRAWBERRY"] < 0
        # Fast crops (Carrot and Wheat) must still be viable
        assert rois["CARROT"] > 0
        assert rois["WHEAT"] > 0

    def test_day_27_fast_crops_only(self, planner: MacroPlanner, sim: MarketSimulator):
        state = create_mock_game_state(day=27)  # days_left = 3
        rois = planner.compute_crop_rois(state, sim)

        assert math.isinf(rois["MELON"])
        assert math.isinf(rois["STRAWBERRY"])
        assert math.isinf(rois["TOMATO"])
        # Carrot (3 days) and Wheat (4 days) can harvest
        assert rois["CARROT"] > 0
        assert rois["WHEAT"] > 0

    def test_day_29_30_zero_plantings(self, planner: MacroPlanner, sim: MarketSimulator):
        state_29 = create_mock_game_state(day=29)  # days_left = 1
        rois_29 = planner.compute_crop_rois(state_29, sim)

        for crop, roi in rois_29.items():
            assert math.isinf(roi) and roi < 0

        plan_29 = planner.generate_daily_macro_plan(state_29, sim)
        assert len(plan_29.seed_orders) == 0


class TestLaborScalingFibonacci:
    """Validates hiring optimization based on workload and Fibonacci costs."""

    def test_low_workload_no_hands(self, planner: MacroPlanner):
        state = create_mock_game_state(day=5, plant_count=4)
        hands = planner.evaluate_labor_needs(state, target_plantings=0, available_budget=500.0)
        assert hands == 0

    def test_high_workload_hires_hands(self, planner: MacroPlanner):
        # 35 plants on farm + 15 planned plantings = high workload (50+ actions)
        state = create_mock_game_state(day=5, plant_count=35, unlocked_quadrants={"NW", "NE"})
        hands = planner.evaluate_labor_needs(state, target_plantings=15, available_budget=500.0)
        assert hands >= 2

    def test_budget_limited_hiring(self, planner: MacroPlanner):
        # High workload, but only $2 budget -> can only afford 2 hands ($1 + $1 = $2)
        state = create_mock_game_state(day=5, plant_count=24)
        hands = planner.evaluate_labor_needs(state, target_plantings=10, available_budget=2.0)
        assert hands == 2


class TestLandExpansionAmortization:
    """Validates quadrant acquisition decisions."""

    def test_approve_land_expansion_early_season(self, planner: MacroPlanner):
        # Day 2 with $1,800 cash, 16 plants in NW quadrant (64% density)
        state = create_mock_game_state(day=2, money=1800.0, plant_count=16)
        quad = planner.evaluate_land_expansion(state, best_daily_roi=15.0)
        assert quad == "NE"

    def test_reject_land_expansion_late_season(self, planner: MacroPlanner):
        # Day 25 with $3,000 cash, but only 5 days left -> cannot amortize $1,000
        state = create_mock_game_state(day=25, money=3000.0, plant_count=20)
        quad = planner.evaluate_land_expansion(state, best_daily_roi=15.0)
        assert quad is None


class TestPriceElasticityPortfolioShift:
    """Validates that when a crop's price crashes, the planner shifts capital to other crops."""

    def test_melon_glut_shifts_to_carrot(self, planner: MacroPlanner, sim: MarketSimulator):
        # Normal market: Melon has top ROI
        state_normal = create_mock_game_state(day=0, melon_market_inv=10000)
        rois_normal = planner.compute_crop_rois(state_normal, sim)
        assert rois_normal["MELON"] > rois_normal["CARROT"]

        # Glutted market: Melon inventory = 10,800 -> price collapses to $1
        state_glut = create_mock_game_state(day=0, melon_market_inv=10800)
        rois_glut = planner.compute_crop_rois(state_glut, sim)
        # Melon ROI becomes negative or very low, Carrot/Wheat take over
        assert rois_glut["CARROT"] > rois_glut["MELON"]
        plan_glut = planner.generate_daily_macro_plan(state_glut, sim)
        assert plan_glut.preferred_seed_order[0] != "MELON"


class TestLivestockROIsAndCutoffs:
    """Validates dynamic livestock ROI computation and strict terminal horizon cutoffs."""

    def test_animal_roi_day_0_positive(self, planner: MacroPlanner, sim: MarketSimulator):
        state = create_mock_game_state(day=0)
        rois = planner.compute_animal_rois(state, sim)

        assert rois["GOOSE"] > 0
        assert rois["COW"] > 0
        assert rois["SHEEP"] > 0

    def test_goose_cutoff_day_15(self, planner: MacroPlanner, sim: MarketSimulator):
        # Goose cutoff is day 14 -> at day 15 ROI must be -inf
        state = create_mock_game_state(day=15)
        rois = planner.compute_animal_rois(state, sim)
        assert math.isinf(rois["GOOSE"]) and rois["GOOSE"] < 0

    def test_cow_sheep_cutoff_day_11(self, planner: MacroPlanner, sim: MarketSimulator):
        # Cow and Sheep cutoff is day 10 -> at day 11 ROI must be -inf
        state = create_mock_game_state(day=11)
        rois = planner.compute_animal_rois(state, sim)
        assert math.isinf(rois["COW"]) and rois["COW"] < 0
        assert math.isinf(rois["SHEEP"]) and rois["SHEEP"] < 0

    def test_livestock_care_bonus_roi_projection(self, planner: MacroPlanner, sim: MarketSimulator):
        # Ensure ROI calculation accounts for (1 + interval) yields from daily CARE
        state = create_mock_game_state(day=0)
        rois = planner.compute_animal_rois(state, sim)
        # Goose yields 2 eggs per day with care, so expected revenue per day is ~2 * $50 = $100
        assert rois["GOOSE"] > 20.0


class TestWheatReservePolicy:
    """Validates 4-day wheat safety buffer, deficit calculation, and 16-unit shed cap."""

    def test_wheat_reserve_blocks_animal_purchase(self, planner: MacroPlanner, sim: MarketSimulator):
        # Money is only $310 -> Goose costs $300 + 4 wheat (~$100) -> not enough budget
        state = create_mock_game_state(day=0, money=310.0)
        animal_rois = planner.compute_animal_rois(state, sim)
        crop_rois = planner.compute_crop_rois(state, sim)

        orders, builds, spent = planner.evaluate_livestock_plan(state, animal_rois, crop_rois, 310.0, sim)
        assert len(orders) == 0

    def test_wheat_reserve_auto_buy_orders(self, planner: MacroPlanner, sim: MarketSimulator):
        # Farm has 1 living goose on farm, shed has 0 wheat -> deficit must be 4 wheat
        state = create_mock_game_state(day=1, money=2000.0)
        state.my_farm.tiles[0][0] = StructureTile(kind="COOP", animal="GOOSE")

        deficit, cost = planner.evaluate_wheat_reserve(state, sim, prospective_animals=0)
        assert deficit == 4
        assert cost > 0

    def test_wheat_reserve_cap_16_units(self, planner: MacroPlanner, sim: MarketSimulator):
        # Farm has 5 animals (would theoretically need 20 wheat), but cap is 16
        state = create_mock_game_state(day=1, money=3000.0)
        for i in range(5):
            state.my_farm.tiles[0][i] = StructureTile(kind="COOP", animal="GOOSE")

        deficit, cost = planner.evaluate_wheat_reserve(state, sim, prospective_animals=0)
        assert deficit == 16


class TestFertilizerShadowValueMatrix:
    """Validates shadow value prioritization: STRAWBERRY >= MELON > TOMATO > Spot."""

    def test_fertilizer_shadow_matrix_strawberry_melon_priority(self, planner: MacroPlanner, sim: MarketSimulator):
        state = create_mock_game_state(day=5)
        # Put 1 Strawberry at (0,0), 1 Melon at (1,1), 1 Tomato at (2,2), 1 Carrot at (3,3)
        state.my_farm.tiles[0][0] = PlantTile(crop="STRAWBERRY", planted_day=5, fertilized_until_day=-1)
        state.my_farm.tiles[1][1] = PlantTile(crop="MELON", planted_day=5, fertilized_until_day=-1)
        state.my_farm.tiles[2][2] = PlantTile(crop="TOMATO", planted_day=5, fertilized_until_day=-1)
        state.my_farm.tiles[3][3] = PlantTile(crop="CARROT", planted_day=5, fertilized_until_day=-1)

        targets = planner.evaluate_fertilizer_shadow_matrix(state, sim)
        # High value crops (Strawberry, Melon, Tomato) should be prioritized before Carrot
        assert (0, 0) in targets or (1, 1) in targets
        assert targets[0] in [(0, 0), (1, 1)]

    def test_fertilizer_shadow_matrix_skips_already_fertilized(self, planner: MacroPlanner, sim: MarketSimulator):
        state = create_mock_game_state(day=5)
        # Strawberry already fertilized until day 7
        state.my_farm.tiles[0][0] = PlantTile(crop="STRAWBERRY", planted_day=5, fertilized_until_day=7)
        targets = planner.evaluate_fertilizer_shadow_matrix(state, sim)
        assert (0, 0) not in targets


class TestIntegratedLivestockMacroPlan:
    """Validates that daily macro plan schedules livestock and structure orders cohesively."""

    def test_macro_plan_generates_animal_and_structure_orders(self, planner: MacroPlanner, sim: MarketSimulator):
        # Day 4 with $3000 cash and 2 unlocked quadrants -> should schedule animal purchase and build structure
        state = create_mock_game_state(day=4, money=3000.0, unlocked_quadrants={"NW", "NE"})
        plan = planner.generate_daily_macro_plan(state, sim)

        assert "GOOSE" in plan.animal_orders or "COW" in plan.animal_orders or "SHEEP" in plan.animal_orders
        assert len(plan.structure_build_orders) > 0 or len(state.my_farm.get_empty_structures()) > 0


class TestModuleCNewMacroPolicies:
    """Validates Module C: Livestock Gate D0-D3, Early NE Rush, Fibonacci Scaling, and Strict Seed Caps."""

    def test_livestock_gate_low_budget_blocked(self, planner: MacroPlanner, sim: MarketSimulator):
        """When money is below $1,500, livestock purchase is blocked to protect operational cash."""
        state_poor = create_mock_game_state(day=2, money=1200.0)
        plan_poor = planner.generate_daily_macro_plan(state_poor, sim)
        assert len(plan_poor.animal_orders) == 0
        assert len(plan_poor.structure_build_orders) == 0

    def test_livestock_gate_day_0_synergy(self, planner: MacroPlanner, sim: MarketSimulator):
        """On Day 0 with $3,000 starting cash, schedules early cow/goose for daily fertilizer & milk."""
        state_rich = create_mock_game_state(day=0, money=3000.0)
        plan = planner.generate_daily_macro_plan(state_rich, sim)
        assert "COW" in plan.animal_orders or "GOOSE" in plan.animal_orders
        assert len(plan.structure_build_orders) > 0

    def test_livestock_gate_day_4_requires_1500_coins(self, planner: MacroPlanner, sim: MarketSimulator):
        """On Day 4+, requires at least $1,500 in bank to prevent quadrant expansion cannibalization."""
        # Low money ($1,200) -> Gate remains closed
        state_poor = create_mock_game_state(day=4, money=1200.0, unlocked_quadrants={"NW", "NE"})
        plan_poor = planner.generate_daily_macro_plan(state_poor, sim)
        assert len(plan_poor.animal_orders) == 0

    def test_livestock_gate_day_4_allows_with_50_tiles(self, planner: MacroPlanner, sim: MarketSimulator):
        """On Day 4+, with $2,000 cash and >= 50 tiles, Gate opens for livestock."""
        state_rich = create_mock_game_state(day=4, money=2000.0, unlocked_quadrants={"NW", "NE"})
        plan_rich = planner.generate_daily_macro_plan(state_rich, sim)
        assert len(plan_rich.animal_orders) > 0

    def test_land_expansion_ne_rush_day_3_to_6(self, planner: MacroPlanner):
        """Early NE Rush: $1,000 cost + $500 carrot seed cushion + $100 buffer = $1,600."""
        # Day 3 with $1,650 cash and 16 plants in NW -> triggers NE expansion
        state_solvent = create_mock_game_state(day=3, money=1650.0, plant_count=16)
        quad = planner.evaluate_land_expansion(state_solvent, best_daily_roi=15.0)
        assert quad == "NE"

        # Day 3 with $1,400 cash -> not enough for expansion + seed cushion -> returns None
        state_insolvent = create_mock_game_state(day=3, money=1400.0, plant_count=16)
        assert planner.evaluate_land_expansion(state_insolvent, best_daily_roi=15.0) is None

    def test_early_ne_rush_with_stock_seeds_discount(self, planner: MacroPlanner):
        """If farm already has 25 seeds in stock, seed cushion is $0 -> NE unlocks at $1,100."""
        state_with_seeds = create_mock_game_state(day=3, money=1150.0, plant_count=16)
        state_with_seeds.my_farm.seeds = {"CARROT": 25}
        quad = planner.evaluate_land_expansion(state_with_seeds, best_daily_roi=15.0)
        assert quad == "NE"

    def test_land_expansion_sw_day_8_to_12(self, planner: MacroPlanner):
        """SW expansion ($2,000) evaluated in Days 8-12."""
        state = create_mock_game_state(day=8, money=2700.0, plant_count=25, unlocked_quadrants={"NW", "NE"})
        quad = planner.evaluate_land_expansion(state, best_daily_roi=15.0)
        assert quad == "SW"

    def test_land_expansion_se_roi_hurdle_rate(self, planner: MacroPlanner):
        """SE expansion ($4,000) requires expected profit hurdle > $500."""
        # High ROI (20.0) with 16 days left -> 25 * 15 * 20 * 0.85 - 4000 = 2375 > 500 -> Approved
        state_high_roi = create_mock_game_state(day=14, money=4700.0, plant_count=40, unlocked_quadrants={"NW", "NE", "SW"})
        assert planner.evaluate_land_expansion(state_high_roi, best_daily_roi=20.0) == "SE"

        # Very low ROI (1.0) -> Rejected
        assert planner.evaluate_land_expansion(state_high_roi, best_daily_roi=1.0) is None

    def test_labor_scaling_proportional_to_quadrants(self, planner: MacroPlanner):
        """Validates dynamic labor caps: 1 quad <= 3 hands; 2 quads <= 6 hands."""
        state_1q = create_mock_game_state(day=5, plant_count=24, unlocked_quadrants={"NW"})
        hands_1q = planner.evaluate_labor_needs(state_1q, target_plantings=20, available_budget=1000.0)
        assert hands_1q <= 3

        state_2q = create_mock_game_state(day=5, plant_count=48, unlocked_quadrants={"NW", "NE"})
        hands_2q = planner.evaluate_labor_needs(state_2q, target_plantings=20, available_budget=1000.0)
        assert hands_2q <= 6

    def test_strict_seed_budget_cap_prevents_excess_buying(self, planner: MacroPlanner, sim: MarketSimulator):
        """Seed orders must never exceed empty unlocked tiles minus seeds in stock."""
        # 16 plants in NW (leaving 9 empty tiles). 4 seeds in stock.
        state = create_mock_game_state(day=5, money=3000.0, plant_count=16, unlocked_quadrants={"NW"})
        state.my_farm.seeds = {"MELON": 4}

        rois = planner.compute_crop_rois(state, sim)
        seed_orders, _ = planner.allocate_seed_budget(state, rois, available_money=3000.0)

        # Max to buy = 9 empty - 4 seeds in stock = 5 seeds
        total_ordered = sum(seed_orders.values())
        assert total_ordered <= 5


