"""
Unit tests for Capa 2 MacroPlanner: Dynamic Crop ROI, Fibonacci Labor Scaling,
Land Expansion Amortization, and Terminal Horizon Harvest Cutoffs.
"""

import math
import pytest
from models.constants import (
    BOARD_SIZE,
    CROP_SPECS,
    CropType,
    ProductType,
)
from models.state_representation import (
    EmptyTile,
    FarmState,
    GameState,
    MarketState,
    PlantTile,
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
