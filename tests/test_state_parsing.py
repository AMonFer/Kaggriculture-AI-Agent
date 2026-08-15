"""
Unit tests for models/state_representation.py and GameState deserialization performance.
"""

import time
import pytest
from models.constants import BOARD_SIZE, CropType
from models.state_representation import (
    GameState,
    PlantTile,
    StructureTile,
    WeedTile,
    EmptyTile,
    LockedTile,
)


@pytest.fixture
def sample_raw_observation() -> dict:
    """Constructs a realistic raw Kaggle observation dictionary."""
    # Build 10x10 tile matrix for player 0
    tiles_0 = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    # NW is unlocked, other 3 quadrants locked
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not (r < 5 and c < 5):
                tiles_0[r][c] = "LOCKED"

    # Place a couple plants and structures in NW
    tiles_0[0][0] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 1,
        "watered_today": True,
        "consecutive_unwatered": 0,
        "yield_units": 2,
        "max_lifespan_step": -1,
        "fertilized_until_day": 3,
    }
    tiles_0[1][1] = {
        "kind": "COOP",
        "animal": "GOOSE",
        "placed_day": 2,
        "yield_units": 1,
        "fed_today": True,
        "consecutive_unfed": 0,
        "cared_today": True,
        "fertilizer_available": True,
        "pending_care_bonus": 1,
    }
    tiles_0[2][2] = {"kind": "WEED"}

    tiles_1 = [["LOCKED" if not (r < 5 and c < 5) else None for c in range(10)] for r in range(10)]

    return {
        "player": 0,
        "day": 5,
        "hour": 12,
        "market": {
            "inventory": {"WHEAT": 9800, "CARROT": 10200, "TOMATO": 10000},
            "prices": {"WHEAT": 27, "CARROT": 30, "TOMATO": 60},
        },
        "town": {
            "unlocked_shops": ["BAKERY", "PIZZA_SHOP"],
        },
        "farms": [
            {
                "money": 2450.0,
                "tiles": tiles_0,
                "farmer": [4, 4],
                "hands": [[4, 3], [3, 4]],
                "unlocked_quadrants": ["NW"],
                "hires_today": 2,
            },
            {
                "money": 3100.0,
                "tiles": tiles_1,
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
        ],
        "private": {
            "shed": {"WHEAT": 20, "EGG": 5},
            "seeds": {"WHEAT": 10, "MELON": 2},
            "inventories": [
                {"WHEAT": 3},  # Farmer
                {"EGG": 1},    # Hand 1
                {},            # Hand 2
            ],
        },
    }


def test_game_state_parsing_accuracy(sample_raw_observation):
    game_state = GameState.from_raw_obs(sample_raw_observation)

    assert game_state.day == 5
    assert game_state.hour == 12
    assert game_state.global_turn == 5 * 24 + 12
    assert game_state.my_player_id == 0

    my_farm = game_state.my_farm
    assert my_farm.money == 2450.0
    assert my_farm.unlocked_quadrants == {"NW"}
    assert my_farm.hires_today == 2
    assert my_farm.worker_count == 3  # 1 farmer + 2 hands
    assert my_farm.farmer.pos == (4, 4)
    assert my_farm.farmer.inventory == {"WHEAT": 3}
    assert len(my_farm.hands) == 2
    assert my_farm.hands[0].pos == (4, 3)
    assert my_farm.hands[0].inventory == {"EGG": 1}

    # Tiles check
    plant_tile = my_farm.get_tile(0, 0)
    assert isinstance(plant_tile, PlantTile)
    assert plant_tile.crop == "WHEAT"
    assert plant_tile.yield_units == 2
    assert plant_tile.is_fertilized is True
    assert plant_tile.is_ready_to_harvest is True

    coop_tile = my_farm.get_tile(1, 1)
    assert isinstance(coop_tile, StructureTile)
    assert coop_tile.kind == "COOP"
    assert coop_tile.animal == "GOOSE"
    assert coop_tile.is_occupied is True

    weed_tile = my_farm.get_tile(2, 2)
    assert isinstance(weed_tile, WeedTile)

    empty_tile = my_farm.get_tile(0, 1)
    assert isinstance(empty_tile, EmptyTile)

    locked_tile = my_farm.get_tile(7, 7)
    assert isinstance(locked_tile, LockedTile)

    # Shed check
    assert my_farm.shed_items_count == 25  # 20 wheat + 5 eggs (seeds excluded)
    assert my_farm.shed_free_capacity == 75
    assert my_farm.shed_occupancy_ratio == 0.25
    assert my_farm.is_shed_critical() is False


def test_game_state_parsing_performance(sample_raw_observation):
    """Ensures parsing 1000 turns takes under 1 second total (< 1 ms per turn)."""
    start = time.perf_counter()
    iterations = 1000
    for _ in range(iterations):
        _ = GameState.from_raw_obs(sample_raw_observation)
    elapsed = time.perf_counter() - start

    ms_per_call = (elapsed / iterations) * 1000.0
    print(f"\nGameState.from_raw_obs performance: {ms_per_call:.3f} ms/call")
    assert ms_per_call < 2.0, f"Parsing too slow: {ms_per_call:.3f} ms per call (must be < 2ms)"
