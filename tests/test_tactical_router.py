"""
Unit tests for Capa 3 TacticalRouter: A* navigation, priority-based task scheduling (P0-P3),
and multi-agent worker coordination on the 10x10 farm grid.
"""

import pytest
from models.constants import (
    BOARD_SIZE,
    CropType,
    Direction,
    FarmerAction,
    SHED_ACCESS_TILES,
)
from models.state_representation import (
    EmptyTile,
    FarmState,
    GameState,
    LockedTile,
    MarketState,
    PlantTile,
    StructureTile,
    TownState,
    UnitState,
    WeedTile,
)
from engine.tactical_router import TacticalRouter, Task


@pytest.fixture
def router() -> TacticalRouter:
    return TacticalRouter()


@pytest.fixture
def empty_game_state() -> GameState:
    """Builds a basic GameState with an empty 10x10 farm (NW quadrant unlocked)."""
    tiles = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    # Default: NW unlocked (0..4, 0..4), rest locked
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not (r < 5 and c < 5):
                tiles[r][c] = "LOCKED"

    raw_obs = {
        "player": 0,
        "day": 1,
        "hour": 0,
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "farms": [
            {
                "money": 1000.0,
                "tiles": tiles,
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
            {
                "money": 1000.0,
                "tiles": tiles,
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
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


class TestNavigationAndPathfinding:
    """Validates A* paths and direction calculations."""

    def test_manhattan_distance(self, router: TacticalRouter):
        assert router.manhattan_distance((0, 0), (0, 0)) == 0
        assert router.manhattan_distance((0, 0), (3, 4)) == 7
        assert router.manhattan_distance((9, 9), (0, 0)) == 18

    def test_find_path_straight_lines(self, router: TacticalRouter):
        # East
        path_east = router.find_path((1, 1), (4, 1))
        assert path_east == ["EAST", "EAST", "EAST"]
        assert len(path_east) == 3

        # North
        path_north = router.find_path((2, 5), (2, 2))
        assert path_north == ["NORTH", "NORTH", "NORTH"]
        assert len(path_north) == 3

    def test_find_path_diagonal_and_locked_transit(self, router: TacticalRouter):
        # From NW quadrant (0,0) to SE quadrant (9,9) crossing locked land
        path = router.find_path((0, 0), (9, 9))
        assert len(path) == 18
        assert path.count("EAST") == 9
        assert path.count("SOUTH") == 9

    def test_get_direction_to(self, router: TacticalRouter):
        assert router.get_direction_to((2, 2), (2, 2)) is None
        assert router.get_direction_to((2, 2), (5, 2)) == "EAST"
        assert router.get_direction_to((5, 2), (2, 2)) == "WEST"
        assert router.get_direction_to((2, 2), (2, 5)) == "SOUTH"
        assert router.get_direction_to((2, 5), (2, 2)) == "NORTH"

    def test_nearest_shed_tile_selection(self, router: TacticalRouter):
        # NW quadrant corner -> (4,4)
        assert router.get_nearest_shed_tile((0, 0)) == (4, 4)
        # NE quadrant corner -> (5,4)
        assert router.get_nearest_shed_tile((9, 0)) == (5, 4)
        # SW quadrant corner -> (4,5)
        assert router.get_nearest_shed_tile((0, 9)) == (4, 5)
        # SE quadrant corner -> (5,5)
        assert router.get_nearest_shed_tile((9, 9)) == (5, 5)


class TestTaskGenerationAndPriorities:
    """Validates classification of farm tasks into P0, P1, P2, P3."""

    def test_task_priority_hierarchy(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm

        # Set up a P0 critical plant (consecutive_unwatered = 1)
        farm.tiles[0][0] = PlantTile(
            crop="WHEAT",
            consecutive_unwatered=1,
            watered_today=False,
            yield_units=0,
        )

        # Set up a P1 harvestable plant
        farm.tiles[1][1] = PlantTile(
            crop="CARROT",
            consecutive_unwatered=0,
            watered_today=True,
            yield_units=3,
        )

        # Set up a P2 regular unwatered plant
        farm.tiles[2][2] = PlantTile(
            crop="TOMATO",
            consecutive_unwatered=0,
            watered_today=False,
            yield_units=0,
        )

        # Set up a P3 weed
        farm.tiles[3][3] = WeedTile()

        tasks = router.generate_daily_tasks(empty_game_state)
        assert len(tasks) == 4

        # Tasks must be strictly sorted by priority (0 -> 1 -> 2 -> 3)
        assert tasks[0].priority == 0
        assert tasks[0].task_type == FarmerAction.WATER
        assert tasks[0].target_pos == (0, 0)

        assert tasks[1].priority == 1
        assert tasks[1].task_type == FarmerAction.HARVEST
        assert tasks[1].target_pos == (1, 1)

        assert tasks[2].priority == 2
        assert tasks[2].task_type == FarmerAction.WATER
        assert tasks[2].target_pos == (2, 2)

        assert tasks[3].priority == 3
        assert tasks[3].task_type == FarmerAction.DIG
        assert tasks[3].target_pos == (3, 3)

    def test_animal_danger_p0_priority(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        farm.tiles[0][1] = StructureTile(
            kind="COOP",
            animal="GOOSE",
            consecutive_unfed=1,
            fed_today=False,
        )

        tasks = router.generate_daily_tasks(empty_game_state)
        assert len(tasks) == 1
        assert tasks[0].priority == 0
        assert tasks[0].task_type == FarmerAction.FEED
        assert tasks[0].target_pos == (1, 0)


class TestMultiAgentActionAssignment:
    """Validates multi-unit coordination, seed stock decrement, and backpack drops."""

    def test_multi_agent_disjoint_assignment(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        # 1 Farmer at (0,0), 2 Hands at (0,1) and (0,2)
        farm.farmer = UnitState(id=0, x=0, y=0)
        farm.hands = [
            UnitState(id=1, x=0, y=1),
            UnitState(id=2, x=0, y=2),
        ]

        # 3 unwatered plants at (1,0), (1,1), (1,2)
        farm.tiles[0][1] = PlantTile(crop="WHEAT", consecutive_unwatered=0, watered_today=False)
        farm.tiles[1][1] = PlantTile(crop="WHEAT", consecutive_unwatered=0, watered_today=False)
        farm.tiles[2][1] = PlantTile(crop="WHEAT", consecutive_unwatered=0, watered_today=False)

        actions = router.assign_actions(empty_game_state)

        # Farmer at (0,0) moves EAST to (1,0)
        assert actions["farmer"] == ["EAST"]
        # Hand 1 at (0,1) moves EAST to (1,1)
        assert actions["hands"][0] == ["EAST"]
        # Hand 2 at (0,2) moves EAST to (1,2)
        assert actions["hands"][1] == ["EAST"]

    def test_seed_stock_decrement_protection(self, router: TacticalRouter, empty_game_state: GameState):
        """If only 1 seed is in stock, only 1 worker gets a PLANT action across multiple empty tiles."""
        farm = empty_game_state.my_farm
        farm.farmer = UnitState(id=0, x=0, y=0)
        farm.hands = [UnitState(id=1, x=0, y=1)]
        farm.seeds = {"WHEAT": 1}  # Only 1 seed!

        # Set 2 empty tiles
        farm.tiles[0][0] = EmptyTile()
        farm.tiles[1][0] = EmptyTile()

        actions = router.assign_actions(empty_game_state)

        # Farmer at (0,0) is directly on empty tile and should plant
        assert actions["farmer"] == ["PLANT", "WHEAT"]
        # Hand 1 at (0,1) cannot plant because seed stock is exhausted!
        assert actions["hands"][0] != ["PLANT", "WHEAT"]

    def test_backpack_drop_at_shed(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        # Farmer at (4,4) [shed adjacent] carrying 5 wheat
        farm.farmer = UnitState(id=0, x=4, y=4, inventory={"WHEAT": 5})
        # Hand 1 at (0,0) carrying 2 carrots
        farm.hands = [UnitState(id=1, x=0, y=0, inventory={"CARROT": 2})]

        actions = router.assign_actions(empty_game_state)

        # Farmer is at shed -> drops inventory
        assert actions["farmer"] == ["DROP"]
        # Hand 1 is far from shed -> moves towards (4,4)
        assert actions["hands"][0] in (["EAST"], ["SOUTH"])
