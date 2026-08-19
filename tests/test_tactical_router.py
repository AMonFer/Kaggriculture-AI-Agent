"""
Unit tests for Capa 3 TacticalRouter: A* navigation, priority-based task scheduling (P0-P3),
and multi-agent worker coordination on the 10x10 farm grid.
"""

import pytest
from models.constants import (
    BOARD_SIZE,
    CropType,
    AnimalType,
    ProductType,
    StructureType,
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

        # Set up a P1 harvestable plant (mature carrot)
        farm.tiles[1][1] = PlantTile(
            crop="CARROT",
            planted_day=-2,
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


class TestLivestockAndFertilizerTaskGeneration:
    """Validates prioritized task generation for livestock production, structures, and fertilization."""

    def test_p1_animal_harvest_task(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        farm.tiles[0][0] = StructureTile(kind="COOP", animal="GOOSE", yield_units=2, fed_today=True)
        tasks = router.generate_daily_tasks(empty_game_state)

        harvest_tasks = [t for t in tasks if t.task_type == FarmerAction.HARVEST and t.target_pos == (0, 0)]
        assert len(harvest_tasks) == 1
        assert harvest_tasks[0].priority == 1

    def test_p1_collect_fertilizer_task(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        farm.tiles[0][0] = StructureTile(kind="COOP", animal="GOOSE", fertilizer_available=True)
        tasks = router.generate_daily_tasks(empty_game_state)

        fert_tasks = [t for t in tasks if t.task_type == FarmerAction.COLLECT_FERTILIZER and t.target_pos == (0, 0)]
        assert len(fert_tasks) == 1
        assert fert_tasks[0].priority == 1

    def test_p2_preventive_feed_and_care_tasks(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        # Unfed animal -> generates FEED task
        farm.tiles[1][1] = StructureTile(kind="PASTURE", animal="COW", fed_today=False, cared_today=False)
        tasks_unfed = router.generate_daily_tasks(empty_game_state)
        feed_tasks = [t for t in tasks_unfed if t.task_type == FarmerAction.FEED and t.target_pos == (1, 1)]
        assert len(feed_tasks) == 1 and feed_tasks[0].priority == 2

        # Once fed -> generates CARE task
        farm.tiles[1][1].fed_today = True
        tasks_fed = router.generate_daily_tasks(empty_game_state)
        care_tasks = [t for t in tasks_fed if t.task_type == FarmerAction.CARE and t.target_pos == (1, 1)]
        assert len(care_tasks) == 1 and care_tasks[0].priority == 2


    def test_p2_fertilize_application_task(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        farm.tiles[2][2] = PlantTile(crop="MELON", planted_day=1, fertilized_until_day=-1, watered_today=True)
        tasks = router.generate_daily_tasks(empty_game_state, fertilizer_targets=[(2, 2)])

        fert_tasks = [t for t in tasks if t.task_type == FarmerAction.FERTILIZE and t.target_pos == (2, 2)]
        assert len(fert_tasks) == 1
        assert fert_tasks[0].priority == 2

    def test_p3_build_structure_tasks(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        farm.tiles[3][3] = EmptyTile()
        tasks = router.generate_daily_tasks(empty_game_state, structure_build_orders=[("COOP", (3, 3))])

        build_tasks = [t for t in tasks if t.task_type == FarmerAction.BUILD_COOP and t.target_pos == (3, 3)]
        assert len(build_tasks) == 1
        assert build_tasks[0].priority == 2


class TestLivestockAndFertilizerLogistics:
    """Validates multi-agent coordination, pickup/place logistics, and conflict avoidance."""

    def test_animal_pickup_from_shed(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        farm.farmer = UnitState(id=0, x=4, y=4)  # Adjacent to shed
        farm.shed = {"GOOSE": 1}
        # Empty coop at (0, 0)
        farm.tiles[0][0] = StructureTile(kind="COOP", animal=None)

        actions = router.assign_actions(empty_game_state)
        # Farmer at shed interacts with shed -> PICKUP GOOSE 1
        assert actions["farmer"] == ["PICKUP", "GOOSE", 1]

    def test_animal_placement_in_structure(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        # Farmer carrying a Goose is at (1, 1), empty coop is at (1, 1)
        farm.farmer = UnitState(id=0, x=1, y=1, inventory={"GOOSE": 1})
        farm.tiles[1][1] = StructureTile(kind="COOP", animal=None)

        actions = router.assign_actions(empty_game_state)
        assert actions["farmer"] == ["PLACE", "GOOSE", 1]

    def test_fertilizer_pickup_from_shed(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        farm.farmer = UnitState(id=0, x=4, y=4)
        farm.shed = {"FERTILIZER": 2}
        farm.tiles[0][0] = PlantTile(crop="MELON", planted_day=1, fertilized_until_day=-1)

        actions = router.assign_actions(empty_game_state, fertilizer_target_tiles=[(0, 0)])
        assert actions["farmer"] == ["PICKUP", "FERTILIZER", 1]

    def test_feed_care_deduplication_multi_agent(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        # 2 workers near 1 cow
        farm.farmer = UnitState(id=0, x=0, y=0)
        farm.hands = [UnitState(id=1, x=0, y=1)]
        farm.tiles[0][0] = StructureTile(kind="PASTURE", animal="COW", fed_today=False, cared_today=False)

        actions = router.assign_actions(empty_game_state)
        # One worker feeds, the other either cares or does something else, but they don't duplicate FEED
        feed_count = sum(1 for act in [actions["farmer"], actions["hands"][0]] if act == ["FEED"])
        assert feed_count <= 1

    def test_fertilize_does_not_repeat_on_fertilized_crop(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        farm.farmer = UnitState(id=0, x=1, y=1, inventory={"FERTILIZER": 1})
        # Plant at (1,1) is already fertilized until day 5
        farm.tiles[1][1] = PlantTile(crop="MELON", planted_day=1, fertilized_until_day=5)

        actions = router.assign_actions(empty_game_state, fertilizer_target_tiles=[(1, 1)])
        # Cannot fertilize the already fertilized plant
        assert actions["farmer"] != ["FERTILIZE"]


class TestSpatialClusteringAndLocalAffinity:
    """Validates SpatialClustering 4-zone partition, Local Affinity matching, and Boundary Crossing."""

    def test_spatial_clustering_quadrant_partition(self, router: TacticalRouter):
        assert router.get_tile_quadrant((0, 0)) == "NW"
        assert router.get_tile_quadrant((4, 4)) == "NW"
        assert router.get_tile_quadrant((5, 0)) == "NE"
        assert router.get_tile_quadrant((9, 4)) == "NE"
        assert router.get_tile_quadrant((0, 5)) == "SW"
        assert router.get_tile_quadrant((4, 9)) == "SW"
        assert router.get_tile_quadrant((5, 5)) == "SE"
        assert router.get_tile_quadrant((9, 9)) == "SE"

    def test_local_affinity_worker_stays_in_assigned_quadrant(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        farm.unlocked_quadrants = {"NW", "NE"}

        # Farmer at (2,2) in NW, Hand at (7,2) in NE
        farm.farmer = UnitState(id=0, x=2, y=2)
        farm.hands = [UnitState(id=1, x=7, y=2)]

        # Plant in NW at (2,3) needing water, Plant in NE at (7,3) needing water
        farm.tiles[3][2] = PlantTile(crop="WHEAT", watered_today=False)
        farm.tiles[3][7] = PlantTile(crop="WHEAT", watered_today=False)

        actions = router.assign_actions(empty_game_state)

        # Farmer should take the local NW task (moving SOUTH to (2,3))
        assert actions["farmer"] == ["SOUTH"]
        # Hand should take the local NE task (moving SOUTH to (7,3))
        assert actions["hands"][0] == ["SOUTH"]

    def test_boundary_crossing_when_local_quadrant_clear(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        farm.unlocked_quadrants = {"NW", "NE"}

        # 2 workers located in NW: Farmer at (2,2), Hand at (2,3)
        farm.farmer = UnitState(id=0, x=2, y=2)
        farm.hands = [UnitState(id=1, x=2, y=3)]

        # NW has 0 tasks. NE has 2 plants needing water at (7,2) and (7,3)
        farm.tiles[2][7] = PlantTile(crop="WHEAT", watered_today=False)
        farm.tiles[3][7] = PlantTile(crop="WHEAT", watered_today=False)

        actions = router.assign_actions(empty_game_state)

        # Since NW has 0 tasks, workers cross the boundary towards NE (EAST)
        assert actions["farmer"] == ["EAST"]
        assert actions["hands"][0] == ["EAST"]


class TestSmartBackpackRetention:
    """Validates Smart Backpack Retention: no mid-day shed transit; free night drop."""

    def test_smart_backpack_no_immediate_drop_midday(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        # Farmer at (1,1) carrying 4 harvested Melons
        farm.farmer = UnitState(id=0, x=1, y=1, inventory={"MELON": 4})

        # Plant at (1,2) needs water
        farm.tiles[2][1] = PlantTile(crop="WHEAT", watered_today=False)

        actions = router.assign_actions(empty_game_state)

        # Farmer does NOT walk to shed (towards (4,4)), but moves SOUTH to (1,2) to water the plant!
        assert actions["farmer"] == ["SOUTH"]

    def test_smart_backpack_drop_at_shed_adjacent_zero_cost(self, router: TacticalRouter, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        # Farmer at (4,4) [adjacent to shed] carrying 3 Strawberries
        farm.farmer = UnitState(id=0, x=4, y=4, inventory={"STRAWBERRY": 3})

        actions = router.assign_actions(empty_game_state)
        # Because cost is 0, executes DROP immediately
        assert actions["farmer"] == ["DROP"]


class TestShedOverflowSentinel:
    """Validates projected load calculation and 80% shed capacity critical alerts."""

    def test_shed_overflow_sentinel_projected_load(self, empty_game_state: GameState):
        farm = empty_game_state.my_farm
        farm.shed = {"WHEAT": 50, "CARROT": 15}  # 65 in shed
        farm.farmer = UnitState(id=0, x=0, y=0, inventory={"MELON": 10})  # 10 in backpack
        farm.hands = [UnitState(id=1, x=1, y=1, inventory={"TOMATO": 8})]  # 8 in backpack

        # Total projected load = 65 + 10 + 8 = 83 items
        assert farm.total_backpack_load == 18
        assert farm.projected_shed_load == 83
        # >= 80% threshold -> is_shed_critical must return True
        assert farm.is_shed_critical(0.80) is True

        # If backpacks are empty (65 items in shed): 65 < 80 -> False
        farm.farmer.inventory = {}
        farm.hands[0].inventory = {}
        assert farm.projected_shed_load == 65
        assert farm.is_shed_critical(0.80) is False


