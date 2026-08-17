"""
Dataclasses and fast state representation for Kaggriculture.
Optimized for low-latency serialization/deserialization.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .constants import (
    BOARD_SIZE,
    SHED_CAPACITY,
    TURNS_PER_DAY,
    TOTAL_DAYS,
    TOTAL_TURNS,
    CropType,
    AnimalType,
    ProductType,
    StructureType,
    Quadrant,
)


# ==========================================
# Tile Representation Dataclasses
# ==========================================

@dataclass
class EmptyTile:
    kind: str = "EMPTY"


@dataclass
class LockedTile:
    kind: str = "LOCKED"


@dataclass
class WeedTile:
    kind: str = "WEED"


@dataclass
class PlantTile:
    kind: str = "PLANT"
    crop: str = "WHEAT"
    planted_day: int = 0
    watered_today: bool = False
    consecutive_unwatered: int = 0
    yield_units: int = 0
    max_lifespan_step: int = -1
    fertilized_until_day: int = -1

    @property
    def is_fertilized(self) -> bool:
        return self.fertilized_until_day >= 0

    @property
    def is_ready_to_harvest(self) -> bool:
        return self.yield_units > 0

    @property
    def is_in_danger(self) -> bool:
        """True if missing water today will turn this plant into a weed."""
        return self.consecutive_unwatered >= 1 and not self.watered_today


@dataclass
class StructureTile:
    kind: str = "COOP"  # "COOP" or "PASTURE"
    animal: Optional[str] = None  # "GOOSE", "COW", "SHEEP", or None
    placed_day: int = 0
    yield_units: int = 0
    fed_today: bool = False
    consecutive_unfed: int = 0
    cared_today: bool = False
    fertilizer_available: bool = False
    pending_care_bonus: int = 0

    @property
    def is_occupied(self) -> bool:
        return self.animal is not None

    @property
    def is_in_danger(self) -> bool:
        #True if missing food today will cause the animal to escape
        return self.is_occupied and self.consecutive_unfed >= 1 and not self.fed_today


TileState = Union[EmptyTile, LockedTile, WeedTile, PlantTile, StructureTile]


# ==========================================
# Unit Representation
# ==========================================

@dataclass
class UnitState:
    id: int  # 0 = Farmer, 1..N = Farm Hands
    x: int
    y: int
    inventory: Dict[str, int] = field(default_factory=dict)

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.x, self.y)

    @property
    def inventory_count(self) -> int:
        return sum(self.inventory.values())

    @property
    def has_inventory(self) -> bool:
        return self.inventory_count > 0


# ==========================================
# Farm State
# ==========================================

@dataclass
class FarmState:
    player_id: int
    money: float
    tiles: List[List[TileState]]
    farmer: UnitState
    hands: List[UnitState] = field(default_factory=list)
    unlocked_quadrants: Set[str] = field(default_factory=lambda: {"NW"})
    hires_today: int = 0
    shed: Dict[str, int] = field(default_factory=dict)
    seeds: Dict[str, int] = field(default_factory=dict)

    @property
    def all_units(self) -> List[UnitState]:
        return [self.farmer] + self.hands

    @property
    def worker_count(self) -> int:
        return 1 + len(self.hands)

    @property
    def shed_items_count(self) -> int:
        """Total items in shed (excluding seeds, which are tracked separately)."""
        return sum(self.shed.values())

    @property
    def shed_free_capacity(self) -> int:
        return max(0, SHED_CAPACITY - self.shed_items_count)

    @property
    def shed_occupancy_ratio(self) -> float:
        return self.shed_items_count / float(SHED_CAPACITY)

    def is_shed_critical(self, threshold: float = 0.85) -> bool:
        """Sentinel check: returns True if shed occupancy reaches or exceeds threshold."""
        return self.shed_items_count >= int(threshold * SHED_CAPACITY)

    def get_tile(self, x: int, y: int) -> TileState:
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            return self.tiles[y][x]
        return LockedTile()

    def get_plant_count(self, crop: Optional[str] = None) -> int:
        count = 0
        for row in self.tiles:
            for tile in row:
                if isinstance(tile, PlantTile):
                    if crop is None or tile.crop == crop:
                        count += 1
        return count

    def get_animal_count(self, animal: Optional[str] = None) -> int:
        count = 0
        for row in self.tiles:
            for tile in row:
                if isinstance(tile, StructureTile) and tile.is_occupied:
                    if animal is None or tile.animal == animal:
                        count += 1
        return count

    def get_empty_structures(self, structure_type: Optional[str] = None) -> List[Tuple[int, int, StructureTile]]:
        """Returns coordinates and tile for unoccupied structures (kind COOP or PASTURE)."""
        empty_structs: List[Tuple[int, int, StructureTile]] = []
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, StructureTile) and not tile.is_occupied:
                    if structure_type is None or tile.kind == structure_type:
                        empty_structs.append((x, y, tile))
        return empty_structs

    def get_occupied_structures(self, animal: Optional[str] = None) -> List[Tuple[int, int, StructureTile]]:
        """Returns coordinates and tile for structures with living animals."""
        occupied: List[Tuple[int, int, StructureTile]] = []
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, StructureTile) and tile.is_occupied:
                    if animal is None or tile.animal == animal:
                        occupied.append((x, y, tile))
        return occupied

    def get_fertilizable_plants(self, current_day: int = 0) -> List[Tuple[int, int, PlantTile]]:
        """Returns active plant tiles that can receive fertilizer."""
        plants: List[Tuple[int, int, PlantTile]] = []
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, PlantTile):
                    if tile.fertilized_until_day < current_day:
                        plants.append((x, y, tile))
        return plants

    def get_animals_in_shed(self) -> Dict[str, int]:
        """Returns purchased animals held in shed awaiting placement."""
        animal_keys = {AnimalType.GOOSE.value, AnimalType.COW.value, AnimalType.SHEEP.value}
        return {k: v for k, v in self.shed.items() if k in animal_keys and v > 0}



# ==========================================
# Market and Town State
# ==========================================

@dataclass
class MarketState:
    inventory: Dict[str, int] = field(default_factory=dict)
    prices: Dict[str, int] = field(default_factory=dict)

    def get_price(self, resource: str) -> int:
        return self.prices.get(resource, 1)

    def get_inventory(self, resource: str) -> int:
        return self.inventory.get(resource, 10000)


@dataclass
class TownState:
    unlocked_shops: List[str] = field(default_factory=list)

    @property
    def shop_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for shop in self.unlocked_shops:
            counts[shop] = counts.get(shop, 0) + 1
        return counts


# ==========================================
# Root Game State and Fast Parser
# ==========================================

@dataclass
class GameState:
    day: int
    hour: int
    global_turn: int
    my_player_id: int
    my_farm: FarmState
    opponent_farm: FarmState
    market: MarketState
    town: TownState

    @property
    def turns_left_today(self) -> int:
        return max(0, TURNS_PER_DAY - self.hour - 1)

    @property
    def days_left(self) -> int:
        return max(0, TOTAL_DAYS - self.day - 1)

    @property
    def total_turns_left(self) -> int:
        return max(0, TOTAL_TURNS - self.global_turn - 1)

    @property
    def is_first_turn_of_day(self) -> bool:
        return self.hour == 0

    @property
    def is_last_turn_of_day(self) -> bool:
        return self.hour == TURNS_PER_DAY - 1

    @classmethod
    def from_raw_obs(cls, obs: Dict[str, Any]) -> GameState:
        """
        Fast deserializer from raw Kaggle observation dictionary.
        """
        my_id = obs.get("player", 0)
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)
        global_turn = day * TURNS_PER_DAY + hour

        # Market
        raw_market = obs.get("market", {})
        market = MarketState(
            inventory=dict(raw_market.get("inventory", {})),
            prices=dict(raw_market.get("prices", {})),
        )

        # Town
        raw_town = obs.get("town", {})
        town = TownState(
            unlocked_shops=list(raw_town.get("unlocked_shops", [])),
        )

        # Private state for active player
        private = obs.get("private", {})
        shed_inv = dict(private.get("shed", {}))
        seeds_inv = dict(private.get("seeds", {}))
        unit_inventories = private.get("inventories", [])

        # Process both farms
        raw_farms = obs.get("farms", [{}, {}])
        farms: List[FarmState] = []

        for p_idx in range(len(raw_farms)):
            raw_farm = raw_farms[p_idx]
            is_me = (p_idx == my_id)

            money = float(raw_farm.get("money", 0.0))
            unlocked_quads = set(raw_farm.get("unlocked_quadrants", ["NW"]))
            hires_today = int(raw_farm.get("hires_today", 0))

            # Units
            farmer_pos = raw_farm.get("farmer", [4, 4])
            farmer_inv = unit_inventories[0] if (is_me and len(unit_inventories) > 0) else {}
            farmer = UnitState(id=0, x=farmer_pos[0], y=farmer_pos[1], inventory=dict(farmer_inv))

            hands: List[UnitState] = []
            raw_hands = raw_farm.get("hands", [])
            for h_idx, h_pos in enumerate(raw_hands):
                hand_inv = unit_inventories[h_idx + 1] if (is_me and len(unit_inventories) > h_idx + 1) else {}
                hands.append(UnitState(id=h_idx + 1, x=h_pos[0], y=h_pos[1], inventory=dict(hand_inv)))

            # Tiles matrix
            raw_tiles = raw_farm.get("tiles", [])
            tiles_matrix: List[List[TileState]] = []

            for r_idx in range(BOARD_SIZE):
                row_list: List[TileState] = []
                raw_row = raw_tiles[r_idx] if r_idx < len(raw_tiles) else []
                for c_idx in range(BOARD_SIZE):
                    t_val = raw_row[c_idx] if c_idx < len(raw_row) else None
                    if t_val is None:
                        row_list.append(EmptyTile())
                    elif t_val == "LOCKED":
                        row_list.append(LockedTile())
                    elif isinstance(t_val, dict):
                        kind = t_val.get("kind", "")
                        if kind == "PLANT":
                            row_list.append(PlantTile(
                                kind="PLANT",
                                crop=t_val.get("crop", "WHEAT"),
                                planted_day=t_val.get("planted_day", 0),
                                watered_today=bool(t_val.get("watered_today", False)),
                                consecutive_unwatered=t_val.get("consecutive_unwatered", 0),
                                yield_units=t_val.get("yield_units", 0),
                                max_lifespan_step=t_val.get("max_lifespan_step", -1),
                                fertilized_until_day=t_val.get("fertilized_until_day", -1),
                            ))
                        elif kind in ("COOP", "PASTURE"):
                            row_list.append(StructureTile(
                                kind=kind,
                                animal=t_val.get("animal", None),
                                placed_day=t_val.get("placed_day", 0),
                                yield_units=t_val.get("yield_units", 0),
                                fed_today=bool(t_val.get("fed_today", False)),
                                consecutive_unfed=t_val.get("consecutive_unfed", 0),
                                cared_today=bool(t_val.get("cared_today", False)),
                                fertilizer_available=bool(t_val.get("fertilizer_available", False)),
                                pending_care_bonus=t_val.get("pending_care_bonus", 0),
                            ))
                        elif kind == "WEED":
                            row_list.append(WeedTile())
                        else:
                            row_list.append(EmptyTile())
                    else:
                        row_list.append(EmptyTile())
                tiles_matrix.append(row_list)

            farm_state = FarmState(
                player_id=p_idx,
                money=money,
                tiles=tiles_matrix,
                farmer=farmer,
                hands=hands,
                unlocked_quadrants=unlocked_quads,
                hires_today=hires_today,
                shed=shed_inv if is_me else {},
                seeds=seeds_inv if is_me else {},
            )
            farms.append(farm_state)

        return cls(
            day=day,
            hour=hour,
            global_turn=global_turn,
            my_player_id=my_id,
            my_farm=farms[my_id],
            opponent_farm=farms[1 - my_id] if len(farms) > 1 else farms[0],
            market=market,
            town=town,
        )
