"""
Game constants, enums, market parameters, and unit/crop specifications for Kaggriculture.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple


# ==========================================
# Enums
# ==========================================

class CropType(str, Enum):
    WHEAT = "WHEAT"
    CARROT = "CARROT"
    TOMATO = "TOMATO"
    STRAWBERRY = "STRAWBERRY"
    MELON = "MELON"


class AnimalType(str, Enum):
    GOOSE = "GOOSE"
    COW = "COW"
    SHEEP = "SHEEP"


class ProductType(str, Enum):
    WHEAT = "WHEAT"
    CARROT = "CARROT"
    TOMATO = "TOMATO"
    STRAWBERRY = "STRAWBERRY"
    MELON = "MELON"
    EGG = "EGG"
    MILK = "MILK"
    WOOL = "WOOL"
    FERTILIZER = "FERTILIZER"


class StructureType(str, Enum):
    COOP = "COOP"
    PASTURE = "PASTURE"


class Quadrant(str, Enum):
    NW = "NW"
    NE = "NE"
    SW = "SW"
    SE = "SE"


class Direction(str, Enum):
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"


class FarmerAction(str, Enum):
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"
    PLANT = "PLANT"
    WATER = "WATER"
    HARVEST = "HARVEST"
    FERTILIZE = "FERTILIZE"
    FEED = "FEED"
    CARE = "CARE"
    COLLECT_FERTILIZER = "COLLECT_FERTILIZER"
    BUILD_COOP = "BUILD_COOP"
    BUILD_PASTURE = "BUILD_PASTURE"
    DIG = "DIG"
    PICKUP = "PICKUP"
    DROP = "DROP"
    PLACE = "PLACE"
    PASS = "PASS"


class MarketAction(str, Enum):
    BUY_SEED = "BUY_SEED"
    BUY_ANIMAL = "BUY_ANIMAL"
    BUY_PRODUCT = "BUY_PRODUCT"
    SELL = "SELL"
    HIRE = "HIRE"
    BUY_LAND = "BUY_LAND"


class ShopType(str, Enum):
    BAKERY = "BAKERY"
    PIZZA_SHOP = "PIZZA_SHOP"
    BRUNCH_SPOT = "BRUNCH_SPOT"
    YARN_STORE = "YARN_STORE"
    ICE_CREAM_SHOP = "ICE_CREAM_SHOP"
    PET_CAFE = "PET_CAFE"
    SMOOTHIE_SHOP = "SMOOTHIE_SHOP"
    FARMERS_MARKET = "FARMERS_MARKET"


# ==========================================
# Global Game Constants
# ==========================================

BOARD_SIZE: int = 10
QUADRANT_SIZE: int = 5
TOTAL_DAYS: int = 30
TURNS_PER_DAY: int = 24
TOTAL_TURNS: int = TOTAL_DAYS * TURNS_PER_DAY
STARTING_MONEY: int = 3000
SHED_CAPACITY: int = 100
BASE_MARKET_INVENTORY: int = 10000
PRICE_FLOOR: int = 1
MAX_MARKET_ORDERS_PER_TURN: int = 10

# Quadrant purchase costs and bounds
QUADRANT_COSTS: Dict[str, int] = {
    "NW": 0,
    "NE": 1000,
    "SW": 2000,
    "SE": 4000,
}

QUADRANT_BOUNDS: Dict[str, Tuple[int, int, int, int]] = {
    # (min_x, max_x, min_y, max_y) inclusive
    "NW": (0, 4, 0, 4),
    "NE": (5, 9, 0, 4),
    "SW": (0, 4, 5, 9),
    "SE": (5, 9, 5, 9),
}

SHED_ACCESS_TILES: List[Tuple[int, int]] = [
    (4, 4),  # NW quadrant
    (5, 4),  # NE quadrant
    (4, 5),  # SW quadrant
    (5, 5),  # SE quadrant
]

# Hiring cost
FIBONACCI_COSTS: List[int] = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

def get_fibonacci_cost(n_already_hired_today: int, cost_mult: int = 1) -> int:
    #Returns the cost of hiring the next farm hand given how many were hired today
    if n_already_hired_today < len(FIBONACCI_COSTS):
        return cost_mult * FIBONACCI_COSTS[n_already_hired_today]
    # Compute further Fibonacci numbers
    a, b = FIBONACCI_COSTS[-2], FIBONACCI_COSTS[-1]
    for _ in range(len(FIBONACCI_COSTS), n_already_hired_today + 1):
        a, b = b, a + b
    return cost_mult * b


# ==========================================
# Object Specifications
# ==========================================

@dataclass(frozen=True)
class CropSpec:
    crop_type: CropType
    seed_cost: int
    base_market_price: int
    time_to_first_yield: int
    time_to_max_yield: int
    max_yield_unfertilized: int
    max_yield_fertilized: int
    is_ongoing: bool
    yield_interval: Optional[int] = None
    scheduled_yield_count: Optional[int] = None


CROP_SPECS: Dict[str, CropSpec] = {
    CropType.WHEAT.value: CropSpec(
        crop_type=CropType.WHEAT,
        seed_cost=10,
        base_market_price=25,
        time_to_first_yield=2,
        time_to_max_yield=4,
        max_yield_unfertilized=4,
        max_yield_fertilized=6,
        is_ongoing=False,
    ),
    CropType.CARROT.value: CropSpec(
        crop_type=CropType.CARROT,
        seed_cost=20,
        base_market_price=35,
        time_to_first_yield=2,
        time_to_max_yield=3,
        max_yield_unfertilized=3,
        max_yield_fertilized=4,
        is_ongoing=False,
    ),
    CropType.TOMATO.value: CropSpec(
        crop_type=CropType.TOMATO,
        seed_cost=50,
        base_market_price=60,
        time_to_first_yield=8,
        time_to_max_yield=11,
        max_yield_unfertilized=4,
        max_yield_fertilized=8,
        is_ongoing=True,
        yield_interval=1,
        scheduled_yield_count=4,
    ),
    CropType.STRAWBERRY.value: CropSpec(
        crop_type=CropType.STRAWBERRY,
        seed_cost=100,
        base_market_price=120,
        time_to_first_yield=10,
        time_to_max_yield=16,
        max_yield_unfertilized=4,
        max_yield_fertilized=8,
        is_ongoing=True,
        yield_interval=2,
        scheduled_yield_count=4,
    ),
    CropType.MELON.value: CropSpec(
        crop_type=CropType.MELON,
        seed_cost=80,
        base_market_price=250,
        time_to_first_yield=10,
        time_to_max_yield=10,
        max_yield_unfertilized=6,
        max_yield_fertilized=6,
        is_ongoing=False,
    ),
}


@dataclass(frozen=True)
class AnimalSpec:
    animal_type: AnimalType
    product_type: ProductType
    purchase_cost: int
    base_market_price: int
    structure_type: StructureType
    time_to_first_yield: int
    yield_interval: int
    max_held: int
    daily_feed: ProductType = ProductType.WHEAT


ANIMAL_SPECS: Dict[str, AnimalSpec] = {
    AnimalType.GOOSE.value: AnimalSpec(
        animal_type=AnimalType.GOOSE,
        product_type=ProductType.EGG,
        purchase_cost=300,
        base_market_price=50,
        structure_type=StructureType.COOP,
        time_to_first_yield=4,
        yield_interval=1,
        max_held=4,
    ),
    AnimalType.COW.value: AnimalSpec(
        animal_type=AnimalType.COW,
        product_type=ProductType.MILK,
        purchase_cost=400,
        base_market_price=160,
        structure_type=StructureType.PASTURE,
        time_to_first_yield=8,
        yield_interval=2,
        max_held=6,
    ),
    AnimalType.SHEEP.value: AnimalSpec(
        animal_type=AnimalType.SHEEP,
        product_type=ProductType.WOOL,
        purchase_cost=500,
        base_market_price=200,
        structure_type=StructureType.PASTURE,
        time_to_first_yield=6,
        yield_interval=3,
        max_held=6,
    ),
}


# ==========================================
# Market Parameters (Official Engine Spec)
# ==========================================

CurveShapeFunc = Literal["linear", "sq", "sqrt", "log", "log10"]

@dataclass(frozen=True)
class MarketParamSpec:
    base: int
    I0: int
    T: int
    below_func: CurveShapeFunc
    below_target: float
    above_func: CurveShapeFunc
    above_target: float


MARKET_PARAMS: Dict[str, MarketParamSpec] = {
    ProductType.WHEAT.value: MarketParamSpec(
        base=25,
        I0=10000,
        T=400,
        below_func="sqrt",
        below_target=0.80,
        above_func="log",
        above_target=0.20,
    ),
    ProductType.CARROT.value: MarketParamSpec(
        base=35,
        I0=10000,
        T=450,
        below_func="log",
        below_target=0.20,
        above_func="sqrt",
        above_target=0.70,
    ),
    ProductType.TOMATO.value: MarketParamSpec(
        base=60,
        I0=10000,
        T=200,
        below_func="linear",
        below_target=0.40,
        above_func="sqrt",
        above_target=0.60,
    ),
    ProductType.STRAWBERRY.value: MarketParamSpec(
        base=120,
        I0=10000,
        T=100,
        below_func="sqrt",
        below_target=0.70,
        above_func="linear",
        above_target=1.60,
    ),
    ProductType.MELON.value: MarketParamSpec(
        base=250,
        I0=10000,
        T=300,
        below_func="log",
        below_target=0.20,
        above_func="sq",
        above_target=3.60,
    ),
    ProductType.EGG.value: MarketParamSpec(
        base=50,
        I0=10000,
        T=332,
        below_func="linear",
        below_target=0.40,
        above_func="log",
        above_target=0.20,
    ),
    ProductType.MILK.value: MarketParamSpec(
        base=160,
        I0=10000,
        T=122,
        below_func="sqrt",
        below_target=0.60,
        above_func="linear",
        above_target=1.60,
    ),
    ProductType.WOOL.value: MarketParamSpec(
        base=200,
        I0=10000,
        T=105,
        below_func="log",
        below_target=0.20,
        above_func="sq",
        above_target=3.20,
    ),
    ProductType.FERTILIZER.value: MarketParamSpec(
        base=100,
        I0=10000,
        T=200,
        below_func="linear",
        below_target=0.40,
        above_func="linear",
        above_target=0.40,
    ),
}


# ==========================================
# Town Demand Parameters
# ==========================================

TOWN_SHOP_UNLOCK_INTERVAL_DAYS: int = 3
TOWN_SHOP_SELL_INTERVAL_TURNS: int = 4
TOWN_CENTER_SELL_INTERVAL_TURNS: int = 24
MAX_TOWN_SHOPS: int = 8

TOWN_SHOPS: Dict[str, Dict[str, int]] = {
    ShopType.BAKERY.value: {ProductType.EGG.value: 1, ProductType.WHEAT.value: 1},
    ShopType.PIZZA_SHOP.value: {ProductType.MILK.value: 1, ProductType.TOMATO.value: 1, ProductType.WHEAT.value: 1},
    ShopType.BRUNCH_SPOT.value: {ProductType.EGG.value: 1, ProductType.WHEAT.value: 1, ProductType.STRAWBERRY.value: 1},
    ShopType.YARN_STORE.value: {ProductType.WOOL.value: 2},
    ShopType.ICE_CREAM_SHOP.value: {ProductType.STRAWBERRY.value: 1, ProductType.MILK.value: 1, ProductType.WHEAT.value: 1},
    ShopType.PET_CAFE.value: {ProductType.CARROT.value: 2},
    ShopType.SMOOTHIE_SHOP.value: {ProductType.STRAWBERRY.value: 1, ProductType.MILK.value: 1},
    ShopType.FARMERS_MARKET.value: {
        ProductType.WHEAT.value: 1,
        ProductType.CARROT.value: 1,
        ProductType.TOMATO.value: 1,
        ProductType.STRAWBERRY.value: 1,
    },
}
