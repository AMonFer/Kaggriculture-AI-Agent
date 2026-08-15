"""
Entry point for the Kaggle Kaggriculture environment.
"""

from typing import Any, Dict, List, Optional
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.constants import (
    CropType,
    ProductType,
    FarmerAction,
    MarketAction,
    SHED_ACCESS_TILES,
    PRICE_FLOOR,
)
from models.state_representation import (
    GameState,
    PlantTile,
    EmptyTile,
    WeedTile,
)
from engine.market_simulator import MarketSimulator
from utils.logger import log


# Global simulator instance
market_sim = MarketSimulator()


def agent(obs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main agent function called by kaggle-environments at every simulation turn.
    """
    # Obtain the current state
    game_state = GameState.from_raw_obs(obs)
    my_farm = game_state.my_farm
    market_orders: List[List[Any]] = []
    farmer_actions: List[str] = []
    hands_actions: List[List[str]] = []

    # Market Analytics & Liquidation (Capa 1)
    # Check if we have produce in the shed to sell
    for prod_name, qty in my_farm.shed.items():
        if qty > 0 and len(market_orders) < 10:
            current_market_inv = game_state.market.get_inventory(prod_name)
            # Find how many we can sell profitably today
            # If shed is getting full (>80%), sell more aggressively
            min_p = 2 if my_farm.is_shed_critical(0.80) else 5
            sell_qty = market_sim.find_optimal_sell_batch(
                resource=prod_name,
                current_inventory=current_market_inv,
                available_quantity=qty,
                min_acceptable_price=min_p,
            )
            if sell_qty > 0:
                market_orders.append([MarketAction.SELL.value, prod_name, sell_qty])

    # Check if we need seeds and have budget
    wheat_seeds = my_farm.seeds.get(CropType.WHEAT.value, 0)
    if wheat_seeds == 0 and my_farm.money >= 50 and len(market_orders) < 10:
        # Buy a small batch of seeds to keep operations going
        seeds_to_buy = min(5, int(my_farm.money // 10))
        if seeds_to_buy > 0:
            market_orders.append([MarketAction.BUY_SEED.value, CropType.WHEAT.value, seeds_to_buy])

    # Farmer Tactical Action (For now it stands at the same tile)
    fx, fy = my_farm.farmer.pos
    current_tile = my_farm.get_tile(fx, fy)

    if isinstance(current_tile, PlantTile):
        if current_tile.is_ready_to_harvest:
            farmer_actions = [FarmerAction.HARVEST.value]
        elif not current_tile.watered_today:
            farmer_actions = [FarmerAction.WATER.value]
        else:
            farmer_actions = [FarmerAction.PASS.value]
    elif isinstance(current_tile, EmptyTile):
        if my_farm.seeds.get(CropType.WHEAT.value, 0) > 0:
            farmer_actions = [FarmerAction.PLANT.value, CropType.WHEAT.value]
        else:
            farmer_actions = [FarmerAction.PASS.value]
    elif isinstance(current_tile, WeedTile):
        farmer_actions = [FarmerAction.DIG.value]
    else:
        farmer_actions = [FarmerAction.PASS.value]

    # For any hired hands, pass by default in Phase 1
    for _ in my_farm.hands:
        hands_actions.append([FarmerAction.PASS.value])

    return {
        "farmer": farmer_actions,
        "hands": hands_actions,
        "market": market_orders,
    }
