"""
Entry point for the Kaggle Kaggriculture environment.
Combines Market Analytics Engine (Capa 1) and Tactical Spatial Scheduler (Capa 3).
"""

from typing import Any, Dict, List, Optional
import os
import sys

# Ensure local modules are accessible in Kaggle submission or local runs
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
)
from engine.market_simulator import MarketSimulator
from engine.tactical_router import TacticalRouter
from utils.logger import log


# Global engine instances
market_sim = MarketSimulator()
tactical_router = TacticalRouter()


def agent(obs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main agent function called by kaggle-environments at every simulation turn.
    """
    # 1. High speed state parsing (< 0.2 ms)
    game_state = GameState.from_raw_obs(obs)
    my_farm = game_state.my_farm
    market_orders: List[List[Any]] = []

    # 2. Market Analytics & Liquidation (Capa 1)
    # Check if we have produce in the shed to sell
    for prod_name, qty in my_farm.shed.items():
        if qty > 0 and len(market_orders) < 10:
            current_market_inv = game_state.market.get_inventory(prod_name)
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
    total_seeds = sum(my_farm.seeds.values())
    if total_seeds == 0 and my_farm.money >= 50 and len(market_orders) < 10:
        # Buy a batch of seeds
        seeds_to_buy = min(5, int(my_farm.money // 10))
        if seeds_to_buy > 0:
            market_orders.append([MarketAction.BUY_SEED.value, CropType.WHEAT.value, seeds_to_buy])

    # 3. Tactical Spatial Routing & Action Dispatch (Capa 3)
    tactical_response = tactical_router.assign_actions(game_state)

    return {
        "farmer": tactical_response["farmer"],
        "hands": tactical_response["hands"],
        "market": market_orders,
    }
