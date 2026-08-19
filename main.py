"""
Entry point for the Kaggle Kaggriculture environment.
Combines Market Analytics Engine (Capa 1), Macro-Planner (Capa 2), and Tactical Router (Capa 3).
"""

from typing import Any, Dict, List, Optional
import os
import sys

# Ensure local modules and packages are accessible across all execution modes
candidates = [
    os.getcwd(),
    os.path.dirname(__file__) if "__file__" in globals() else None,
    "/kaggle_simulations/agent",
    os.path.abspath("."),
]
for candidate in candidates:
    if candidate and os.path.exists(os.path.join(candidate, "models")):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        break

from models.constants import (
    CropType,
    ProductType,
    FarmerAction,
    MarketAction,
    MAX_MARKET_ORDERS_PER_TURN,
    PRICE_FLOOR,
)
from models.state_representation import (
    GameState,
)
from engine.market_simulator import MarketSimulator
from engine.macro_planner import MacroPlanner, MacroPlan
from engine.tactical_router import TacticalRouter
from utils.logger import log


# Global engine instances
market_sim = MarketSimulator()
macro_planner = MacroPlanner()
tactical_router = TacticalRouter()

# State cache per player (handles both single agent and self-play in same process)
_current_macro_plans: Dict[int, MacroPlan] = {}
_last_plan_days: Dict[int, int] = {}


def agent(obs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main agent function called by kaggle-environments at every simulation turn.
    """
    global _current_macro_plans, _last_plan_days

    # 1. High speed state parsing (< 0.2 ms)
    game_state = GameState.from_raw_obs(obs)
    my_farm = game_state.my_farm
    my_id = game_state.my_player_id
    current_day = game_state.day
    market_orders: List[List[Any]] = []

    # Reset player state on new match
    if game_state.global_turn == 0:
        _last_plan_days[my_id] = -1

    # 2. Daily Macro-Planning (Capa 2)
    # Generate new plan at turn 0 of each day or reset upon new game
    if _last_plan_days.get(my_id, -1) != current_day or my_id not in _current_macro_plans:
        _current_macro_plans[my_id] = macro_planner.generate_daily_macro_plan(game_state, market_sim)
        _last_plan_days[my_id] = current_day

    plan = _current_macro_plans[my_id]
    days_left = 30 - current_day

    # 3. Market Analytics & Liquidation (Capa 1) - 1° SELL ORDERS
    # Shed Overflow Sentinel: If projected load >= 80, force mass liquidation
    is_critical_shed = my_farm.is_shed_critical(0.80)

    for prod_name, qty in my_farm.shed.items():
        if qty > 0 and prod_name not in ("GOOSE", "COW", "SHEEP") and len(market_orders) < MAX_MARKET_ORDERS_PER_TURN:
            current_market_inv = game_state.market.get_inventory(prod_name)

            if days_left <= 2:
                # Terminal Liquidation: sell all inventory before simulation ends
                market_orders.append([MarketAction.SELL.value, prod_name, qty])
            elif is_critical_shed or days_left <= 4:
                # Emergency Shed Sentinel or late season: aggressive batch selling with price floor 1-2
                sell_qty = market_sim.find_optimal_sell_batch(
                    resource=prod_name,
                    current_inventory=current_market_inv,
                    available_quantity=qty,
                    min_acceptable_price=1 if is_critical_shed else 2,
                )
                if sell_qty > 0:
                    market_orders.append([MarketAction.SELL.value, prod_name, sell_qty])
                elif is_critical_shed:
                    # Force selling at least batch to prevent discard
                    market_orders.append([MarketAction.SELL.value, prod_name, min(qty, 20)])
            else:
                # Regular season batch selling
                sell_qty = market_sim.find_optimal_sell_batch(
                    resource=prod_name,
                    current_inventory=current_market_inv,
                    available_quantity=qty,
                    min_acceptable_price=5,
                )
                if sell_qty > 0:
                    market_orders.append([MarketAction.SELL.value, prod_name, sell_qty])

    # 4. Turn 0 Macro Expansion & Resource Orders (Sequential Execution)
    if game_state.is_first_turn_of_day:
        # 2° BUY_LAND Order
        if plan.buy_land_quadrant is not None and len(market_orders) < MAX_MARKET_ORDERS_PER_TURN:
            market_orders.append([MarketAction.BUY_LAND.value])

        # 3° HIRE Orders (Labor scaling)
        for _ in range(plan.hands_to_hire):
            if len(market_orders) < MAX_MARKET_ORDERS_PER_TURN:
                market_orders.append([MarketAction.HIRE.value])

        # 4° BUY_PRODUCT / BUY_ANIMAL Orders
        if plan.wheat_buy_orders > 0 and len(market_orders) < MAX_MARKET_ORDERS_PER_TURN:
            market_orders.append([MarketAction.BUY_PRODUCT.value, ProductType.WHEAT.value, plan.wheat_buy_orders])

        for animal_name, qty in plan.animal_orders.items():
            if qty > 0 and len(market_orders) < MAX_MARKET_ORDERS_PER_TURN:
                market_orders.append([MarketAction.BUY_ANIMAL.value, animal_name, qty])

        # 5° BUY_SEED Orders
        for crop_name, qty in plan.seed_orders.items():
            if qty > 0 and len(market_orders) < MAX_MARKET_ORDERS_PER_TURN:
                market_orders.append([MarketAction.BUY_SEED.value, crop_name, qty])

    # Mid-day emergency seed replenishment if farm is completely out of seeds (only after turn 0)
    total_seeds = sum(my_farm.seeds.values())
    if not game_state.is_first_turn_of_day and total_seeds == 0 and my_farm.money >= 100 and days_left > 3 and len(market_orders) < MAX_MARKET_ORDERS_PER_TURN:
        pref = plan.preferred_seed_order if plan else [CropType.CARROT.value, CropType.WHEAT.value]
        top_crop = pref[0] if pref else CropType.WHEAT.value
        market_orders.append([MarketAction.BUY_SEED.value, top_crop, 5])


    # 4. Tactical Spatial Routing & Action Dispatch (Capa 3)
    preferred_order = plan.preferred_seed_order if plan else None
    fert_targets = plan.fertilizer_target_tiles if plan else None
    build_orders = plan.structure_build_orders if plan else None

    tactical_response = tactical_router.assign_actions(
        game_state,
        preferred_seed_order=preferred_order,
        fertilizer_target_tiles=fert_targets,
        structure_build_orders=build_orders,
    )

    return {
        "farmer": tactical_response["farmer"],
        "hands": tactical_response["hands"],
        "market": market_orders,
    }

