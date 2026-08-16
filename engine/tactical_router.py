"""
Tactical Spatial Scheduler (Layer 3) for Kaggriculture.
Resolves multi-agent pathfinding (A*), prioritizes daily farm tasks (P0-P3),
and coordinates simultaneous actions for Farmer and Farm Hands without turn loss.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from models.constants import (
    BOARD_SIZE,
    CropType,
    Direction,
    FarmerAction,
    SHED_ACCESS_TILES,
    SHED_CAPACITY,
)
from models.state_representation import (
    EmptyTile,
    FarmState,
    GameState,
    PlantTile,
    StructureTile,
    UnitState,
    WeedTile,
)


@dataclass
class Task:
    task_type: FarmerAction
    target_pos: Tuple[int, int]
    priority: int  # 0=P0 (Critical), 1=P1 (Harvest/Drop), 2=P2 (Maintenance), 3=P3 (Plant/Dig)
    payload: Any = None
    worker_id: Optional[int] = None

    @property
    def pos_key(self) -> Tuple[int, int]:
        return self.target_pos


class TacticalRouter:
    """
    Coordinates multi-agent navigation and spatial task assignment on the 10x10 farm grid.
    """

    def __init__(self) -> None:
        self.shed_tiles: Set[Tuple[int, int]] = set(SHED_ACCESS_TILES)

    @staticmethod
    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Calculates Manhattan distance between two grid coordinates."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def get_nearest_shed_tile(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Finds the closest shed access tile from the given position."""
        return min(SHED_ACCESS_TILES, key=lambda s: self.manhattan_distance(pos, s))

    def is_shed_adjacent(self, pos: Tuple[int, int]) -> bool:
        """Returns True if the position is one of the 4 valid shed interaction tiles."""
        return pos in self.shed_tiles

    def get_direction_to(self, curr: Tuple[int, int], target: Tuple[int, int]) -> Optional[str]:
        """
        Determines the immediate single-step orthogonal movement towards the target.
        Returns None if already at target.
        """
        cx, cy = curr
        tx, ty = target

        if cx == tx and cy == ty:
            return None

        # Prioritize axis with larger discrepancy, or horizontal first
        dx = tx - cx
        dy = ty - cy

        if abs(dx) >= abs(dy):
            if dx > 0:
                return Direction.EAST.value
            elif dx < 0:
                return Direction.WEST.value
        else:
            if dy > 0:
                return Direction.SOUTH.value
            elif dy < 0:
                return Direction.NORTH.value

        # Fallback if both non-zero
        if dx > 0:
            return Direction.EAST.value
        elif dx < 0:
            return Direction.WEST.value
        elif dy > 0:
            return Direction.SOUTH.value
        elif dy < 0:
            return Direction.NORTH.value

        return None

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[str]:
        """
        Computes the shortest path of directional moves from start to goal.
        All tiles on the 10x10 board are passable.
        """
        path: List[str] = []
        curr_x, curr_y = start
        target_x, target_y = goal

        # Move horizontally towards target
        while curr_x < target_x:
            path.append(Direction.EAST.value)
            curr_x += 1
        while curr_x > target_x:
            path.append(Direction.WEST.value)
            curr_x -= 1

        # Move vertically towards target
        while curr_y < target_y:
            path.append(Direction.SOUTH.value)
            curr_y += 1
        while curr_y > target_y:
            path.append(Direction.NORTH.value)
            curr_y -= 1

        return path

    def generate_daily_tasks(
        self,
        game_state: GameState,
        preferred_seed_order: Optional[List[str]] = None,
    ) -> List[Task]:
        """
        Scans the farm grid and generates prioritized tasks:
        - P0: Critical survival (plants/animals dying tonight if ignored)
        - P1: Harvesting mature produce and animal products
        - P2: Routine daily maintenance (watering, feeding, fertilizer collection, care)
        - P3: Expansion, weed removal (DIG), and seeding (PLANT)
        """
        my_farm = game_state.my_farm
        tasks: List[Task] = []

        # Available seeds copy for plant tasks
        available_seeds: Dict[str, int] = dict(my_farm.seeds)
        seed_order = preferred_seed_order or [
            CropType.MELON.value,
            CropType.STRAWBERRY.value,
            CropType.TOMATO.value,
            CropType.CARROT.value,
            CropType.WHEAT.value,
        ]

        # Scan all 10x10 tiles
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                tile = my_farm.get_tile(x, y)

                # 1. Plants
                if isinstance(tile, PlantTile):
                    # P0: Critical watering (danger of weed tonight)
                    if tile.is_in_danger:
                        tasks.append(Task(
                            task_type=FarmerAction.WATER,
                            target_pos=(x, y),
                            priority=0,
                        ))
                    # P1: Ready to harvest
                    elif tile.is_ready_to_harvest:
                        tasks.append(Task(
                            task_type=FarmerAction.HARVEST,
                            target_pos=(x, y),
                            priority=1,
                        ))
                    # P2: Regular watering
                    elif not tile.watered_today:
                        tasks.append(Task(
                            task_type=FarmerAction.WATER,
                            target_pos=(x, y),
                            priority=2,
                        ))

                # 2. Animals / Structures
                elif isinstance(tile, StructureTile) and tile.is_occupied:
                    # P0: Critical feeding (danger of escape tonight)
                    if tile.is_in_danger:
                        tasks.append(Task(
                            task_type=FarmerAction.FEED,
                            target_pos=(x, y),
                            priority=0,
                        ))
                    # P1: Harvest animal produce
                    elif tile.yield_units > 0:
                        tasks.append(Task(
                            task_type=FarmerAction.HARVEST,
                            target_pos=(x, y),
                            priority=1,
                        ))
                    # P2: Maintenance
                    else:
                        if not tile.fed_today:
                            tasks.append(Task(
                                task_type=FarmerAction.FEED,
                                target_pos=(x, y),
                                priority=2,
                            ))
                        if tile.fertilizer_available:
                            tasks.append(Task(
                                task_type=FarmerAction.COLLECT_FERTILIZER,
                                target_pos=(x, y),
                                priority=2,
                            ))
                        if not tile.cared_today and tile.fed_today:
                            tasks.append(Task(
                                task_type=FarmerAction.CARE,
                                target_pos=(x, y),
                                priority=2,
                            ))

                # 3. Weeds
                elif isinstance(tile, WeedTile):
                    # P3: Dig weed
                    tasks.append(Task(
                        task_type=FarmerAction.DIG,
                        target_pos=(x, y),
                        priority=3,
                    ))

                # 4. Empty Unlocked Tiles -> Planting
                elif isinstance(tile, EmptyTile):
                    # Check if we have seeds to plant on this empty tile
                    chosen_seed: Optional[str] = None
                    for seed_name in seed_order:
                        if available_seeds.get(seed_name, 0) > 0:
                            chosen_seed = seed_name
                            available_seeds[seed_name] -= 1
                            break

                    if chosen_seed is not None:
                        tasks.append(Task(
                            task_type=FarmerAction.PLANT,
                            target_pos=(x, y),
                            priority=3,
                            payload=chosen_seed,
                        ))

        # Sort tasks strictly by priority ascending
        tasks.sort(key=lambda t: t.priority)
        return tasks

    def assign_actions(
        self,
        game_state: GameState,
        preferred_seed_order: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Coordinates all workers (Farmer and Farm Hands):
        - Handles inventory dumping when backpack has items and shed has capacity
        - Assigns prioritized tasks via bipartite greedy matching
        - Prevents multiple units from claiming the same tile or seed
        - Emits valid Kaggle action commands for all workers
        """
        my_farm = game_state.my_farm
        all_workers: List[UnitState] = my_farm.all_units

        farmer_action: List[str] = [FarmerAction.PASS.value]
        hands_actions: List[List[str]] = [[FarmerAction.PASS.value] for _ in my_farm.hands]

        # Tracking assigned positions and seeds
        claimed_tiles: Set[Tuple[int, int]] = set()
        local_seeds: Dict[str, int] = dict(my_farm.seeds)

        # Worker status: True if action already determined
        worker_busy: List[bool] = [False] * len(all_workers)

        # -------------------------------------------------------------
        # STEP 1: Handle backpack drops for workers carrying inventory
        # -------------------------------------------------------------
        for w_idx, worker in enumerate(all_workers):
            if worker.has_inventory:
                # If unit is orthogonally adjacent to shed, emit DROP
                if self.is_shed_adjacent(worker.pos):
                    act = [FarmerAction.DROP.value]
                    if w_idx == 0:
                        farmer_action = act
                    else:
                        hands_actions[w_idx - 1] = act
                    worker_busy[w_idx] = True
                else:
                    # Navigate towards nearest shed access tile
                    nearest_shed = self.get_nearest_shed_tile(worker.pos)
                    step = self.get_direction_to(worker.pos, nearest_shed)
                    act = [step] if step else [FarmerAction.PASS.value]
                    if w_idx == 0:
                        farmer_action = act
                    else:
                        hands_actions[w_idx - 1] = act
                    worker_busy[w_idx] = True

        # -------------------------------------------------------------
        # STEP 2: Generate and assign prioritized tasks to free workers
        # -------------------------------------------------------------
        unassigned_workers = [i for i, busy in enumerate(worker_busy) if not busy]

        if unassigned_workers:
            tasks = self.generate_daily_tasks(game_state, preferred_seed_order)

            # Group tasks by priority level (0, 1, 2, 3)
            for priority_level in (0, 1, 2, 3):
                level_tasks = [t for t in tasks if t.priority == priority_level and t.target_pos not in claimed_tiles]

                for task in level_tasks:
                    if not unassigned_workers:
                        break

                    # If this is a PLANT task, verify local seed stock
                    if task.task_type == FarmerAction.PLANT:
                        crop_name = task.payload
                        if local_seeds.get(crop_name, 0) <= 0:
                            continue  # No more seeds of this type available this turn

                    # Find the nearest unassigned worker to this task
                    best_w_idx: Optional[int] = None
                    best_dist = 999

                    for w_idx in unassigned_workers:
                        w_pos = all_workers[w_idx].pos
                        d = self.manhattan_distance(w_pos, task.target_pos)
                        if d < best_dist:
                            best_dist = d
                            best_w_idx = w_idx

                    if best_w_idx is not None:
                        worker = all_workers[best_w_idx]
                        unassigned_workers.remove(best_w_idx)
                        claimed_tiles.add(task.target_pos)

                        # Decrement seed stock if PLANT
                        if task.task_type == FarmerAction.PLANT:
                            local_seeds[task.payload] -= 1

                        # Determine worker command
                        if worker.pos == task.target_pos:
                            # At target: execute action
                            if task.task_type == FarmerAction.PLANT:
                                act = [FarmerAction.PLANT.value, task.payload]
                            else:
                                act = [task.task_type.value]
                        else:
                            # Away from target: move towards target
                            step = self.get_direction_to(worker.pos, task.target_pos)
                            act = [step] if step else [FarmerAction.PASS.value]

                        if best_w_idx == 0:
                            farmer_action = act
                        else:
                            hands_actions[best_w_idx - 1] = act

        # -------------------------------------------------------------
        # STEP 3: Remaining idle workers (no tasks available)
        # -------------------------------------------------------------
        for w_idx in unassigned_workers:
            worker = all_workers[w_idx]
            # Move towards center (4, 4) if far away, else PASS
            if worker.pos not in self.shed_tiles:
                nearest_shed = self.get_nearest_shed_tile(worker.pos)
                step = self.get_direction_to(worker.pos, nearest_shed)
                act = [step] if step else [FarmerAction.PASS.value]
            else:
                act = [FarmerAction.PASS.value]

            if w_idx == 0:
                farmer_action = act
            else:
                hands_actions[w_idx - 1] = act

        return {
            "farmer": farmer_action,
            "hands": hands_actions,
            "market": [],
        }
