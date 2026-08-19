"""
Tactical Spatial Scheduler (Layer 3) for Kaggriculture.
Resolves multi-agent pathfinding (A*), prioritizes daily farm tasks (P0-P3),
and coordinates simultaneous actions for Farmer and Farm Hands without turn loss.
Handles livestock logistics (PICKUP -> PLACE), structure construction (BUILD_COOP, BUILD_PASTURE),
fertilizer routing (PICKUP -> FERTILIZE), and anti-collision deduplication.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from models.constants import (
    BOARD_SIZE,
    CropType,
    AnimalType,
    ProductType,
    StructureType,
    Direction,
    FarmerAction,
    SHED_ACCESS_TILES,
    SHED_CAPACITY,
    STRUCTURE_FOR_ANIMAL,
    BUILD_ACTION_FOR_STRUCTURE,
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
    priority: int  # 0=P0 (Critical), 1=P1 (Harvest/Drop/Fertilizer), 2=P2 (Maintenance/Fertilize), 3=P3 (Build/Place/Plant/Dig)
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

    @staticmethod
    def get_tile_quadrant(pos: Tuple[int, int]) -> str:
        """Determines the quadrant (NW, NE, SW, SE) for a given (x, y) coordinate."""
        x, y = pos
        if x < 5 and y < 5:
            return "NW"
        elif x >= 5 and y < 5:
            return "NE"
        elif x < 5 and y >= 5:
            return "SW"
        else:
            return "SE"

    @staticmethod
    def get_quadrant_center(quadrant: str) -> Tuple[int, int]:
        """Returns approximate center coordinate for a quadrant."""
        centers = {
            "NW": (2, 2),
            "NE": (7, 2),
            "SW": (2, 7),
            "SE": (7, 7),
        }
        return centers.get(quadrant, (2, 2))

    def assign_worker_clusters(self, game_state: GameState, tasks: List[Task]) -> Dict[int, str]:
        """
        Assigns each active worker (Farmer=0, Hands=1..N) to an unlocked quadrant
        based on pending task density and spatial proximity.
        """
        my_farm = game_state.my_farm
        unlocked_quads = list(my_farm.unlocked_quadrants)
        workers = my_farm.all_units

        # If only 1 quadrant unlocked, all workers assigned there
        if len(unlocked_quads) == 1:
            return {w.id: unlocked_quads[0] for w in workers}

        # Count tasks per unlocked quadrant
        quad_task_counts = {q: 0 for q in unlocked_quads}
        for task in tasks:
            q = self.get_tile_quadrant(task.target_pos)
            if q in quad_task_counts:
                quad_task_counts[q] += 1

        total_tasks = sum(quad_task_counts.values())
        if total_tasks == 0:
            # Distribute evenly among unlocked quadrants
            assignments: Dict[int, str] = {}
            for idx, w in enumerate(workers):
                assignments[w.id] = unlocked_quads[idx % len(unlocked_quads)]
            return assignments

        # Allocate quota of workers proportionally to task density
        assignments = {}
        remaining_workers = list(workers)

        # Sort quadrants by task count descending
        sorted_quads = sorted(unlocked_quads, key=lambda k: quad_task_counts[k], reverse=True)
        for q in sorted_quads:
            if not remaining_workers:
                break
            count = quad_task_counts[q]
            if count == 0:
                continue

            quota = int(round(len(workers) * (count / float(total_tasks))))
            quota = max(1, quota)

            assigned_count = 0
            while remaining_workers and assigned_count < quota:
                q_center = self.get_quadrant_center(q)
                best_w = min(remaining_workers, key=lambda w: self.manhattan_distance(w.pos, q_center))
                assignments[best_w.id] = q
                remaining_workers.remove(best_w)
                assigned_count += 1

        # Any leftover workers assigned to quadrant with highest tasks or closest center
        for w in remaining_workers:
            best_q = sorted_quads[0]
            assignments[w.id] = best_q

        return assignments

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
        fertilizer_targets: Optional[List[Tuple[int, int]]] = None,
        structure_build_orders: Optional[List[Tuple[str, Tuple[int, int]]]] = None,
    ) -> List[Task]:
        """
        Scans the farm grid and generates prioritized tasks:
        - P0: Critical survival (plants/animals dying tonight if ignored)
        - P1: Harvesting mature produce, animal yields, collecting fertilizer
        - P2: Routine daily maintenance (watering, feeding, caring, fertilizing)
        - P3: Structure construction (BUILD_COOP/BUILD_PASTURE), weed removal (DIG), and seeding (PLANT)
        """
        my_farm = game_state.my_farm
        current_day = game_state.day
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

        build_order_dict: Dict[Tuple[int, int], str] = {}
        if structure_build_orders:
            for struct_kind, pos in structure_build_orders:
                build_order_dict[pos] = struct_kind

        fert_target_set: Set[Tuple[int, int]] = set(fertilizer_targets or [])

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
                    # P1: Ready to harvest (only when plant reaches required maturity)
                    elif tile.is_mature(current_day):
                        tasks.append(Task(
                            task_type=FarmerAction.HARVEST,
                            target_pos=(x, y),
                            priority=1,
                        ))
                    # P2: Regular maintenance & fertilization
                    else:
                        if not tile.watered_today:
                            tasks.append(Task(
                                task_type=FarmerAction.WATER,
                                target_pos=(x, y),
                                priority=2,
                            ))
                        # P2: Fertilizer application on planned target
                        if (x, y) in fert_target_set and tile.fertilized_until_day < current_day:
                            tasks.append(Task(
                                task_type=FarmerAction.FERTILIZE,
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
                    # P1: Collect available fertilizer
                    if tile.fertilizer_available:
                        tasks.append(Task(
                            task_type=FarmerAction.COLLECT_FERTILIZER,
                            target_pos=(x, y),
                            priority=1,
                        ))
                    # P2: Maintenance (Feeding & Care)
                    if not tile.fed_today and not tile.is_in_danger:
                        tasks.append(Task(
                            task_type=FarmerAction.FEED,
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

                # 4. Empty Unlocked Tiles -> Construction or Planting
                elif isinstance(tile, EmptyTile):
                    if (x, y) in build_order_dict:
                        struct_kind = build_order_dict[(x, y)]
                        act = FarmerAction.BUILD_COOP if struct_kind == StructureType.COOP.value else FarmerAction.BUILD_PASTURE
                        tasks.append(Task(
                            task_type=act,
                            target_pos=(x, y),
                            priority=2,
                            payload=struct_kind,
                        ))
                    else:
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
        fertilizer_target_tiles: Optional[List[Tuple[int, int]]] = None,
        structure_build_orders: Optional[List[Tuple[str, Tuple[int, int]]]] = None,
    ) -> Dict[str, Any]:
        """
        Coordinates all workers (Farmer and Farm Hands):
        - Handles inventory dumping when backpack has items and shed has capacity
        - Coordinates livestock placement (PICKUP in shed -> PLACE in empty structure)
        - Coordinates fertilizer distribution (PICKUP in shed -> FERTILIZE on target crop)
        - Assigns prioritized tasks via bipartite greedy matching
        - Prevents multiple units from claiming the same animal, tile, or seed
        - Emits valid Kaggle action commands for all workers
        """
        my_farm = game_state.my_farm
        current_day = game_state.day
        all_workers: List[UnitState] = my_farm.all_units

        farmer_action: List[str] = [FarmerAction.PASS.value]
        hands_actions: List[List[str]] = [[FarmerAction.PASS.value] for _ in my_farm.hands]

        # Tracking assigned positions, animals, and resources
        claimed_tiles: Set[Tuple[int, int]] = set()
        claimed_feed_animals: Set[Tuple[int, int]] = set()
        claimed_care_animals: Set[Tuple[int, int]] = set()
        claimed_structures: Set[Tuple[int, int]] = set()
        local_seeds: Dict[str, int] = dict(my_farm.seeds)
        local_shed: Dict[str, int] = dict(my_farm.shed)

        # Worker status: True if action already determined
        worker_busy: List[bool] = [False] * len(all_workers)

        # -------------------------------------------------------------
        # STEP 1: Handle workers carrying inventory
        # -------------------------------------------------------------
        for w_idx, worker in enumerate(all_workers):
            if not worker.has_inventory:
                continue

            inv = worker.inventory
            # Case A: Worker is carrying an Animal (GOOSE, COW, SHEEP) -> route to empty structure
            carried_animal: Optional[str] = None
            for a_name in (AnimalType.GOOSE.value, AnimalType.COW.value, AnimalType.SHEEP.value):
                if inv.get(a_name, 0) > 0:
                    carried_animal = a_name
                    break

            if carried_animal is not None:
                req_struct_type = STRUCTURE_FOR_ANIMAL.get(carried_animal, StructureType.COOP.value)
                empty_structs = [
                    (x, y) for x, y, tile in my_farm.get_empty_structures(req_struct_type)
                    if (x, y) not in claimed_structures
                ]
                if empty_structs:
                    # Target closest empty structure
                    target_struct = min(empty_structs, key=lambda pos: self.manhattan_distance(worker.pos, pos))
                    claimed_structures.add(target_struct)

                    if worker.pos == target_struct:
                        act = [FarmerAction.PLACE.value, carried_animal, 1]
                    else:
                        step = self.get_direction_to(worker.pos, target_struct)
                        act = [step] if step else [FarmerAction.PASS.value]

                    if w_idx == 0:
                        farmer_action = act
                    else:
                        hands_actions[w_idx - 1] = act
                    worker_busy[w_idx] = True
                    continue

            # Case B: Worker is carrying Fertilizer -> route to target crop
            if inv.get(ProductType.FERTILIZER.value, 0) > 0:
                fert_targets = [
                    pos for pos in (fertilizer_target_tiles or [])
                    if pos not in claimed_tiles and isinstance(my_farm.get_tile(pos[0], pos[1]), PlantTile)
                    and my_farm.get_tile(pos[0], pos[1]).fertilized_until_day < current_day
                ]
                if not fert_targets:
                    # Fallback to any unfertilized plant
                    fert_targets = [
                        (x, y) for x, y, tile in my_farm.get_fertilizable_plants(current_day)
                        if (x, y) not in claimed_tiles
                    ]

                if fert_targets:
                    target_crop_pos = min(fert_targets, key=lambda pos: self.manhattan_distance(worker.pos, pos))
                    claimed_tiles.add(target_crop_pos)

                    if worker.pos == target_crop_pos:
                        act = [FarmerAction.FERTILIZE.value]
                    else:
                        step = self.get_direction_to(worker.pos, target_crop_pos)
                        act = [step] if step else [FarmerAction.PASS.value]

                    if w_idx == 0:
                        farmer_action = act
                    else:
                        hands_actions[w_idx - 1] = act
                    worker_busy[w_idx] = True
                    continue

            # Case C: Worker is carrying produce / items (Smart Backpack Retention)
            # - If adjacent to Shed (step cost = 0): DROP immediately.
            # - Otherwise: DO NOT waste 4-8 steps walking to Shed. Retain backpack contents
            #   and continue working; simulator auto-drops at Turn 23 for free!
            if self.is_shed_adjacent(worker.pos):
                act = [FarmerAction.DROP.value]
                if w_idx == 0:
                    farmer_action = act
                else:
                    hands_actions[w_idx - 1] = act
                worker_busy[w_idx] = True
            else:
                # Do not mark worker_busy; worker remains available for agricultural tasks
                pass

        # -------------------------------------------------------------
        # STEP 2: Handle Shed Pickup Logistics for Animal / Fertilizer
        # -------------------------------------------------------------
        unassigned_workers = [i for i, busy in enumerate(worker_busy) if not busy]

        # 2A. Check animal placement from Shed
        shed_animals = {
            k: v for k, v in local_shed.items()
            if k in (AnimalType.GOOSE.value, AnimalType.COW.value, AnimalType.SHEEP.value) and v > 0
        }

        for animal_name, qty in shed_animals.items():
            if not unassigned_workers or qty <= 0:
                break
            req_struct_type = STRUCTURE_FOR_ANIMAL.get(animal_name, StructureType.COOP.value)
            empty_structs = [
                (x, y) for x, y, tile in my_farm.get_empty_structures(req_struct_type)
                if (x, y) not in claimed_structures
            ]
            if not empty_structs:
                continue

            # Find closest unassigned worker to Shed
            best_w_idx = min(unassigned_workers, key=lambda i: self.manhattan_distance(all_workers[i].pos, self.get_nearest_shed_tile(all_workers[i].pos)))
            worker = all_workers[best_w_idx]
            unassigned_workers.remove(best_w_idx)
            worker_busy[best_w_idx] = True
            local_shed[animal_name] -= 1
            claimed_structures.add(empty_structs[0])

            if self.is_shed_adjacent(worker.pos):
                act = [FarmerAction.PICKUP.value, animal_name, 1]
            else:
                nearest_shed = self.get_nearest_shed_tile(worker.pos)
                step = self.get_direction_to(worker.pos, nearest_shed)
                act = [step] if step else [FarmerAction.PASS.value]

            if best_w_idx == 0:
                farmer_action = act
            else:
                hands_actions[best_w_idx - 1] = act

        # 2B. Check fertilizer pickup from Shed if target fertilizable crops exist
        shed_fert_qty = local_shed.get(ProductType.FERTILIZER.value, 0)
        unfertilized_crops = [
            pos for pos in (fertilizer_target_tiles or [])
            if pos not in claimed_tiles and isinstance(my_farm.get_tile(pos[0], pos[1]), PlantTile)
            and my_farm.get_tile(pos[0], pos[1]).fertilized_until_day < current_day
        ]

        if shed_fert_qty > 0 and unfertilized_crops and unassigned_workers:
            for _ in range(min(shed_fert_qty, len(unfertilized_crops), len(unassigned_workers))):
                if not unassigned_workers:
                    break
                best_w_idx = min(unassigned_workers, key=lambda i: self.manhattan_distance(all_workers[i].pos, self.get_nearest_shed_tile(all_workers[i].pos)))
                worker = all_workers[best_w_idx]
                unassigned_workers.remove(best_w_idx)
                worker_busy[best_w_idx] = True
                local_shed[ProductType.FERTILIZER.value] -= 1

                if self.is_shed_adjacent(worker.pos):
                    act = [FarmerAction.PICKUP.value, ProductType.FERTILIZER.value, 1]
                else:
                    nearest_shed = self.get_nearest_shed_tile(worker.pos)
                    step = self.get_direction_to(worker.pos, nearest_shed)
                    act = [step] if step else [FarmerAction.PASS.value]

                if best_w_idx == 0:
                    farmer_action = act
                else:
                    hands_actions[best_w_idx - 1] = act

        # -------------------------------------------------------------
        # STEP 3: Generate and assign prioritized tasks with Spatial Affinity
        # -------------------------------------------------------------
        if unassigned_workers:
            tasks = self.generate_daily_tasks(
                game_state,
                preferred_seed_order=preferred_seed_order,
                fertilizer_targets=fertilizer_target_tiles,
                structure_build_orders=structure_build_orders,
            )

            # Assign spatial cluster/quadrant to each worker
            worker_clusters = self.assign_worker_clusters(game_state, tasks)

            # Group tasks by priority level (0, 1, 2, 3)
            for priority_level in (0, 1, 2, 3):
                level_tasks = [t for t in tasks if t.priority == priority_level]

                for task in level_tasks:
                    if not unassigned_workers:
                        break

                    # Check deduplication rules
                    if task.task_type == FarmerAction.FEED and task.target_pos in claimed_feed_animals:
                        continue
                    if task.task_type == FarmerAction.CARE and task.target_pos in claimed_care_animals:
                        continue
                    if task.task_type not in (FarmerAction.FEED, FarmerAction.CARE) and task.target_pos in claimed_tiles:
                        continue

                    # If this is a PLANT task, verify local seed stock
                    if task.task_type == FarmerAction.PLANT:
                        crop_name = task.payload
                        if local_seeds.get(crop_name, 0) <= 0:
                            continue  # No more seeds of this type available this turn

                    task_quad = self.get_tile_quadrant(task.target_pos)

                    # Find best unassigned worker using Local Affinity & Boundary Crossing
                    best_w_idx: Optional[int] = None
                    best_score = 999

                    for w_idx in unassigned_workers:
                        worker_obj = all_workers[w_idx]
                        w_pos = worker_obj.pos
                        d = self.manhattan_distance(w_pos, task.target_pos)

                        # P0 tasks ignore affinity penalty (rush to save dying plants/animals)
                        # P1/P2/P3 add affinity penalty (+10) for crossing into other quadrants
                        w_quad = worker_clusters.get(worker_obj.id, "NW")
                        affinity_penalty = 0 if (priority_level == 0 or w_quad == task_quad) else 10
                        score = d + affinity_penalty

                        if score < best_score:
                            best_score = score
                            best_w_idx = w_idx

                    if best_w_idx is not None:
                        worker = all_workers[best_w_idx]
                        unassigned_workers.remove(best_w_idx)

                        if task.task_type == FarmerAction.FEED:
                            claimed_feed_animals.add(task.target_pos)
                        elif task.task_type == FarmerAction.CARE:
                            claimed_care_animals.add(task.target_pos)
                        else:
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
        # STEP 4: Remaining idle workers (no tasks available on board)
        # -------------------------------------------------------------
        for w_idx in unassigned_workers:
            worker = all_workers[w_idx]
            # If idle and far from shed, move towards center/shed; if at shed, PASS
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

