"""A* 路径摘要特征（3 维），带缓存。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tank_sim.config import PlannerConfig
from tank_sim.core.types import GameMap, TankState
from tank_sim.planning.astar import astar


@dataclass
class PlannerCache:
    """规划器缓存，避免每帧全图 A*。"""

    last_step: int = -1
    last_enemy_x: float = 0.0
    last_enemy_y: float = 0.0
    features: tuple[float, float, float] = (0.0, 0.0, 0.0)


def compute_planner_features(
    observer: TankState,
    enemy: TankState,
    game_map: GameMap,
    planner_cfg: PlannerConfig,
    step: int,
    cache: PlannerCache,
) -> tuple[float, float, float]:
    """
    计算 3 维摘要：[局部下一步角度, 归一化路径长度, 可达标记]。

    不可达时返回 (0, 0, 0)。
    """
    need_refresh = (
        cache.last_step < 0
        or (step - cache.last_step) >= planner_cfg.refresh_every_steps
        or _dist(observer.x, observer.y, cache.last_enemy_x, cache.last_enemy_y)
        > planner_cfg.replan_distance_threshold
        or _dist(enemy.x, enemy.y, cache.last_enemy_x, cache.last_enemy_y)
        > planner_cfg.replan_distance_threshold
    )
    if not need_refresh:
        return cache.features

    path, reachable = astar(game_map, (observer.x, observer.y), (enemy.x, enemy.y))
    if not reachable or len(path) < 2:
        feat = (0.0, 0.0, 0.0)
    else:
        # 路径下一步 tile 中心
        nx, ny = path[1]
        wx = nx * game_map.cell_px + game_map.cell_px * 0.5
        wy = ny * game_map.cell_px + game_map.cell_px * 0.5
        world_angle = math.atan2(wy - observer.y, wx - observer.x)
        local_angle = _wrap_angle(world_angle - observer.theta)
        max_len = game_map.width * game_map.height
        norm_len = min(1.0, len(path) / max(max_len, 1))
        feat = (local_angle / math.pi, norm_len, 1.0)

    cache.last_step = step
    cache.last_enemy_x = enemy.x
    cache.last_enemy_y = enemy.y
    cache.features = feat
    return feat


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
