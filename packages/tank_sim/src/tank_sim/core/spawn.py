"""双车随机出生（空场瞄准课程）。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tank_sim.config import TankConfig
from tank_sim.core.collision import tank_overlaps_wall, tanks_overlap
from tank_sim.core.types import GameMap, TankState


@dataclass(frozen=True)
class DualSpawn:
    """红蓝出生位姿。"""

    red_x: float
    red_y: float
    red_theta: float
    blue_x: float
    blue_y: float
    blue_theta: float


def sample_dual_spawn(
    game_map: GameMap,
    tank_cfg: TankConfig,
    rng: np.random.Generator,
    *,
    min_dist: float,
    max_dist: float,
    max_tries: int = 400,
) -> DualSpawn:
    """
    均匀采样两车位置与朝向，满足间距 ∈ [min_dist, max_dist]，且不嵌墙、互不重叠。
    """
    if min_dist > max_dist:
        raise ValueError(f"min_dist ({min_dist}) > max_dist ({max_dist})")

    margin = 0.5 * math.hypot(tank_cfg.width, tank_cfg.height) + game_map.wall_thickness + 4.0
    width = game_map.cols * game_map.cell_px
    height = game_map.rows * game_map.cell_px
    lo_x, hi_x = margin, width - margin
    lo_y, hi_y = margin, height - margin
    if hi_x <= lo_x or hi_y <= lo_y:
        raise ValueError("地图过小，无法随机出生")

    for _ in range(max_tries):
        rx = float(rng.uniform(lo_x, hi_x))
        ry = float(rng.uniform(lo_y, hi_y))
        bx = float(rng.uniform(lo_x, hi_x))
        by = float(rng.uniform(lo_y, hi_y))
        dist = math.hypot(rx - bx, ry - by)
        if dist < min_dist or dist > max_dist:
            continue
        rth = float(rng.uniform(0.0, 2.0 * math.pi))
        bth = float(rng.uniform(0.0, 2.0 * math.pi))
        red = TankState(x=rx, y=ry, theta=rth, owner="red")
        blue = TankState(x=bx, y=by, theta=bth, owner="blue")
        if tank_overlaps_wall(red, tank_cfg, game_map):
            continue
        if tank_overlaps_wall(blue, tank_cfg, game_map):
            continue
        if tanks_overlap(red, blue, tank_cfg):
            continue
        return DualSpawn(rx, ry, rth, bx, by, bth)

    # 回退：地图默认出生点 + 随机朝向
    return DualSpawn(
        game_map.spawn_red[0],
        game_map.spawn_red[1],
        float(rng.uniform(0.0, 2.0 * math.pi)),
        game_map.spawn_blue[0],
        game_map.spawn_blue[1],
        float(rng.uniform(0.0, 2.0 * math.pi)),
    )
