"""车体对齐的 7×7 局部墙栅格。"""

from __future__ import annotations

import math

from tank_sim.core.map_loader import point_hits_wall
from tank_sim.core.types import GameMap, TankState


def build_local_grid(observer: TankState, game_map: GameMap, size: int = 7) -> list[float]:
    """
    以观察者为原点、车体朝向为 +x 的局部栅格。

    采样点落在边墙 AABB 内则为 1，否则 0。
    """
    half = size // 2
    out: list[float] = []
    cos_t = math.cos(observer.theta)
    sin_t = math.sin(observer.theta)
    step = game_map.cell_px

    for ly in range(-half, half + 1):
        for lx in range(-half, half + 1):
            wx_off = lx * cos_t - ly * sin_t
            wy_off = lx * sin_t + ly * cos_t
            wx = observer.x + wx_off * step
            wy = observer.y + wy_off * step
            # 用半墙厚作采样半径，避免只打中墙缝
            hit = point_hits_wall(wx, wy, game_map, pad=game_map.wall_thickness * 0.5)
            out.append(1.0 if hit else 0.0)
    return out
