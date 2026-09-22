"""车体坐标系 8 向墙雷达（射线测距）。"""

from __future__ import annotations

import math

from tank_sim.core.collision import map_bounds
from tank_sim.core.geometry import segment_all_aabb_edge_hits
from tank_sim.core.types import GameMap, TankState

_EPS_T = 1e-6


def min_wall_distance_px(
    x: float, y: float, game_map: GameMap, n_rays: int = 8
) -> float:
    """
    车心到最近障碍的距离（像素）：边墙 AABB ∪ 地图世界边界。

    无墙空场（``wall_rects`` 为空）时退化为到四边边界的距离，否则贴边
    软钳制无法触发贴墙惩罚。``n_rays`` 仅兼容旧调用，忽略。
    """
    del n_rays
    w, h = map_bounds(game_map)
    # 到世界边界（四边）
    best = min(x, y, w - x, h - y)
    best = max(0.0, best)
    for wall in game_map.wall_rects:
        dx = 0.0
        if x < wall.left:
            dx = wall.left - x
        elif x > wall.right:
            dx = x - wall.right
        dy = 0.0
        if y < wall.top:
            dy = wall.top - y
        elif y > wall.bottom:
            dy = y - wall.bottom
        if dx == 0.0 and dy == 0.0:
            return 0.0
        d = math.hypot(dx, dy)
        if d < best:
            best = d
    return best


def build_wall_radar(
    observer: TankState, game_map: GameMap, n_rays: int = 8
) -> list[float]:
    """
    沿车体朝向每 360°/n_rays 一条射线，测到最近墙的距离。

    尺度：scale = 0.5 * max(W, H)（房间较长边的一半 = 1）。
    未命中：max_range / scale，max_range = max(W, H)（即约 2.0）。
    角序：k=0 正前（车头），逆时针。
    """
    if n_rays < 1:
        raise ValueError("wall_radar_rays 必须 ≥ 1")
    w, h = map_bounds(game_map)
    longer = max(w, h)
    scale = 0.5 * longer
    max_range = longer
    miss = max_range / scale

    out: list[float] = []
    step = (2.0 * math.pi) / n_rays
    ox, oy = observer.x, observer.y
    for k in range(n_rays):
        ang = observer.theta + k * step
        dx = math.cos(ang)
        dy = math.sin(ang)
        x1 = ox + dx * max_range
        y1 = oy + dy * max_range
        best_t: float | None = None
        for wall in game_map.wall_rects:
            hits = segment_all_aabb_edge_hits(
                ox, oy, x1, y1, wall.left, wall.top, wall.right, wall.bottom
            )
            for t, *_rest in hits:
                if t <= _EPS_T:
                    continue
                if best_t is None or t < best_t:
                    best_t = t
        if best_t is None:
            out.append(miss)
        else:
            out.append((best_t * max_range) / scale)
    return out
