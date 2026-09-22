"""碰撞检测：OBB 车体、圆弹、边墙薄 AABB。"""

from __future__ import annotations

import math

from tank_sim.config import TankConfig
from tank_sim.core.geometry import (
    circle_overlaps_obb,
    circle_overlaps_rect,
    obb_overlaps_obb,
    obb_overlaps_rect,
    segment_all_aabb_edge_hits,
    tank_corners,
    tank_half_extents,
)
from tank_sim.core.types import BulletState, GameMap, TankState, WallRect


def _tank_hw_hh(tank_cfg: TankConfig) -> tuple[float, float]:
    return tank_half_extents(tank_cfg.width, tank_cfg.height)


def tank_obb_corners(tank: TankState, tank_cfg: TankConfig) -> list[tuple[float, float]]:
    """当前坦克 OBB 四顶点。"""
    hw, hh = _tank_hw_hh(tank_cfg)
    return tank_corners(tank.x, tank.y, tank.theta, hw, hh)


def _nearby_walls(game_map: GameMap, x: float, y: float, reach: float) -> list[WallRect]:
    """粗筛：与点周围 reach 可能相交的墙。"""
    out: list[WallRect] = []
    for w in game_map.wall_rects:
        if (
            w.left - reach <= x <= w.right + reach
            and w.top - reach <= y <= w.bottom + reach
        ):
            out.append(w)
    return out


def tank_overlaps_wall(tank: TankState, tank_cfg: TankConfig, game_map: GameMap) -> bool:
    """坦克旋转矩形是否与任一边墙重叠。"""
    corners = tank_obb_corners(tank, tank_cfg)
    hw, hh = _tank_hw_hh(tank_cfg)
    reach = math.hypot(hw, hh) + game_map.wall_thickness
    for w in _nearby_walls(game_map, tank.x, tank.y, reach):
        if obb_overlaps_rect(corners, w.left, w.top, w.right, w.bottom):
            return True
    return False


def tanks_overlap(a: TankState, b: TankState, tank_cfg: TankConfig) -> bool:
    """两坦克 OBB 是否重叠。"""
    if not a.alive or not b.alive:
        return False
    return obb_overlaps_obb(tank_obb_corners(a, tank_cfg), tank_obb_corners(b, tank_cfg))


def bullet_hits_tank(
    bullet: BulletState,
    tank: TankState,
    tank_cfg: TankConfig,
) -> bool:
    """圆弹 vs 旋转矩形车体。"""
    if not tank.alive:
        return False
    hw, hh = _tank_hw_hh(tank_cfg)
    corners = tank_corners(tank.x, tank.y, tank.theta, hw, hh)
    return circle_overlaps_obb(
        bullet.x,
        bullet.y,
        bullet.radius,
        corners,
        (tank.x, tank.y),
        tank.theta,
        hw,
        hh,
    )


def map_bounds(game_map: GameMap) -> tuple[float, float]:
    """地图世界宽高（像素）。"""
    return game_map.cols * game_map.cell_px, game_map.rows * game_map.cell_px


def clamp_bullet_to_map(
    x: float, y: float, radius: float, game_map: GameMap
) -> tuple[float, float, bool]:
    """
    将弹心钳制在地图内边距内。

    返回 (x, y, 是否发生了钳制)。
    """
    w, h = map_bounds(game_map)
    # 外框墙有厚度，内边约 wall_thickness/2
    margin = game_map.wall_thickness * 0.5 + radius
    nx = max(margin, min(x, w - margin))
    ny = max(margin, min(y, h - margin))
    return nx, ny, (nx != x or ny != y)


def clamp_tank_to_map(tank: TankState, tank_cfg: TankConfig, game_map: GameMap) -> bool:
    """
    将车心软钳制在地图内（无墙空场防开出世界）；不反弹。

    返回是否发生了钳制。
    """
    w, h = map_bounds(game_map)
    # 半对角 + 薄墙半厚，近似车体不越界
    half = 0.5 * math.hypot(tank_cfg.width, tank_cfg.height)
    margin = game_map.wall_thickness * 0.5 + half
    nx = max(margin, min(tank.x, w - margin))
    ny = max(margin, min(tank.y, h - margin))
    moved = nx != tank.x or ny != tank.y
    tank.x, tank.y = nx, ny
    return moved


def bullet_in_wall(x: float, y: float, radius: float, game_map: GameMap) -> bool:
    """弹心（考虑半径）是否与边墙相交。"""
    for w in _nearby_walls(game_map, x, y, radius + game_map.wall_thickness):
        if circle_overlaps_rect(x, y, radius, w.left, w.top, w.right, w.bottom):
            return True
    return False


def eject_bullet_from_wall(
    x: float,
    y: float,
    vx: float,
    vy: float,
    radius: float,
    game_map: GameMap,
) -> tuple[float, float, float, float, bool]:
    """
    若子弹嵌在墙内或越界，推出到空地并翻转移动方向分量。

    返回 (x, y, vx, vy, bounced)；``bounced`` 表示本函数因边界/嵌墙做了速度反射
    （无墙空场贴边钳制翻转移速也算反弹）。
    """
    bounced = False
    x, y, clamped = clamp_bullet_to_map(x, y, radius, game_map)
    if clamped:
        w, h = map_bounds(game_map)
        margin = game_map.wall_thickness * 0.5 + radius
        if x <= margin + 1e-6 and vx < 0:
            vx = -vx
            bounced = True
        if x >= w - margin - 1e-6 and vx > 0:
            vx = -vx
            bounced = True
        if y <= margin + 1e-6 and vy < 0:
            vy = -vy
            bounced = True
        if y >= h - margin - 1e-6 and vy > 0:
            vy = -vy
            bounced = True

    for _ in range(8):
        if not bullet_in_wall(x, y, radius, game_map):
            break
        # 推向最近墙中心的反方向
        best: WallRect | None = None
        best_d2 = 1e30
        for wall in _nearby_walls(game_map, x, y, radius + game_map.cell_px):
            if not circle_overlaps_rect(x, y, radius, wall.left, wall.top, wall.right, wall.bottom):
                continue
            d2 = (x - wall.cx) ** 2 + (y - wall.cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = wall
        if best is None:
            break
        dx, dy = x - best.cx, y - best.cy
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            dx = 1.0 if abs(vx) >= abs(vy) else 0.0
            dy = 0.0 if dx != 0.0 else 1.0
        # 沿墙的短轴推出
        ww = best.right - best.left
        wh = best.bottom - best.top
        if ww <= wh:
            # 竖墙：沿 x 推
            push = (ww * 0.5 + radius + 0.5) * (1.0 if dx >= 0 else -1.0)
            x = best.cx + push
            new_vx = abs(vx) * (1.0 if dx >= 0 else -1.0)
            if new_vx != vx:
                bounced = True
            vx = new_vx
        else:
            push = (wh * 0.5 + radius + 0.5) * (1.0 if dy >= 0 else -1.0)
            y = best.cy + push
            new_vy = abs(vy) * (1.0 if dy >= 0 else -1.0)
            if new_vy != vy:
                bounced = True
            vy = new_vy
        x, y, _ = clamp_bullet_to_map(x, y, radius, game_map)

    return x, y, vx, vy, bounced


def resolve_bullet_wall_step(
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    radius: float,
    game_map: GameMap,
    dt: float = 1.0,
) -> tuple[float, float, float, float, bool]:
    """
    子弹单子步：线段扫掠 + 墙边法线反射。

    对角点：若最早时刻附近同时撞到水平边与竖直边，则 vx、vy 同时翻转。
    返回 (x, y, vx, vy, bounced)；bounced 表示本子步发生了墙/边界反射。
    """
    bounced = False
    if bullet_in_wall(x0, y0, radius, game_map):
        x0, y0, vx, vy, ej = eject_bullet_from_wall(x0, y0, vx, vy, radius, game_map)
        bounced = bounced or ej

    x1 = x0 + vx * dt
    y1 = y0 + vy * dt

    candidates: list[tuple[float, float, float, float, float]] = []
    pad = radius + game_map.wall_thickness + abs(vx) * dt + abs(vy) * dt
    for wall in _nearby_walls(game_map, 0.5 * (x0 + x1), 0.5 * (y0 + y1), pad + game_map.cell_px):
        left = wall.left - radius
        top = wall.top - radius
        right = wall.right + radius
        bottom = wall.bottom + radius
        for t, ix, iy, nx, ny in segment_all_aabb_edge_hits(
            x0, y0, x1, y1, left, top, right, bottom
        ):
            if vx * nx + vy * ny >= -1e-9:
                continue
            candidates.append((t, ix, iy, nx, ny))

    if not candidates:
        x, y = x1, y1
    else:
        bounced = True
        candidates.sort(key=lambda c: c[0])
        t0 = candidates[0][0]
        corner_eps = 1e-3
        near = [c for c in candidates if c[0] <= t0 + corner_eps]
        flip_x = any(abs(c[3]) > 0.5 for c in near)
        flip_y = any(abs(c[4]) > 0.5 for c in near)

        ix, iy = candidates[0][1], candidates[0][2]
        if flip_x:
            vx = -vx
        if flip_y:
            vy = -vy

        nx = sum(c[3] for c in near)
        ny = sum(c[4] for c in near)
        nlen = math.hypot(nx, ny)
        if nlen < 1e-12:
            nx, ny = candidates[0][3], candidates[0][4]
            nlen = math.hypot(nx, ny) or 1.0
        nx, ny = nx / nlen, ny / nlen
        eps = max(0.35, radius * 0.15)
        x = ix + nx * eps
        y = iy + ny * eps

    x, y, vx, vy, ej = eject_bullet_from_wall(x, y, vx, vy, radius, game_map)
    bounced = bounced or ej
    return x, y, vx, vy, bounced
