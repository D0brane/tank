"""双车随机出生（空场瞄准课程）。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tank_sim.config import TankConfig
from tank_sim.core.collision import tank_overlaps_wall, tanks_overlap
from tank_sim.core.map_loader import is_blocked, world_to_tile
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


def _free_cells(game_map: GameMap) -> list[tuple[int, int]]:
    """不可走格（#）以外的格子。"""
    cells: list[tuple[int, int]] = []
    for ty in range(game_map.rows):
        for tx in range(game_map.cols):
            if not is_blocked(game_map, tx, ty):
                cells.append((tx, ty))
    return cells


def _sample_in_cell(
    game_map: GameMap,
    tx: int,
    ty: int,
    rng: np.random.Generator,
    *,
    inset: float,
) -> tuple[float, float]:
    """在格子内采样一点，远离格边 inset，避免贴薄墙。"""
    cell = game_map.cell_px
    half = max(1.0, 0.5 * cell - inset)
    cx = (tx + 0.5) * cell
    cy = (ty + 0.5) * cell
    x = float(rng.uniform(cx - half, cx + half))
    y = float(rng.uniform(cy - half, cy + half))
    return x, y


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
    仅在可走格内采样两车位置与朝向，满足间距 ∈ [min_dist, max_dist]，
    且不嵌墙、互不重叠、不落在 # 格上。
    """
    if min_dist > max_dist:
        raise ValueError(f"min_dist ({min_dist}) > max_dist ({max_dist})")

    free = _free_cells(game_map)
    if len(free) < 2:
        raise ValueError("地图可走格不足，无法随机出生")

    inset = 0.5 * game_map.wall_thickness + 2.0

    for _ in range(max_tries):
        (rtx, rty), (btx, bty) = free[int(rng.integers(0, len(free)))], free[
            int(rng.integers(0, len(free)))
        ]
        rx, ry = _sample_in_cell(game_map, rtx, rty, rng, inset=inset)
        bx, by = _sample_in_cell(game_map, btx, bty, rng, inset=inset)
        # 采样后再次确认 tile 仍可走（数值边界）
        r_tile = world_to_tile(rx, ry, game_map.cell_px)
        b_tile = world_to_tile(bx, by, game_map.cell_px)
        if is_blocked(game_map, r_tile[0], r_tile[1]):
            continue
        if is_blocked(game_map, b_tile[0], b_tile[1]):
            continue
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

    # 回退：地图默认出生点（须可走）+ 随机朝向
    for side, pos in (("red", game_map.spawn_red), ("blue", game_map.spawn_blue)):
        tx, ty = world_to_tile(pos[0], pos[1], game_map.cell_px)
        if is_blocked(game_map, tx, ty):
            raise ValueError(f"默认出生点落在不可走格上: {side} {pos}")
    return DualSpawn(
        game_map.spawn_red[0],
        game_map.spawn_red[1],
        float(rng.uniform(0.0, 2.0 * math.pi)),
        game_map.spawn_blue[0],
        game_map.spawn_blue[1],
        float(rng.uniform(0.0, 2.0 * math.pi)),
    )
