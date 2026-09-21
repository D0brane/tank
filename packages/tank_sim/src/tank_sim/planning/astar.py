"""栅格 A* 路径规划（边墙连通）。"""

from __future__ import annotations

import heapq
from typing import Iterable

from tank_sim.core.map_loader import can_step, is_blocked, world_to_tile
from tank_sim.core.types import GameMap


def astar(
    game_map: GameMap,
    start_px: tuple[float, float],
    goal_px: tuple[float, float],
) -> tuple[list[tuple[int, int]], bool]:
    """
    在格子图上搜索路径（边墙阻挡邻接）。

    返回 (tile 路径列表, 是否可达)。路径含起点。
    """
    sx, sy = world_to_tile(start_px[0], start_px[1], game_map.cell_px)
    gx, gy = world_to_tile(goal_px[0], goal_px[1], game_map.cell_px)

    if is_blocked(game_map, sx, sy) or is_blocked(game_map, gx, gy):
        return [], False

    start = (sx, sy)
    goal = (gx, gy)
    if start == goal:
        return [start], True

    open_heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct(came_from, current), True

        for nb in _neighbors(current, game_map):
            tentative = g_score[current] + 1.0
            if nb not in g_score or tentative < g_score[nb]:
                came_from[nb] = current
                g_score[nb] = tentative
                f = tentative + _heuristic(nb, goal)
                heapq.heappush(open_heap, (f, nb))

    return [], False


def _neighbors(cell: tuple[int, int], game_map: GameMap) -> Iterable[tuple[int, int]]:
    x, y = cell
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if can_step(game_map, x, y, nx, ny):
            yield (nx, ny)


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _reconstruct(
    came_from: dict[tuple[int, int], tuple[int, int] | None],
    current: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path
