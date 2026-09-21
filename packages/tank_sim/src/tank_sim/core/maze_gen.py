"""程序化迷宫：格子间边墙（Prim 风格打通）。"""

from __future__ import annotations

import random
from collections import deque

from tank_sim.core.map_loader import can_step, cell_center, make_game_map
from tank_sim.core.types import GameMap


def generate_maze(
    cols: int = 11,
    rows: int = 7,
    cell_px: float = 60.0,
    wall_thickness: float = 4.0,
    seed: int | None = None,
    openness: float = 0.12,
    *,
    tile_px: float | None = None,
) -> GameMap:
    """
    生成边墙迷宫：所有格可走，墙为格子分割线；外框封闭。

    - cols/rows: 逻辑格子数（TT2 经典约 11×7）
    - openness: 额外拆除内墙比例
    - tile_px: 兼容旧参数名，等同 cell_px
    """
    if tile_px is not None:
        cell_px = float(tile_px)
    cols = max(int(cols), 5)
    rows = max(int(rows), 5)

    rng = random.Random(seed)
    # 初始：所有边都有墙
    h_walls = [[True for _ in range(cols)] for _ in range(rows + 1)]
    v_walls = [[True for _ in range(cols + 1)] for _ in range(rows)]

    _carve_prim(h_walls, v_walls, cols, rows, rng)
    _knock_inner_walls(h_walls, v_walls, cols, rows, rng, openness)

    # 外框必须保留
    for c in range(cols):
        h_walls[0][c] = True
        h_walls[rows][c] = True
    for r in range(rows):
        v_walls[r][0] = True
        v_walls[r][cols] = True

    blocked = [[False for _ in range(cols)] for _ in range(rows)]
    spawn_red, spawn_blue = _place_spawns(h_walls, v_walls, cols, rows, cell_px, rng)
    return make_game_map(
        cols=cols,
        rows=rows,
        cell_px=cell_px,
        wall_thickness=wall_thickness,
        h_walls=h_walls,
        v_walls=v_walls,
        spawn_red=spawn_red,
        spawn_blue=spawn_blue,
        blocked=blocked,
    )


def maze_to_ascii(game_map: GameMap) -> str:
    """边墙迷宫 ASCII 预览（+─│ 与格子）。"""
    rx = int(game_map.spawn_red[0] // game_map.cell_px)
    ry = int(game_map.spawn_red[1] // game_map.cell_px)
    bx = int(game_map.spawn_blue[0] // game_map.cell_px)
    by = int(game_map.spawn_blue[1] // game_map.cell_px)
    cols, rows = game_map.cols, game_map.rows
    lines: list[str] = []

    for r in range(rows + 1):
        # 水平墙行
        row_chars: list[str] = ["+"]
        for c in range(cols):
            row_chars.append("───" if game_map.h_walls[r][c] else "   ")
            row_chars.append("+")
        lines.append("".join(row_chars))
        if r >= rows:
            break
        # 格子行
        cell_chars: list[str] = []
        for c in range(cols):
            cell_chars.append("│" if game_map.v_walls[r][c] else " ")
            if game_map.blocked[r][c]:
                mark = "###"
            elif (c, r) == (rx, ry):
                mark = " R "
            elif (c, r) == (bx, by):
                mark = " B "
            else:
                mark = "   "
            cell_chars.append(mark)
        cell_chars.append("│" if game_map.v_walls[r][cols] else " ")
        lines.append("".join(cell_chars))
    return "\n".join(lines)


def _carve_prim(
    h_walls: list[list[bool]],
    v_walls: list[list[bool]],
    cols: int,
    rows: int,
    rng: random.Random,
) -> None:
    """Prim：从 (0,0) 扩展，打通边墙形成生成树迷宫。"""
    in_maze = [[False for _ in range(cols)] for _ in range(rows)]
    in_maze[0][0] = True
    # 前沿：(x,y, nx,ny) 从已访问指向候选
    frontier: list[tuple[int, int, int, int]] = []

    def add_frontier(x: int, y: int) -> None:
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < cols and 0 <= ny < rows and not in_maze[ny][nx]:
                frontier.append((x, y, nx, ny))

    add_frontier(0, 0)

    while frontier:
        i = rng.randrange(len(frontier))
        x, y, nx, ny = frontier.pop(i)
        if in_maze[ny][nx]:
            continue
        _knock_edge(h_walls, v_walls, x, y, nx, ny)
        in_maze[ny][nx] = True
        add_frontier(nx, ny)


def _knock_edge(
    h_walls: list[list[bool]],
    v_walls: list[list[bool]],
    x: int,
    y: int,
    nx: int,
    ny: int,
) -> None:
    if nx == x + 1 and ny == y:
        v_walls[y][x + 1] = False
    elif nx == x - 1 and ny == y:
        v_walls[y][x] = False
    elif ny == y + 1 and nx == x:
        h_walls[y + 1][x] = False
    elif ny == y - 1 and nx == x:
        h_walls[y][x] = False


def _knock_inner_walls(
    h_walls: list[list[bool]],
    v_walls: list[list[bool]],
    cols: int,
    rows: int,
    rng: random.Random,
    openness: float,
) -> None:
    """随机拆除部分内墙，增加开阔度。"""
    # 内部水平墙：行 1..rows-1
    for r in range(1, rows):
        for c in range(cols):
            if h_walls[r][c] and rng.random() < openness:
                h_walls[r][c] = False
    # 内部竖直墙：列 1..cols-1
    for r in range(rows):
        for c in range(1, cols):
            if v_walls[r][c] and rng.random() < openness:
                v_walls[r][c] = False


def _list_cells(cols: int, rows: int, blocked: list[list[bool]] | None = None) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for y in range(rows):
        for x in range(cols):
            if blocked is None or not blocked[y][x]:
                out.append((x, y))
    return out


def _bfs_reachable_edges(
    h_walls: list[list[bool]],
    v_walls: list[list[bool]],
    cols: int,
    rows: int,
    start: tuple[int, int],
    blocked: list[list[bool]] | None = None,
) -> set[tuple[int, int]]:
    """边墙感知 BFS。"""
    gm = GameMap(
        cols=cols,
        rows=rows,
        cell_px=1.0,
        wall_thickness=0.1,
        h_walls=h_walls,
        v_walls=v_walls,
        blocked=blocked or [[False] * cols for _ in range(rows)],
        wall_rects=[],
        spawn_red=(0.0, 0.0),
        spawn_blue=(0.0, 0.0),
    )
    q = deque([start])
    seen = {start}
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (nx, ny) in seen:
                continue
            if can_step(gm, x, y, nx, ny):
                seen.add((nx, ny))
                q.append((nx, ny))
    return seen


def _free_cells(game_map: GameMap) -> list[tuple[int, int]]:
    """可走格子列表（供预览）。"""
    return _list_cells(game_map.cols, game_map.rows, game_map.blocked)


def _bfs_reachable(game_map: GameMap, start: tuple[int, int]) -> set[tuple[int, int]]:
    """对 GameMap 的边墙 BFS（供测试 / 预览）。"""
    return _bfs_reachable_edges(
        game_map.h_walls,
        game_map.v_walls,
        game_map.cols,
        game_map.rows,
        start,
        game_map.blocked,
    )


def _place_spawns(
    h_walls: list[list[bool]],
    v_walls: list[list[bool]],
    cols: int,
    rows: int,
    cell_px: float,
    rng: random.Random,
) -> tuple[tuple[float, float], tuple[float, float]]:
    free = _list_cells(cols, rows)
    left = [c for c in free if c[0] <= cols // 3] or free[:]
    right = [c for c in free if c[0] >= (2 * cols) // 3] or free[:]

    best: tuple[tuple[int, int], tuple[int, int], int] | None = None
    for _ in range(48):
        a = rng.choice(left)
        reach = _bfs_reachable_edges(h_walls, v_walls, cols, rows, a)
        candidates = [c for c in right if c in reach and c != a]
        if not candidates:
            candidates = [c for c in free if c in reach and c != a]
        if not candidates:
            continue
        b = max(candidates, key=lambda c: abs(c[0] - a[0]) + abs(c[1] - a[1]))
        dist = abs(b[0] - a[0]) + abs(b[1] - a[1])
        if best is None or dist > best[2]:
            best = (a, b, dist)

    if best is None:
        a = free[0]
        reach = _bfs_reachable_edges(h_walls, v_walls, cols, rows, a)
        others = [c for c in free if c in reach and c != a]
        if not others:
            raise RuntimeError("迷宫不连通，无法放置双方出生点")
        b = others[-1]
    else:
        a, b, _ = best

    return cell_center(a[0], a[1], cell_px), cell_center(b[0], b[1], cell_px)

