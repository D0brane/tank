"""地图加载与边墙构建。"""

from __future__ import annotations

from pathlib import Path

from tank_sim.core.types import GameMap, WallRect


def build_wall_rects(
    h_walls: list[list[bool]],
    v_walls: list[list[bool]],
    cell_px: float,
    wall_thickness: float,
) -> list[WallRect]:
    """
    将边墙标记转为薄 AABB。

    端点各外扩半墙厚，避免墙缝漏弹（与 JustDoIt / 官方 5px overlap 同思路）。
    """
    rows = len(v_walls)
    cols = len(h_walls[0]) if h_walls else 0
    ht = wall_thickness * 0.5
    rects: list[WallRect] = []

    for r in range(rows + 1):
        for c in range(cols):
            if not h_walls[r][c]:
                continue
            rects.append(
                WallRect(
                    left=c * cell_px - ht,
                    top=r * cell_px - ht,
                    right=(c + 1) * cell_px + ht,
                    bottom=r * cell_px + ht,
                )
            )

    for r in range(rows):
        for c in range(cols + 1):
            if not v_walls[r][c]:
                continue
            rects.append(
                WallRect(
                    left=c * cell_px - ht,
                    top=r * cell_px - ht,
                    right=c * cell_px + ht,
                    bottom=(r + 1) * cell_px + ht,
                )
            )
    return rects


def make_game_map(
    cols: int,
    rows: int,
    cell_px: float,
    wall_thickness: float,
    h_walls: list[list[bool]],
    v_walls: list[list[bool]],
    spawn_red: tuple[float, float],
    spawn_blue: tuple[float, float],
    blocked: list[list[bool]] | None = None,
) -> GameMap:
    """组装 GameMap 并缓存 wall_rects。"""
    if blocked is None:
        blocked = [[False for _ in range(cols)] for _ in range(rows)]
    return GameMap(
        cols=cols,
        rows=rows,
        cell_px=cell_px,
        wall_thickness=wall_thickness,
        h_walls=h_walls,
        v_walls=v_walls,
        blocked=blocked,
        wall_rects=build_wall_rects(h_walls, v_walls, cell_px, wall_thickness),
        spawn_red=spawn_red,
        spawn_blue=spawn_blue,
    )


def load_map(
    path: str | Path,
    cell_px: float,
    wall_thickness: float = 4.0,
) -> GameMap:
    """
    加载 ASCII 地图并转为边墙模型。

    字符：`#` 不可走格；`.`/空格 空地；`R`/`B` 出生点。
    空地与不可走格（或越界）相邻处生成薄墙；外框强制封边。
    """
    p = Path(path)
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"地图为空: {p}")

    rows = len(lines)
    cols = max(len(ln) for ln in lines)
    blocked: list[list[bool]] = []
    spawn_red: tuple[float, float] | None = None
    spawn_blue: tuple[float, float] | None = None

    for y, row in enumerate(lines):
        row = row.ljust(cols, ".")
        blocked_row: list[bool] = []
        for x, ch in enumerate(row):
            if ch == "#":
                blocked_row.append(True)
            elif ch in ("R", "r"):
                blocked_row.append(False)
                spawn_red = cell_center(x, y, cell_px)
            elif ch in ("B", "b"):
                blocked_row.append(False)
                spawn_blue = cell_center(x, y, cell_px)
            else:
                blocked_row.append(False)
        blocked.append(blocked_row)

    if spawn_red is None or spawn_blue is None:
        raise ValueError(f"地图必须包含 R 与 B 出生点: {p}")

    h_walls, v_walls = edges_from_blocked(blocked, seal_border=True)
    gm = make_game_map(
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
    _assert_spawns_connected(gm, path=str(p))
    return gm


def _assert_spawns_connected(game_map: GameMap, path: str = "") -> None:
    """红蓝出生点必须在同一连通分量（否则对局无法交火相遇）。"""
    from collections import deque

    rx, ry = world_to_tile(game_map.spawn_red[0], game_map.spawn_red[1], game_map.cell_px)
    bx, by = world_to_tile(game_map.spawn_blue[0], game_map.spawn_blue[1], game_map.cell_px)
    if is_blocked(game_map, rx, ry) or is_blocked(game_map, bx, by):
        raise ValueError(f"出生点落在不可走格上: {path} red=({rx},{ry}) blue=({bx},{by})")

    q = deque([(rx, ry)])
    seen = {(rx, ry)}
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (nx, ny) in seen:
                continue
            if can_step(game_map, x, y, nx, ny):
                seen.add((nx, ny))
                q.append((nx, ny))
    if (bx, by) not in seen:
        raise ValueError(
            f"地图红蓝出生点不连通（中间被墙隔开）: {path} "
            f"red=({rx},{ry}) blue=({bx},{by})"
        )


def edges_from_blocked(
    blocked: list[list[bool]],
    *,
    seal_border: bool = True,
) -> tuple[list[list[bool]], list[list[bool]]]:
    """
    由不可走格推导边墙：空地贴墙/越界处立薄墙。

    ``seal_border=True``（默认）强制外框，与 ASCII ``load_map`` / 迷宫一致。
    ``seal_border=False`` 时不封外框；若 ``blocked`` 全 False，则无任何墙。
    """
    rows = len(blocked)
    cols = len(blocked[0])
    h_walls = [[False for _ in range(cols)] for _ in range(rows + 1)]
    v_walls = [[False for _ in range(cols + 1)] for _ in range(rows)]

    def is_block(x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= cols or y >= rows:
            # 无外框时，越界不视为墙；有外框时越界当墙以生成边界边
            return bool(seal_border)
        return blocked[y][x]

    for y in range(rows):
        for x in range(cols):
            if blocked[y][x]:
                continue
            if is_block(x - 1, y):
                v_walls[y][x] = True
            if is_block(x + 1, y):
                v_walls[y][x + 1] = True
            if is_block(x, y - 1):
                h_walls[y][x] = True
            if is_block(x, y + 1):
                h_walls[y + 1][x] = True

    if seal_border:
        # 外框（即使角上是 # 也封死世界边界）
        for c in range(cols):
            h_walls[0][c] = True
            h_walls[rows][c] = True
        for r in range(rows):
            v_walls[r][0] = True
            v_walls[r][cols] = True

    return h_walls, v_walls


def generate_open_arena(
    cols: int,
    rows: int,
    cell_px: float,
    wall_thickness: float = 4.0,
) -> GameMap:
    """
    无内墙、有外框的空场（瞄准课程）。

    世界边界为真正的边墙：碰撞 / 子弹反弹 / 雷达 / 贴墙惩罚与有墙地图一致。
    出生点占位为中心左右；局内通常用 random_spawn 覆盖。
    """
    if cols < 2 or rows < 1:
        raise ValueError(f"open arena 尺寸无效: cols={cols} rows={rows}")
    blocked = [[False for _ in range(cols)] for _ in range(rows)]
    h_walls, v_walls = edges_from_blocked(blocked, seal_border=True)
    mid_y = (rows - 1) * 0.5
    spawn_red = cell_center(1, int(mid_y), cell_px)
    spawn_blue = cell_center(cols - 2, int(mid_y), cell_px)
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


def cell_center(tx: int, ty: int, cell_px: float) -> tuple[float, float]:
    """格子中心的世界坐标（像素）。"""
    return (tx * cell_px + cell_px * 0.5, ty * cell_px + cell_px * 0.5)


def world_to_tile(x: float, y: float, cell_px: float) -> tuple[int, int]:
    """世界坐标 → 格子索引。"""
    return int(x // cell_px), int(y // cell_px)


def is_blocked(game_map: GameMap, tx: int, ty: int) -> bool:
    """格子是否不可走（越界或 ASCII #）。"""
    if ty < 0 or ty >= game_map.rows or tx < 0 or tx >= game_map.cols:
        return True
    return game_map.blocked[ty][tx]


def is_wall(game_map: GameMap, tx: int, ty: int) -> bool:
    """兼容旧名：不可走格。边墙不占格。"""
    return is_blocked(game_map, tx, ty)


def can_step(game_map: GameMap, x0: int, y0: int, x1: int, y1: int) -> bool:
    """相邻格子之间是否可通行（无被挡且中间无边墙）。"""
    if is_blocked(game_map, x1, y1):
        return False
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) + abs(dy) != 1:
        return False
    if dx == 1:
        return not game_map.v_walls[y0][x0 + 1]
    if dx == -1:
        return not game_map.v_walls[y0][x0]
    if dy == 1:
        return not game_map.h_walls[y0 + 1][x0]
    if dy == -1:
        return not game_map.h_walls[y0][x0]
    return False


def point_hits_wall(x: float, y: float, game_map: GameMap, pad: float = 0.0) -> bool:
    """点（可带半径 pad）是否与任一边墙 AABB 相交。"""
    for w in game_map.wall_rects:
        if (
            w.left - pad <= x <= w.right + pad
            and w.top - pad <= y <= w.bottom + pad
        ):
            return True
    return False
