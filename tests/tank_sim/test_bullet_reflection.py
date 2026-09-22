"""子弹墙反射 golden 测试。"""

import math

from tank_sim.core.collision import map_bounds, resolve_bullet_wall_step
from tank_sim.core.map_loader import edges_from_blocked, load_map, make_game_map
from tank_sim.core.types import GameMap


def _make_single_cell_room(cell: float = 60.0, wall: float = 4.0) -> GameMap:
    """单格封闭房间（四周边墙）。"""
    cols, rows = 1, 1
    h_walls = [[True], [True]]
    v_walls = [[True, True]]
    return make_game_map(
        cols=cols,
        rows=rows,
        cell_px=cell,
        wall_thickness=wall,
        h_walls=h_walls,
        v_walls=v_walls,
        spawn_red=(cell * 0.5, cell * 0.5),
        spawn_blue=(cell * 0.5, cell * 0.5),
    )


def _make_open_room(cols: int = 4, rows: int = 4, cell: float = 60.0, wall: float = 4.0) -> GameMap:
    """仅外框边墙的开阔房间。"""
    h_walls = [[False for _ in range(cols)] for _ in range(rows + 1)]
    v_walls = [[False for _ in range(cols + 1)] for _ in range(rows)]
    for c in range(cols):
        h_walls[0][c] = True
        h_walls[rows][c] = True
    for r in range(rows):
        v_walls[r][0] = True
        v_walls[r][cols] = True
    return make_game_map(
        cols=cols,
        rows=rows,
        cell_px=cell,
        wall_thickness=wall,
        h_walls=h_walls,
        v_walls=v_walls,
        spawn_red=(cell * 1.5, cell * 1.5),
        spawn_blue=(cell * 2.5, cell * 2.5),
    )


def test_horizontal_bullet_reflects_on_right_wall():
    """向右飞的子弹撞右墙后 vx 反向。"""
    gm = _make_single_cell_room()
    x, y = 30.0, 30.0
    vx, vy = 6.0, 0.0
    for _ in range(40):
        x, y, vx, vy, _b = resolve_bullet_wall_step(x, y, vx, vy, radius=2.0, game_map=gm, dt=1.0)
        if vx < 0:
            break
    assert vx < 0
    assert abs(vy) < 1e-6


def test_vertical_bullet_reflects_on_bottom_wall():
    gm = _make_single_cell_room()
    x, y = 30.0, 30.0
    vx, vy = 0.0, 6.0
    for _ in range(40):
        x, y, vx, vy, _b = resolve_bullet_wall_step(x, y, vx, vy, radius=2.0, game_map=gm, dt=1.0)
        if vy < 0:
            break
    assert vy < 0
    assert abs(vx) < 1e-6


def test_empty_map_file_bullet_survives():
    """真实 empty 地图上子弹沿空地飞行若干步不崩。"""
    gm = load_map("assets/maps/empty.txt", cell_px=60, wall_thickness=4)
    x, y = gm.spawn_red[0] + 40.0, gm.spawn_red[1]
    vx, vy = 3.77, 0.0
    for _ in range(30):
        x, y, vx, vy, _b = resolve_bullet_wall_step(x, y, vx, vy, radius=2.5, game_map=gm, dt=0.5)
    assert math.isfinite(x) and math.isfinite(y)


def test_bullet_stays_inside_map_bounds():
    """多角度高速子弹不得飞出地图外框。"""
    gm = load_map("assets/maps/empty.txt", cell_px=60, wall_thickness=4)
    w, h = map_bounds(gm)
    radius = 2.5
    margin = gm.wall_thickness * 0.5 + radius
    angles = [0.0, 0.3, 0.7, 1.1, 1.57, 2.2, 2.8, 3.5, 4.2, 5.0, 5.8]
    for ang in angles:
        x, y = gm.spawn_red[0], gm.spawn_red[1]
        vx, vy = 8.0 * math.cos(ang), 8.0 * math.sin(ang)
        for _ in range(200):
            x, y, vx, vy, _b = resolve_bullet_wall_step(
                x, y, vx, vy, radius=radius, game_map=gm, dt=0.5
            )
            assert margin - 1e-2 <= x <= w - margin + 1e-2
            assert margin - 1e-2 <= y <= h - margin + 1e-2


def test_diagonal_into_inner_corner_bounces_both_axes():
    """对角线冲进外框内角：vx、vy 都应翻转。"""
    gm = _make_open_room(cols=3, rows=3, cell=60, wall=4)
    # 从中心射向左上角
    x, y = 90.0, 90.0
    speed = 6.0
    vx = vy = -speed
    saw_flip = False
    for _ in range(80):
        ovx, ovy = vx, vy
        x, y, vx, vy, _b = resolve_bullet_wall_step(x, y, vx, vy, radius=2.0, game_map=gm, dt=1.0)
        if vx > 0 and vy > 0 and (ovx < 0 or ovy < 0):
            saw_flip = True
            break
        w, h = map_bounds(gm)
        assert 0 <= x <= w and 0 <= y <= h
    assert saw_flip, f"角点未双轴反弹，最终 v=({vx},{vy}) pos=({x},{y})"


def test_ascii_map_converts_to_edge_walls():
    """ASCII # 转为边墙后，空地格可走且有外框。"""
    blocked = [
        [True, True, True],
        [True, False, True],
        [True, True, True],
    ]
    h, v = edges_from_blocked(blocked)
    assert h[1][1] and h[2][1]  # 空地上下边
    assert v[1][1] and v[1][2]  # 空地左右边
