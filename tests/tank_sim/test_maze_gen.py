"""程序化迷宫生成测试。"""

from tank_sim.config import TankConfig
from tank_sim.core.collision import tank_overlaps_wall
from tank_sim.core.maze_gen import _bfs_reachable, generate_maze, maze_to_ascii
from tank_sim.core.types import TankState


def test_generate_maze_connected_and_bordered():
    for seed in (0, 1, 7, 42, 99):
        m = generate_maze(cols=11, rows=7, cell_px=60, wall_thickness=4, seed=seed, openness=0.12)
        assert m.cols >= 5 and m.rows >= 5
        # 外框边墙
        assert all(m.h_walls[0][c] for c in range(m.cols))
        assert all(m.h_walls[m.rows][c] for c in range(m.cols))
        assert all(m.v_walls[r][0] for r in range(m.rows))
        assert all(m.v_walls[r][m.cols] for r in range(m.rows))
        # 全部格子可走（无 solid blocked）
        assert all(not m.blocked[y][x] for y in range(m.rows) for x in range(m.cols))
        rx = int(m.spawn_red[0] // m.cell_px)
        ry = int(m.spawn_red[1] // m.cell_px)
        bx = int(m.spawn_blue[0] // m.cell_px)
        by = int(m.spawn_blue[1] // m.cell_px)
        reach = _bfs_reachable(m, (rx, ry))
        assert (bx, by) in reach


def test_corridor_wide_enough_to_turn():
    """单格走廊净宽 > 车对角线，出生点可旋转。"""
    tank_cfg = TankConfig(
        width=20.0, height=28.0, speed_forward=2.24, angular_speed=0.065, collision_radius=12.0
    )
    import math

    diag = math.hypot(20.0, 28.0)
    m = generate_maze(cols=11, rows=7, cell_px=60, wall_thickness=4, seed=42)
    open_w = m.cell_px - m.wall_thickness
    assert open_w > diag
    for owner, pos, theta0 in (
        ("red", m.spawn_red, 0.0),
        ("blue", m.spawn_blue, 0.0),
    ):
        tank = TankState(x=pos[0], y=pos[1], theta=theta0, owner=owner)  # type: ignore[arg-type]
        for _ in range(30):
            tank.theta += tank_cfg.angular_speed
            assert not tank_overlaps_wall(tank, tank_cfg, m)


def test_spawns_clear_of_tank_obb():
    """出生点放置后，默认车体不应嵌墙。"""
    tank_cfg = TankConfig(
        width=20.0, height=28.0, speed_forward=2.24, angular_speed=0.065, collision_radius=12.0
    )
    for seed in (3, 11, 42):
        m = generate_maze(cols=11, rows=7, cell_px=60, wall_thickness=4, seed=seed)
        for owner, pos, theta in (
            ("red", m.spawn_red, 0.0),
            ("blue", m.spawn_blue, 3.141592653589793),
        ):
            tank = TankState(x=pos[0], y=pos[1], theta=theta, owner=owner)  # type: ignore[arg-type]
            assert not tank_overlaps_wall(tank, tank_cfg, m), f"seed={seed} {owner} nest wall"


def test_maze_ascii_contains_spawns():
    m = generate_maze(seed=42)
    text = maze_to_ascii(m)
    assert "R" in text and "B" in text
    assert "─" in text or "│" in text
