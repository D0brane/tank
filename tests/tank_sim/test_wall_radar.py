"""车体 8 向墙雷达：细边墙用射线-AABB 交点测距。"""

from tank_sim.config import default_config_path, load_env_config
from tank_sim.core.collision import map_bounds
from tank_sim.core.types import TankState
from tank_sim.core.world import create_initial_state
from tank_sim.observation.wall_radar import build_wall_radar


def test_facing_outer_wall_forward_shorter():
    """贴右墙、车头朝右：0° 射线明显短于侧向/后方。"""
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    gm = state.game_map
    tank = TankState(
        x=gm.cols * gm.cell_px - gm.cell_px * 1.2,
        y=gm.rows * gm.cell_px * 0.5,
        theta=0.0,
        owner="red",
    )
    radar = build_wall_radar(tank, gm, n_rays=8)
    assert radar[0] < radar[2]
    assert radar[0] < radar[4]
    assert radar[0] < 0.5


def test_open_room_center_scale():
    """方形空房中心：各向接近半短边 / scale。"""
    from tank_sim.core.map_loader import make_game_map

    cols, rows, cell, wall = 11, 11, 60.0, 4.0
    h = [[False] * cols for _ in range(rows + 1)]
    v = [[False] * (cols + 1) for _ in range(rows)]
    for c in range(cols):
        h[0][c] = True
        h[rows][c] = True
    for r in range(rows):
        v[r][0] = True
        v[r][cols] = True
    gm = make_game_map(
        cols=cols,
        rows=rows,
        cell_px=cell,
        wall_thickness=wall,
        h_walls=h,
        v_walls=v,
        spawn_red=(cell * 5.5, cell * 5.5),
        spawn_blue=(cell * 6.5, cell * 5.5),
    )
    tank = TankState(x=cell * 5.5, y=cell * 5.5, theta=0.0, owner="red")
    radar = build_wall_radar(tank, gm, n_rays=8)
    w, h_px = map_bounds(gm)
    scale = 0.5 * max(w, h_px)
    # 中心到边框内缘约半边；厚度使有效距离略小于 half
    half_short = 0.5 * min(w, h_px)
    expected = half_short / scale
    for d in radar[0], radar[2], radar[4], radar[6]:
        assert abs(d - expected) < 0.15


def test_thin_interior_wall_ray_hit():
    """中间细竖墙：正前射线命中，距离约为半格。"""
    from tank_sim.core.map_loader import make_game_map

    cell = 60.0
    wall = 4.0
    h = [[True, True], [True, True]]
    v = [[True, True, True]]
    gm = make_game_map(
        cols=2,
        rows=1,
        cell_px=cell,
        wall_thickness=wall,
        h_walls=h,
        v_walls=v,
        spawn_red=(cell * 0.5, cell * 0.5),
        spawn_blue=(cell * 1.5, cell * 0.5),
    )
    tank = TankState(x=cell * 0.5, y=cell * 0.5, theta=0.0, owner="red")
    radar = build_wall_radar(tank, gm, n_rays=8)
    w, h_px = map_bounds(gm)
    scale = 0.5 * max(w, h_px)
    # 左格中心到中间竖墙内缘 ≈ cell/2 - wall/2
    expected = (0.5 * cell - 0.5 * wall) / scale
    assert abs(radar[0] - expected) < 0.05
    assert radar[0] < 1.0
    # 未命中上限为 2；正前应明显短于对角线方向（角落到外框更远）
    assert radar[0] < radar[1]
    assert radar[0] < radar[7]
