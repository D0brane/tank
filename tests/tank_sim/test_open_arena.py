"""开放空场（有外框墙、无内墙）。"""

import numpy as np

from tank_sim.config import load_env_config
from tank_sim.core.collision import clamp_tank_to_map, map_bounds
from tank_sim.core.map_loader import edges_from_blocked, generate_open_arena
from tank_sim.core.types import TankState
from tank_sim.envs.duel_env import DuelEnv


def test_generate_open_arena_has_border_only():
    gm = generate_open_arena(cols=10, rows=5, cell_px=60.0, wall_thickness=4.0)
    assert gm.cols == 10 and gm.rows == 5
    assert len(gm.wall_rects) > 0
    # 外框封边
    assert all(gm.h_walls[0]) and all(gm.h_walls[gm.rows])
    assert all(gm.v_walls[r][0] and gm.v_walls[r][gm.cols] for r in range(gm.rows))
    # 内部无竖/横隔墙（中间格子之间）
    assert not gm.v_walls[2][5]
    assert not gm.h_walls[2][5]
    assert all(not any(row) for row in gm.blocked)


def test_seal_border_false_no_outer_walls():
    blocked = [[False, False], [False, False]]
    h, v = edges_from_blocked(blocked, seal_border=False)
    assert not any(any(row) for row in h)
    assert not any(any(row) for row in v)
    h2, v2 = edges_from_blocked(blocked, seal_border=True)
    assert any(any(row) for row in h2)
    assert any(any(row) for row in v2)


def test_clamp_tank_to_map_soft_bounds():
    gm = generate_open_arena(cols=8, rows=4, cell_px=60.0)
    w, h = map_bounds(gm)
    from tank_sim.config import TankConfig

    tank_cfg = TankConfig(
        width=20.0, height=28.0, speed_forward=2.0, angular_speed=0.06, collision_radius=12.0
    )
    tank = TankState(x=-50.0, y=h + 100.0, theta=0.0, owner="red")
    moved = clamp_tank_to_map(tank, tank_cfg, gm)
    assert moved
    assert 0.0 <= tank.x <= w
    assert 0.0 <= tank.y <= h


def test_duel_env_random_open_resizes():
    env = DuelEnv(
        config_path="configs/env/sim_p0_tt2_aim_open.yaml",
        opponent="none",
        random_spawn=True,
        min_spawn_dist=80.0,
        max_spawn_dist=200.0,
        open_arena={
            "mode": "random_open",
            "cols_min": 8,
            "cols_max": 11,
            "rows_min": 4,
            "rows_max": 6,
        },
    )
    sizes = set()
    for i in range(12):
        obs, _ = env.reset(seed=100 + i)
        assert obs.shape == (58,)
        assert env._state is not None
        gm = env._state.game_map
        sizes.add((gm.cols, gm.rows))
        assert len(gm.wall_rects) > 0
    env.close()
    assert len(sizes) >= 2


def test_aim_open_env_config_dim():
    cfg = load_env_config("configs/env/sim_p0_tt2_aim_open.yaml")
    assert cfg.obs.dim == 58
    assert cfg.obs.wall_radar_rays == 8
