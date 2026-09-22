"""贴墙距离惩罚（含高次项）。"""

from dataclasses import replace

from tank_sim.config import default_config_path, load_env_config
from tank_sim.core.types import TankState
from tank_sim.core.world import create_initial_state
from tank_sim.observation.wall_radar import min_wall_distance_px
from tank_sim.reward.shaping import RewardState, compute_reward_breakdown


def _dense_off(cfg, **kwargs):
    return replace(
        cfg.reward,
        survive_per_step=0.0,
        aim_align_scale=0.0,
        path_delta_scale=0.0,
        bullet_near_enemy=0.0,
        bullet_threat_self=0.0,
        fire_penalty=0.0,
        fire_on_cd_penalty=0.0,
        move_penalty=0.0,
        rotate_penalty=0.0,
        move_switch_penalty=0.0,
        enemy_proximity_scale=0.0,
        **kwargs,
    )


def test_wall_proximity_config_defaults():
    cfg = load_env_config(default_config_path())
    assert cfg.reward.wall_proximity_scale == -0.03
    assert cfg.reward.wall_proximity_margin == 40.0
    assert cfg.reward.wall_proximity_power == 1.0


def test_near_wall_gets_proximity_penalty():
    cfg = load_env_config(default_config_path())
    reward_cfg = _dense_off(
        cfg,
        wall_proximity_scale=-0.1,
        wall_proximity_margin=40.0,
        wall_proximity_power=1.0,
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    gm = state.game_map
    red, blue = state.tanks
    red = TankState(
        x=gm.cols * gm.cell_px - 10.0,
        y=gm.rows * gm.cell_px * 0.5,
        theta=0.0,
        owner="red",
    )
    state.tanks = (red, blue)
    d = min_wall_distance_px(red.x, red.y, gm)
    assert d < 40.0
    expected = -0.1 * (1.0 - d / 40.0)
    bd = compute_reward_breakdown(
        state, state, "red", reward_cfg, RewardState()
    )
    assert abs(bd.wall_proximity - expected) < 1e-6
    assert abs(bd.total - expected) < 1e-6


def test_near_wall_power_sharpens_penalty():
    cfg = load_env_config(default_config_path())
    reward_cfg = _dense_off(
        cfg,
        wall_proximity_scale=-0.1,
        wall_proximity_margin=40.0,
        wall_proximity_power=4.0,
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    gm = state.game_map
    red, blue = state.tanks
    # 贴墙一点，但未贴死 → 高次惩罚应明显小于线性
    red = TankState(
        x=gm.cols * gm.cell_px - 25.0,
        y=gm.rows * gm.cell_px * 0.5,
        theta=0.0,
        owner="red",
    )
    state.tanks = (red, blue)
    d = min_wall_distance_px(red.x, red.y, gm)
    assert 0.0 < d < 40.0
    frac = 1.0 - d / 40.0
    linear = -0.1 * frac
    powered = -0.1 * (frac**4)
    assert abs(powered) < abs(linear)
    bd = compute_reward_breakdown(
        state, state, "red", reward_cfg, RewardState()
    )
    assert abs(bd.wall_proximity - powered) < 1e-6


def test_far_from_wall_no_proximity_penalty():
    cfg = load_env_config(default_config_path())
    reward_cfg = _dense_off(
        cfg, wall_proximity_scale=-0.1, wall_proximity_margin=40.0
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    gm = state.game_map
    red, blue = state.tanks
    red = TankState(
        x=gm.cols * gm.cell_px * 0.5,
        y=gm.rows * gm.cell_px * 0.5,
        theta=0.0,
        owner="red",
    )
    state.tanks = (red, blue)
    d = min_wall_distance_px(red.x, red.y, gm)
    assert d >= 40.0
    bd = compute_reward_breakdown(
        state, state, "red", reward_cfg, RewardState()
    )
    assert bd.wall_proximity == 0.0
    assert bd.total == 0.0


def test_open_arena_edge_triggers_wall_proximity():
    """无墙空场：贴世界边界也应有贴墙惩罚。"""
    from tank_sim.core.map_loader import generate_open_arena
    from tank_sim.core.world import create_initial_state as _cis

    cfg = load_env_config("configs/env/sim_p0_tt2_aim_open.yaml")
    reward_cfg = _dense_off(
        cfg,
        wall_proximity_scale=-0.25,
        wall_proximity_margin=50.0,
        wall_proximity_power=4.0,
    )
    gm = generate_open_arena(10, 5, cfg.map.cell_px, cfg.map.wall_thickness)
    assert gm.wall_rects == []
    state = _cis(cfg, game_map=gm)
    red, blue = state.tanks
    red = TankState(x=8.0, y=gm.rows * gm.cell_px * 0.5, theta=0.0, owner="red")
    state.tanks = (red, blue)
    d = min_wall_distance_px(red.x, red.y, gm)
    assert d == 8.0  # 到左边界
    expected = -0.25 * ((1.0 - 8.0 / 50.0) ** 4)
    bd = compute_reward_breakdown(
        state, state, "red", reward_cfg, RewardState()
    )
    assert abs(bd.wall_proximity - expected) < 1e-6
