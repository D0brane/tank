"""贴敌高次距离惩罚。"""

from dataclasses import replace

from tank_sim.config import default_config_path, load_env_config
from tank_sim.core.types import TankState
from tank_sim.core.world import create_initial_state
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
        wall_proximity_scale=0.0,
        **kwargs,
    )


def test_enemy_proximity_config_defaults():
    cfg = load_env_config(default_config_path())
    assert cfg.reward.enemy_proximity_scale == 0.0
    assert cfg.reward.enemy_proximity_margin == 100.0
    assert cfg.reward.enemy_proximity_power == 4.0


def test_close_enemy_gets_powered_penalty():
    cfg = load_env_config(default_config_path())
    reward_cfg = _dense_off(
        cfg,
        enemy_proximity_scale=-0.25,
        enemy_proximity_margin=100.0,
        enemy_proximity_power=4.0,
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    red, blue = state.tanks
    red = TankState(x=200.0, y=150.0, theta=0.0, owner="red")
    blue = TankState(x=240.0, y=150.0, theta=3.14, owner="blue")  # d=40
    state.tanks = (red, blue)
    frac = 1.0 - 40.0 / 100.0
    expected = -0.25 * (frac**4)
    bd = compute_reward_breakdown(
        state, state, "red", reward_cfg, RewardState()
    )
    assert abs(bd.enemy_proximity - expected) < 1e-6
    assert abs(bd.total - expected) < 1e-6


def test_far_enemy_no_proximity_penalty():
    cfg = load_env_config(default_config_path())
    reward_cfg = _dense_off(
        cfg,
        enemy_proximity_scale=-0.25,
        enemy_proximity_margin=100.0,
        enemy_proximity_power=4.0,
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    red, blue = state.tanks
    red = TankState(x=100.0, y=150.0, theta=0.0, owner="red")
    blue = TankState(x=300.0, y=150.0, theta=3.14, owner="blue")  # d=200
    state.tanks = (red, blue)
    bd = compute_reward_breakdown(
        state, state, "red", reward_cfg, RewardState()
    )
    assert bd.enemy_proximity == 0.0
    assert bd.total == 0.0
