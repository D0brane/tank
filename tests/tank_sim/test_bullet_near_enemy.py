"""己弹近敌高次距离塑形。"""

from dataclasses import replace

from tank_sim.config import default_config_path, load_env_config
from tank_sim.core.types import BulletState, TankState
from tank_sim.core.world import create_initial_state
from tank_sim.reward.shaping import RewardState, compute_reward_breakdown


def _dense_off(cfg, **kwargs):
    base = dict(
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
        enemy_proximity_scale=0.0,
    )
    base.update(kwargs)
    return replace(cfg.reward, **base)


def test_bullet_near_enemy_config_defaults():
    cfg = load_env_config(default_config_path())
    assert cfg.reward.bullet_near_enemy_margin == 80.0
    assert cfg.reward.bullet_near_enemy_power == 1.0


def test_power1_matches_legacy_linear():
    cfg = load_env_config(default_config_path())
    reward_cfg = _dense_off(
        cfg,
        bullet_near_enemy=0.2,
        bullet_near_enemy_margin=80.0,
        bullet_near_enemy_power=1.0,
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    red = TankState(x=100.0, y=150.0, theta=0.0, owner="red")
    blue = TankState(x=200.0, y=150.0, theta=3.14, owner="blue")
    state.tanks = (red, blue)
    # 弹距敌 d=40 → 旧式 0.2 * (1 - 40/80) = 0.1
    state.bullets = [
        BulletState(x=160.0, y=150.0, vx=1.0, vy=0.0, owner="red", id=1)
    ]
    expected = 0.2 * (1.0 - 40.0 / 80.0)
    bd = compute_reward_breakdown(state, state, "red", reward_cfg, RewardState())
    assert abs(bd.bullet_near_enemy - expected) < 1e-6


def test_higher_power_smaller_far_reward():
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    red = TankState(x=100.0, y=150.0, theta=0.0, owner="red")
    blue = TankState(x=200.0, y=150.0, theta=3.14, owner="blue")
    state.tanks = (red, blue)
    state.bullets = [
        BulletState(x=160.0, y=150.0, vx=1.0, vy=0.0, owner="red", id=1)
    ]  # d=40, margin=80 → frac=0.5
    r1 = _dense_off(
        cfg,
        bullet_near_enemy=0.2,
        bullet_near_enemy_margin=80.0,
        bullet_near_enemy_power=1.0,
    )
    r4 = _dense_off(
        cfg,
        bullet_near_enemy=0.2,
        bullet_near_enemy_margin=80.0,
        bullet_near_enemy_power=4.0,
    )
    b1 = compute_reward_breakdown(state, state, "red", r1, RewardState())
    b4 = compute_reward_breakdown(state, state, "red", r4, RewardState())
    assert abs(b1.bullet_near_enemy - 0.1) < 1e-6
    assert abs(b4.bullet_near_enemy - 0.2 * (0.5**4)) < 1e-6
    assert b4.bullet_near_enemy < b1.bullet_near_enemy


def test_beyond_margin_zero():
    cfg = load_env_config(default_config_path())
    reward_cfg = _dense_off(
        cfg,
        bullet_near_enemy=0.2,
        bullet_near_enemy_margin=80.0,
        bullet_near_enemy_power=4.0,
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    red = TankState(x=100.0, y=150.0, theta=0.0, owner="red")
    blue = TankState(x=300.0, y=150.0, theta=3.14, owner="blue")
    state.tanks = (red, blue)
    state.bullets = [
        BulletState(x=100.0, y=150.0, vx=1.0, vy=0.0, owner="red", id=1)
    ]  # d=200
    bd = compute_reward_breakdown(state, state, "red", reward_cfg, RewardState())
    assert bd.bullet_near_enemy == 0.0
