"""开火冷却惩罚。"""

from dataclasses import replace

from tank_sim.config import default_config_path, load_env_config
from tank_sim.core.world import create_initial_state, step_world
from tank_sim.core.types import ControlIntent
from tank_sim.reward.shaping import RewardState, compute_reward_for_side


def test_fire_on_cd_penalty():
    cfg = load_env_config(default_config_path())
    assert cfg.reward.fire_on_cd_penalty == -0.05
    reward_cfg = replace(
        cfg.reward,
        survive_per_step=0.0,
        aim_align_scale=0.0,
        path_delta_scale=0.0,
        bullet_near_enemy=0.0,
        bullet_threat_self=0.0,
        fire_penalty=-0.01,
        fire_on_cd_penalty=-0.05,
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    state.tanks[0].fire_cooldown = 50
    prev = state
    idle = ControlIntent()
    fire = ControlIntent(fire=True)
    curr = step_world(state, fire, idle, cfg.sim)
    assert not curr.red_fired
    assert curr.tanks[0].fire_cooldown > 0
    r = compute_reward_for_side(
        prev, curr, "red", reward_cfg, RewardState(), fire_intent=True
    )
    assert r == reward_cfg.fire_on_cd_penalty


def test_successful_fire_uses_fire_penalty_not_cd():
    cfg = load_env_config(default_config_path())
    reward_cfg = replace(
        cfg.reward,
        survive_per_step=0.0,
        aim_align_scale=0.0,
        path_delta_scale=0.0,
        bullet_near_enemy=0.0,
        bullet_threat_self=0.0,
        fire_penalty=-0.01,
        fire_on_cd_penalty=-0.05,
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    state.tanks[0].fire_cooldown = 0
    prev = state
    idle = ControlIntent()
    fire = ControlIntent(fire=True)
    curr = step_world(state, fire, idle, cfg.sim)
    assert curr.red_fired
    r = compute_reward_for_side(
        prev, curr, "red", reward_cfg, RewardState(), fire_intent=True
    )
    assert r == reward_cfg.fire_penalty
