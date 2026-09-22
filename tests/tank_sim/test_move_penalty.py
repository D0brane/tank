"""平移 / 转向运动惩罚。"""

from dataclasses import replace

from tank_sim.config import default_config_path, load_env_config
from tank_sim.core.types import ControlIntent, MoveIntent, RotateIntent
from tank_sim.core.world import create_initial_state, step_world
from tank_sim.reward.shaping import RewardState, compute_reward_for_side


def test_move_penalty_on_forward():
    cfg = load_env_config(default_config_path())
    assert cfg.reward.move_penalty == -0.002
    assert cfg.reward.rotate_penalty == 0.0
    reward_cfg = replace(
        cfg.reward,
        survive_per_step=0.0,
        aim_align_scale=0.0,
        path_delta_scale=0.0,
        bullet_near_enemy=0.0,
        bullet_threat_self=0.0,
        fire_penalty=0.0,
        fire_on_cd_penalty=0.0,
        move_penalty=-0.002,
        rotate_penalty=0.0,
        wall_proximity_scale=0.0,
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    prev = state
    idle = ControlIntent()
    go = ControlIntent(move=MoveIntent.FORWARD)
    curr = step_world(state, go, idle, cfg.sim)
    r = compute_reward_for_side(
        prev,
        curr,
        "red",
        reward_cfg,
        RewardState(),
        move_intent=MoveIntent.FORWARD,
    )
    assert r == reward_cfg.move_penalty


def test_stop_has_no_move_penalty():
    cfg = load_env_config(default_config_path())
    reward_cfg = replace(
        cfg.reward,
        survive_per_step=0.0,
        aim_align_scale=0.0,
        path_delta_scale=0.0,
        bullet_near_enemy=0.0,
        bullet_threat_self=0.0,
        move_penalty=-0.002,
        rotate_penalty=-0.001,
        wall_proximity_scale=0.0,
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    prev = state
    idle = ControlIntent()
    curr = step_world(state, idle, idle, cfg.sim)
    r = compute_reward_for_side(
        prev,
        curr,
        "red",
        reward_cfg,
        RewardState(),
        move_intent=MoveIntent.STOP,
        rotate_intent=RotateIntent.STOP,
    )
    assert r == 0.0


def test_rotate_penalty_on_left():
    cfg = load_env_config(default_config_path())
    reward_cfg = replace(
        cfg.reward,
        survive_per_step=0.0,
        aim_align_scale=0.0,
        path_delta_scale=0.0,
        bullet_near_enemy=0.0,
        bullet_threat_self=0.0,
        move_penalty=0.0,
        rotate_penalty=-0.001,
        move_switch_penalty=0.0,
        wall_proximity_scale=0.0,
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    prev = state
    idle = ControlIntent()
    turn = ControlIntent(rotate=RotateIntent.LEFT)
    curr = step_world(state, turn, idle, cfg.sim)
    r = compute_reward_for_side(
        prev,
        curr,
        "red",
        reward_cfg,
        RewardState(),
        rotate_intent=RotateIntent.LEFT,
    )
    assert r == reward_cfg.rotate_penalty


def test_move_switch_penalty_forward_to_backward():
    cfg = load_env_config(default_config_path())
    assert cfg.reward.move_switch_penalty == -0.02
    reward_cfg = replace(
        cfg.reward,
        survive_per_step=0.0,
        aim_align_scale=0.0,
        path_delta_scale=0.0,
        bullet_near_enemy=0.0,
        bullet_threat_self=0.0,
        move_penalty=0.0,
        rotate_penalty=0.0,
        move_switch_penalty=-0.02,
        wall_proximity_scale=0.0,
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    idle = ControlIntent()
    rstate = RewardState()
    go = ControlIntent(move=MoveIntent.FORWARD)
    curr = step_world(state, go, idle, cfg.sim)
    r1 = compute_reward_for_side(
        state, curr, "red", reward_cfg, rstate, move_intent=MoveIntent.FORWARD
    )
    assert r1 == 0.0
    back = ControlIntent(move=MoveIntent.BACKWARD)
    next_state = step_world(curr, back, idle, cfg.sim)
    r2 = compute_reward_for_side(
        curr, next_state, "red", reward_cfg, rstate, move_intent=MoveIntent.BACKWARD
    )
    assert r2 == reward_cfg.move_switch_penalty


def test_move_switch_via_stop_not_penalized():
    cfg = load_env_config(default_config_path())
    reward_cfg = replace(
        cfg.reward,
        survive_per_step=0.0,
        aim_align_scale=0.0,
        path_delta_scale=0.0,
        bullet_near_enemy=0.0,
        bullet_threat_self=0.0,
        move_penalty=0.0,
        rotate_penalty=0.0,
        move_switch_penalty=-0.02,
        wall_proximity_scale=0.0,
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    idle = ControlIntent()
    rstate = RewardState()
    go = ControlIntent(move=MoveIntent.FORWARD)
    s1 = step_world(state, go, idle, cfg.sim)
    compute_reward_for_side(
        state, s1, "red", reward_cfg, rstate, move_intent=MoveIntent.FORWARD
    )
    s2 = step_world(s1, idle, idle, cfg.sim)
    r_stop = compute_reward_for_side(
        s1, s2, "red", reward_cfg, rstate, move_intent=MoveIntent.STOP
    )
    assert r_stop == 0.0
    back = ControlIntent(move=MoveIntent.BACKWARD)
    s3 = step_world(s2, back, idle, cfg.sim)
    r_back = compute_reward_for_side(
        s2, s3, "red", reward_cfg, rstate, move_intent=MoveIntent.BACKWARD
    )
    assert r_back == 0.0
