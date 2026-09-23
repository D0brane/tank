"""炮塔靶与「直击前被击中」惩罚。"""

import math

import pytest
from dataclasses import replace

from tank_sim.bots.curriculum_bot import CurriculumBot
from tank_sim.config import default_config_path, load_env_config
from tank_sim.core.types import HitEvent
from tank_sim.core.world import create_initial_state
from tank_sim.reward.shaping import RewardState, compute_reward_breakdown


def _zero_dense(cfg):
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
        enemy_proximity_scale=0.0,
        kill=60.0,
        kill_bounce=0.0,
        death=-40.0,
        hit_before_direct=-80.0,
    )


def test_turret_does_not_translate_and_fires_when_aimed():
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    red, blue = state.tanks
    blue.x, blue.y = 200.0, 100.0
    blue.theta = 0.0
    blue.fire_cooldown = 0
    red.x, red.y = 260.0, 100.0
    bot = CurriculumBot(mode="turret", speed_scale=0.0, seed=0)
    bot.reset(seed=0)
    act = bot.act(None, state, "blue")  # type: ignore[arg-type]
    assert abs(act[0]) < 0.2
    assert abs(act[1]) < 0.2
    assert act[2] > 0.0


def test_turret_turns_without_firing_when_misaligned():
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    red, blue = state.tanks
    blue.x, blue.y = 200.0, 100.0
    blue.theta = math.pi / 2
    blue.fire_cooldown = 0
    red.x, red.y = 260.0, 100.0
    bot = CurriculumBot(mode="turret", speed_scale=0.0, seed=0)
    act = bot.act(None, state, "blue")  # type: ignore[arg-type]
    assert abs(act[0]) < 0.2
    assert abs(act[1]) > 0.5
    assert act[2] < 0.0


def test_turret_holds_fire_on_cooldown():
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    red, blue = state.tanks
    blue.x, blue.y = 200.0, 100.0
    blue.theta = 0.0
    blue.fire_cooldown = 10
    red.x, red.y = 260.0, 100.0
    bot = CurriculumBot(mode="turret", speed_scale=0.0, seed=0)
    act = bot.act(None, state, "blue")  # type: ignore[arg-type]
    assert act[2] < 0.0


def test_hit_before_direct_then_latches_off():
    cfg = load_env_config(default_config_path())
    reward_cfg = _zero_dense(cfg)
    prev = create_initial_state(cfg, "assets/maps/empty.txt")
    curr = create_initial_state(cfg, "assets/maps/empty.txt")
    curr.hit_events = [HitEvent(attacker="blue", victim="red", bounces=0)]
    rstate = RewardState()
    parts = compute_reward_breakdown(prev, curr, "red", reward_cfg, rstate)
    assert parts.hit_before_direct == pytest.approx(-80.0)
    assert parts.death == pytest.approx(-40.0)
    assert rstate.direct_scored_red is False

    curr.hit_events = [HitEvent(attacker="red", victim="blue", bounces=0)]
    parts = compute_reward_breakdown(prev, curr, "red", reward_cfg, rstate)
    assert parts.hit_before_direct == pytest.approx(0.0)
    assert parts.kill == pytest.approx(60.0)
    assert rstate.direct_scored_red is True

    curr.hit_events = [HitEvent(attacker="blue", victim="red", bounces=1)]
    parts = compute_reward_breakdown(prev, curr, "red", reward_cfg, rstate)
    assert parts.hit_before_direct == pytest.approx(0.0)
    assert parts.death == pytest.approx(-40.0)


def test_hit_race_preemptive_and_same_step():
    from tank_sim.envs.duel_env import _note_hit_race

    class _Env:
        agent_side = "red"
        _episode_direct_hit_enemy = False
        _episode_hit_enemy = False
        _episode_hits_on_enemy = 0
        _episode_direct_hits_on_enemy = 0
        _episode_hits_on_self = 0
        _hit_before_direct = False
        _preemptive = False

    env = _Env()
    _note_hit_race(env, [HitEvent(attacker="red", victim="blue", bounces=0)])
    assert env._preemptive is True
    assert env._episode_hits_on_self == 0

    env2 = _Env()
    _note_hit_race(
        env2,
        [
            HitEvent(attacker="blue", victim="red", bounces=0),
            HitEvent(attacker="red", victim="blue", bounces=0),
        ],
    )
    assert env2._preemptive is False
    assert env2._episode_direct_hit_enemy is True
    _note_hit_race(env2, [HitEvent(attacker="blue", victim="red", bounces=0)])
    assert env2._preemptive is False
    assert env2._episode_hits_on_self == 2


def test_same_step_hit_still_penalizes_before_latch():
    cfg = load_env_config(default_config_path())
    reward_cfg = _zero_dense(cfg)
    prev = create_initial_state(cfg, "assets/maps/empty.txt")
    curr = create_initial_state(cfg, "assets/maps/empty.txt")
    curr.hit_events = [
        HitEvent(attacker="blue", victim="red", bounces=0),
        HitEvent(attacker="red", victim="blue", bounces=0),
    ]
    rstate = RewardState()
    parts = compute_reward_breakdown(prev, curr, "red", reward_cfg, rstate)
    assert parts.hit_before_direct == pytest.approx(-80.0)
    assert rstate.direct_scored_red is True
