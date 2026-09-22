"""命中即计 kill / death（含非致命）。"""

from dataclasses import replace

from tank_sim.config import default_config_path, load_env_config
from tank_sim.core.types import HitEvent
from tank_sim.core.world import create_initial_state
from tank_sim.reward.shaping import RewardState, compute_reward_for_side


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
        kill=100.0,
        kill_bounce=40.0,
        death=-100.0,
    )


def test_hit_scores_kill_without_winner():
    cfg = load_env_config(default_config_path())
    reward_cfg = _zero_dense(cfg)
    prev = create_initial_state(cfg, "assets/maps/empty.txt")
    curr = create_initial_state(cfg, "assets/maps/empty.txt")
    curr.hit_events = [HitEvent(attacker="red", victim="blue", bounces=0)]
    assert curr.winner == "none"
    assert compute_reward_for_side(prev, curr, "red", reward_cfg, RewardState()) == 100.0
    assert compute_reward_for_side(prev, curr, "blue", reward_cfg, RewardState()) == -100.0


def test_bounce_hit_uses_kill_bounce():
    cfg = load_env_config(default_config_path())
    reward_cfg = _zero_dense(cfg)
    prev = create_initial_state(cfg, "assets/maps/empty.txt")
    curr = create_initial_state(cfg, "assets/maps/empty.txt")
    curr.hit_events = [HitEvent(attacker="red", victim="blue", bounces=2)]
    assert compute_reward_for_side(prev, curr, "red", reward_cfg, RewardState()) == 40.0


def test_friendly_fire_only_death():
    cfg = load_env_config(default_config_path())
    reward_cfg = _zero_dense(cfg)
    prev = create_initial_state(cfg, "assets/maps/empty.txt")
    curr = create_initial_state(cfg, "assets/maps/empty.txt")
    curr.hit_events = [HitEvent(attacker="red", victim="red", bounces=0)]
    assert compute_reward_for_side(prev, curr, "red", reward_cfg, RewardState()) == -100.0
    assert compute_reward_for_side(prev, curr, "blue", reward_cfg, RewardState()) == 0.0


def test_winner_alone_does_not_score_kill():
    """终局 winner 标记不再单独给 kill（须经 hit_events）。"""
    cfg = load_env_config(default_config_path())
    reward_cfg = _zero_dense(cfg)
    prev = create_initial_state(cfg, "assets/maps/empty.txt")
    curr = create_initial_state(cfg, "assets/maps/empty.txt")
    curr.winner = "red"
    curr.kill_bullet_owner = "red"
    curr.tanks[1].alive = False
    assert compute_reward_for_side(prev, curr, "red", reward_cfg, RewardState()) == 0.0
