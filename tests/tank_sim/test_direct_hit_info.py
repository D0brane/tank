"""直击 / 反弹击杀 info 标记。"""

from tank_sim.config import default_config_path, load_env_config
from tank_sim.core.types import BulletState
from tank_sim.core.world import create_initial_state
from tank_sim.envs.duel_env import DuelEnv


def test_direct_hit_info_on_zero_bounce_kill():
    env = DuelEnv(
        map_path="assets/maps/empty.txt",
        opponent="none",
        random_spawn=False,
    )
    env.reset(seed=0)
    assert env._state is not None
    red, blue = env._state.tanks
    blue.hp = 1
    # 红弹贴脸打蓝，未反弹
    env._state.bullets = [
        BulletState(
            x=blue.x - 8.0,
            y=blue.y,
            vx=10.0,
            vy=0.0,
            owner="red",
            radius=2.5,
            bounces=0,
        )
    ]
    obs, reward, term, trunc, info = env.step([0.0, 0.0, -1.0])
    assert term
    assert info["agent_won"]
    assert info["direct_hit"] is True
    assert info["hit_enemy"] is True
    assert info["kill_bullet_bounces"] == 0
    env.close()


def test_non_lethal_direct_hit_counts_for_hit_rate():
    """五血：一发未反弹命中即 direct_hit，不必打死。"""
    env = DuelEnv(
        map_path="assets/maps/empty.txt",
        opponent="none",
        random_spawn=False,
    )
    env.reset(seed=0)
    red, blue = env._state.tanks
    assert blue.hp == 5
    env._state.bullets = [
        BulletState(
            x=blue.x - 8.0,
            y=blue.y,
            vx=10.0,
            vy=0.0,
            owner="red",
            radius=2.5,
            bounces=0,
        )
    ]
    _obs, _r, term, _t, info = env.step([0.0, 0.0, -1.0])
    assert not term
    assert info["direct_hit"] is True
    assert info["hit_enemy"] is True
    _, blue_after = env._state.tanks
    assert blue_after.hp == 4
    env.close()


def test_bounce_kill_not_direct_hit():
    env = DuelEnv(
        map_path="assets/maps/empty.txt",
        opponent="none",
        random_spawn=False,
    )
    env.reset(seed=0)
    red, blue = env._state.tanks
    blue.hp = 1
    env._state.bullets = [
        BulletState(
            x=blue.x - 8.0,
            y=blue.y,
            vx=10.0,
            vy=0.0,
            owner="red",
            radius=2.5,
            bounces=2,
        )
    ]
    # 本帧 integrate 可能再加 bounce；强制在命中前保留 bounces>0：
    # step_bullets 在命中时读 nb.bounces（含本帧墙反）。贴脸通常不再撞墙。
    _obs, _r, term, _t, info = env.step([0.0, 0.0, -1.0])
    assert term
    assert info["agent_won"]
    assert info["kill_bullet_bounces"] >= 2
    assert info["direct_hit"] is False
    assert info["hit_enemy"] is True  # 总命中：反弹打中也算
    env.close()
