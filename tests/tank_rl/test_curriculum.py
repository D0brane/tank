"""瞄准课程：Bot、出生、奖励、调度。"""

import math

import numpy as np
import pytest

from tank_sim.bots.curriculum_bot import CurriculumBot
from tank_sim.config import default_config_path, load_env_config
from tank_sim.core.spawn import sample_dual_spawn
from tank_sim.core.world import create_initial_state, step_world
from tank_sim.control.action_mapping import continuous_to_intent
from tank_sim.envs.duel_env import DuelEnv
from tank_sim.reward.shaping import RewardState, compute_reward_for_side
from tank_rl.curriculum.config import load_curriculum_aim_config
from tank_rl.curriculum.scheduler import CurriculumScheduler, EvalMetrics


def test_curriculum_bot_static_zero_action():
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    bot = CurriculumBot(mode="static")
    bot.reset(seed=0)
    act = bot.act(np.zeros(58, dtype=np.float32), state, "blue")
    assert np.allclose(act, 0.0)


def test_curriculum_bot_linear_moves():
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    bot = CurriculumBot(mode="linear", speed_scale=1.0, seed=1)
    bot.reset(seed=1)
    # 对齐航向多步
    for _ in range(40):
        act = bot.act(np.zeros(58, dtype=np.float32), state, "blue")
        intent = continuous_to_intent(act)
        idle = continuous_to_intent(np.zeros(3))
        state = step_world(state, idle, intent, cfg.sim)
    blue = state.tanks[1]
    assert blue.alive
    # 相对出生点应有位移
    assert abs(blue.x - state.game_map.spawn_blue[0]) + abs(
        blue.y - state.game_map.spawn_blue[1]
    ) > 5.0


def test_sample_dual_spawn_distance():
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    rng = np.random.default_rng(0)
    min_d, max_d = 100.0, 300.0
    for _ in range(20):
        spawn = sample_dual_spawn(
            state.game_map,
            cfg.sim.tank,
            rng,
            min_dist=min_d,
            max_dist=max_d,
        )
        d = math.hypot(spawn.red_x - spawn.blue_x, spawn.red_y - spawn.blue_y)
        assert min_d <= d <= max_d


def test_sample_dual_spawn_not_in_blocked_cells():
    from tank_sim.core.map_loader import is_blocked, world_to_tile

    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    gm = state.game_map
    rng = np.random.default_rng(1)
    for _ in range(80):
        spawn = sample_dual_spawn(
            gm,
            cfg.sim.tank,
            rng,
            min_dist=80.0,
            max_dist=320.0,
        )
        for x, y in (
            (spawn.red_x, spawn.red_y),
            (spawn.blue_x, spawn.blue_y),
        ):
            tx, ty = world_to_tile(x, y, gm.cell_px)
            assert not is_blocked(gm, tx, ty), f"spawn in blocked cell ({tx},{ty})"


def test_aim_align_current_higher_when_facing():
    cfg = load_env_config(default_config_path())
    from dataclasses import replace

    reward = replace(
        cfg.reward,
        aim_align_scale=1.0,
        aim_mode="current",
        survive_per_step=0.0,
        wall_proximity_scale=0.0,
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    red, blue = state.tanks
    # 红在左，蓝在右；红朝右对准
    red.x, red.y, red.theta = 100.0, 100.0, 0.0
    blue.x, blue.y, blue.theta = 200.0, 100.0, math.pi
    state.tanks = (red, blue)
    prev = state
    r_face = compute_reward_for_side(prev, state, "red", reward, RewardState())
    red.theta = math.pi  # 背对
    state.tanks = (red, blue)
    r_away = compute_reward_for_side(prev, state, "red", reward, RewardState())
    assert r_face > r_away
    assert abs(r_face - 1.0) < 1e-6
    assert abs(r_away) < 1e-6


def test_aim_align_quadratic_sharper_near_perfect():
    """60° 偏角：cos=0.5 → power=8 时仅 1/256，远低于二次型。"""
    cfg = load_env_config(default_config_path())
    from dataclasses import replace

    reward = replace(
        cfg.reward,
        aim_align_scale=1.0,
        aim_align_power=8.0,
        aim_mode="current",
        survive_per_step=0.0,
        path_delta_scale=0.0,
        bullet_near_enemy=0.0,
        bullet_threat_self=0.0,
        wall_proximity_scale=0.0,
    )
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    red, blue = state.tanks
    red.x, red.y, red.theta = 100.0, 100.0, math.radians(60.0)
    blue.x, blue.y, blue.theta = 200.0, 100.0, 0.0
    state.tanks = (red, blue)
    r = compute_reward_for_side(state, state, "red", reward, RewardState())
    assert abs(r - (0.5**8)) < 1e-6


def test_duel_env_random_spawn_and_curriculum():
    bot = CurriculumBot(mode="static", seed=0)
    env = DuelEnv(
        map_path="assets/maps/empty.txt",
        opponent="curriculum",
        curriculum_bot=bot,
        random_spawn=True,
        min_spawn_dist=100.0,
        max_spawn_dist=320.0,
        reward_overrides={
            "survive_per_step": 0.0,
            "path_delta_scale": 0.0,
            "aim_align_scale": 0.1,
            "aim_mode": "current",
        },
    )
    obs, info = env.reset(seed=42)
    assert obs.shape == (env.cfg.obs.dim,)
    d = math.hypot(
        env.state.tanks[0].x - env.state.tanks[1].x,
        env.state.tanks[0].y - env.state.tanks[1].y,
    )
    assert 100.0 <= d <= 320.0
    obs2, reward, term, trunc, info2 = env.step(np.zeros(3, dtype=np.float32))
    assert "agent_won" in info2 and "hit_enemy" in info2 and "direct_hit" in info2
    env.close()


def test_curriculum_yaml_and_promote():
    cfg = load_curriculum_aim_config("configs/train/curriculum_aim.yaml")
    assert len(cfg.stages) == 3
    assert cfg.stages[0].bot.mode == "static"
    assert cfg.stages[1].reward["aim_mode"] == "lead"
    sch = CurriculumScheduler(cfg)
    assert sch.stage.name == "stage1_static"
    # 未达标
    assert not sch.maybe_promote(
        EvalMetrics(kill_rate=0.1, hit_rate=0.1, median_ttk=100.0, n_episodes=10)
    )
    # 达标晋级
    assert sch.maybe_promote(
        EvalMetrics(kill_rate=0.8, hit_rate=0.8, median_ttk=200.0, n_episodes=10)
    )
    assert sch.stage.name == "stage2_linear"


def test_scheduler_anneal_speed():
    cfg = load_curriculum_aim_config("configs/train/curriculum_aim.yaml")
    sch = CurriculumScheduler(cfg)
    # 跳到 stage2
    sch.stage_index = 1
    sch.stage_timesteps = 0
    sch._apply_stage_bot(sch.stage, 0.0)
    assert sch.bot.speed_scale == pytest.approx(0.3)
    sch.on_timesteps(400_000)
    assert sch.bot.speed_scale == pytest.approx(1.0)
