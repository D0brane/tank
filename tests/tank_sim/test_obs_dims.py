"""观测维度与配置校验单测。"""

import numpy as np
import pytest

from tank_sim.config import default_config_path, load_env_config
from tank_sim.core.types import BulletState
from tank_sim.core.world import create_initial_state
from tank_sim.observation.builder import ObservationBuilder
from tank_sim.observation.bullet_features import BulletSlotAssigner
from tank_sim.observation.spec import expected_obs_dim


def test_observation_is_99_v3():
    cfg = load_env_config(default_config_path())
    assert cfg.obs.version == 3
    assert cfg.obs.bullet_slots == 10
    assert cfg.obs.dim == 99
    assert expected_obs_dim(10, 7) == 99

    state = create_initial_state(cfg, "assets/maps/maze_small.txt")
    builder = ObservationBuilder(cfg)
    red = builder.build(state, "red")
    blue = builder.build(state, "blue")
    assert red.shape == (99,)
    assert blue.shape == (99,)
    assert red.dtype == np.float32
    # 开局已就绪满周期 → +1
    assert red[0] == pytest.approx(1.0)


def test_self_fire_phase_in_obs():
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    max_cd = cfg.sim.fire_cooldown_frames
    builder = ObservationBuilder(cfg)

    state.tanks[0].fire_cooldown = max_cd
    state.tanks[0].fire_ready_age = 0
    assert builder.build(state, "red")[0] == pytest.approx(-1.0)

    state.tanks[0].fire_cooldown = max_cd // 2
    assert builder.build(state, "red")[0] == pytest.approx(-0.5)

    state.tanks[0].fire_cooldown = 0
    state.tanks[0].fire_ready_age = 0
    assert builder.build(state, "red")[0] == pytest.approx(0.0)

    state.tanks[0].fire_ready_age = max_cd // 2
    assert builder.build(state, "red")[0] == pytest.approx(0.5)

    state.tanks[0].fire_ready_age = max_cd
    assert builder.build(state, "red")[0] == pytest.approx(1.0)


def test_bullet_slots_stable_across_frames():
    """存活子弹在连续帧保持同一槽位（不因威胁重排）。"""
    assigner = BulletSlotAssigner(slots_per_side=5)
    b1 = BulletState(x=10, y=0, vx=1, vy=0, owner="red", id=1)
    b2 = BulletState(x=100, y=0, vx=1, vy=0, owner="blue", id=2)
    s0 = assigner.assign("red", [b1, b2])
    assert s0[0] is b1 and s0[5] is b2
    b3 = BulletState(x=20, y=0, vx=1, vy=0, owner="red", id=3)
    s1 = assigner.assign("red", [b1, b2, b3])
    assert s1[0] is b1 and s1[1] is b3 and s1[5] is b2
    s2 = assigner.assign("red", [b2, b3])
    assert s2[0] is None and s2[1] is b3 and s2[5] is b2


def test_obs_dim_mismatch_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
sim:
  fps: 60
  tank: {width: 1, height: 1, speed_forward: 1, angular_speed: 0.1, collision_radius: 1}
  bullet: {speed: 1, radius: 1, substeps: 1}
  fire_cooldown_frames: 1
  max_episode_steps: 100
map:
  cell_px: 16
  wall_thickness: 2
  default: assets/maps/empty.txt
obs:
  version: 3
  dim: 78
  bullet_slots: 10
  local_grid: 7
planner:
  refresh_every_steps: 1
  replan_distance_threshold: 1
reward:
  kill: 1
  death: -1
  survive_per_step: 0
  bullet_near_enemy: 0
  bullet_threat_self: 0
  fire_penalty: 0
  path_delta_scale: 0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="obs.dim"):
        load_env_config(bad)
