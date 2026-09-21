"""PolicyBundle 与阶段解析（无真实权重）。"""

from pathlib import Path

import numpy as np
import pytest

from tank_rl.curriculum.config import load_curriculum_aim_config
from tank_sim.bots.curriculum_bot import CurriculumBot


def test_resolve_stage_via_curriculum_yaml():
    cfg = load_curriculum_aim_config("configs/train/curriculum_aim.yaml")
    assert cfg.stages[0].name == "stage1_static"
    bot = CurriculumBot(mode="static")
    # 观战终点难度
    s2 = cfg.stages[1].bot
    bot.configure(
        mode=s2.mode,
        speed_scale=s2.speed_scale_end or s2.speed_scale,
        mean_straight_frames=s2.mean_straight_frames,
        turn_duration=s2.turn_duration,
    )
    assert bot.mode == "linear"
    assert bot.speed_scale == pytest.approx(1.0)


def test_policy_bundle_missing_file():
    from tank_rl.inference.policy_bundle import PolicyBundle

    with pytest.raises(FileNotFoundError):
        PolicyBundle(Path("/tmp/no_such_model_tank.zip"), frame_stack=16)
