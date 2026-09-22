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


def test_latest_model_picks_max_timestep(tmp_path):
    import importlib

    mod = importlib.import_module("apps.eval_watch.__main__")
    root = tmp_path / "runs"
    run = root / "20260921_999999"
    ckpt = run / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "model_t114688_stage0_x.zip").write_bytes(b"x")
    (ckpt / "model_t917504_stage0_x.zip").write_bytes(b"y")
    found = mod._latest_model(root)
    assert found is not None
    assert "917504" in found.name


def test_auto_select_prefers_newest_run(tmp_path, monkeypatch):
    import importlib

    mod = importlib.import_module("apps.eval_watch.__main__")
    root = tmp_path / "runs"
    older = root / "20260921_205249" / "checkpoints"
    newer = root / "20260921_211810" / "checkpoints"
    junk = root / "checkpoints"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    junk.mkdir(parents=True)
    p_old = older / "model_t900000_stage0_x.zip"
    p_new = newer / "model_t114688_stage0_x.zip"
    p_junk = junk / "model_t999999_stage0_x.zip"
    p_old.write_bytes(b"o")
    p_new.write_bytes(b"n")
    p_junk.write_bytes(b"j")

    def fake_peek(path):
        if "999999" in path.name:
            return 928, 999999
        if "900000" in path.name:
            return 928, 900000
        if "114688" in path.name:
            return 928, 114688
        return None, -1

    monkeypatch.setattr(mod, "_zip_obs_flat_and_steps", fake_peek)
    found = mod._auto_select_model(root, expect_flat=928)
    assert found is not None
    assert "211810" in str(found)
    assert "114688" in found.name
