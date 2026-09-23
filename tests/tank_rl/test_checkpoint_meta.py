"""checkpoint 课程元数据存读。"""

from pathlib import Path

import pytest

from tank_rl.checkpoint_meta import (
    CurriculumCheckpointMeta,
    apply_meta_to_scheduler,
    build_meta_snapshot,
    load_curriculum_meta,
    meta_path_for_checkpoint,
    parse_stage_index_from_filename,
    parse_timesteps_from_filename,
    resolve_resume_target,
    save_curriculum_meta,
    save_promotion_sidecar,
)
from tank_rl.curriculum.config import load_curriculum_aim_config
from tank_rl.curriculum.scheduler import CurriculumScheduler, EvalMetrics


def test_save_load_curriculum_meta(tmp_path: Path):
    ckpt = tmp_path / "model_t100_stage1_stage2_linear.zip"
    ckpt.write_bytes(b"dummy")
    meta = CurriculumCheckpointMeta(
        stage_index=1,
        stage_name="stage2_linear",
        stage_timesteps=50_000,
        timesteps=100_000,
        bot_mode="linear",
        speed_scale=0.55,
        mean_straight_frames=180.0,
        turn_duration=25,
        frame_stack=16,
    )
    path = save_curriculum_meta(ckpt, meta)
    assert path == meta_path_for_checkpoint(ckpt)
    assert path.is_file()
    loaded = load_curriculum_meta(ckpt)
    assert loaded is not None
    assert loaded.stage_index == 1
    assert loaded.stage_name == "stage2_linear"
    assert loaded.speed_scale == 0.55
    assert loaded.bot_mode == "linear"
    assert loaded.frame_stack == 16
    assert loaded.frame_stride == 1


def test_parse_stage_index_from_filename():
    assert (
        parse_stage_index_from_filename(
            "runs/x/checkpoints/model_t114688_stage0_stage1_static.zip"
        )
        == 0
    )
    assert (
        parse_stage_index_from_filename("model_t200_stage2_stage3_turn_cruise.zip") == 2
    )
    assert parse_stage_index_from_filename("final_model.zip") is None


def test_promotion_sidecar_and_snapshot(tmp_path: Path):
    ckpt = tmp_path / "stage0_stage1_static_to_stage2_linear_t50000.zip"
    ckpt.write_bytes(b"dummy")
    meta = build_meta_snapshot(
        stage_index=0,
        stage_name="stage1_static",
        stage_timesteps=12_000,
        timesteps=50_000,
        frame_stack=16,
        bot_mode="static",
        speed_scale=0.0,
        mean_straight_frames=180.0,
        turn_duration=30,
    )
    save_curriculum_meta(ckpt, meta)
    metrics = EvalMetrics(
        kill_rate=0.9, hit_rate=0.75, median_ttk=120.0, n_episodes=32
    )
    side = save_promotion_sidecar(
        ckpt,
        metrics=metrics,
        completed_stage_index=0,
        completed_stage_name="stage1_static",
        next_stage_index=1,
        next_stage_name="stage2_linear",
        eval_timesteps=50_000,
    )
    assert side.name.endswith(".promotion.json")
    assert side.is_file()
    loaded = load_curriculum_meta(ckpt)
    assert loaded is not None
    assert loaded.stage_index == 0
    assert loaded.stage_name == "stage1_static"


def test_parse_timesteps_from_filename():
    assert (
        parse_timesteps_from_filename(
            "runs/x/checkpoints/model_t114688_stage0_stage1_static.zip"
        )
        == 114688
    )
    assert parse_timesteps_from_filename("stage0_a_to_b_t9912320.zip") == 9912320
    assert parse_timesteps_from_filename("final_model.zip") is None


def test_resolve_resume_target_zip_and_run_dir(tmp_path: Path):
    run = tmp_path / "20260101_120000"
    ckpt_dir = run / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    early = ckpt_dir / "model_t1000_stage0_stage1_static.zip"
    late = ckpt_dir / "model_t5000_stage0_stage1_static.zip"
    early.write_bytes(b"a")
    late.write_bytes(b"b")
    save_curriculum_meta(
        late,
        CurriculumCheckpointMeta(
            stage_index=0,
            stage_name="stage1_static",
            stage_timesteps=5000,
            timesteps=5000,
            bot_mode="static",
            speed_scale=0.0,
            mean_straight_frames=180.0,
            turn_duration=30,
            frame_stack=16,
        ),
    )
    ckpt, run_dir = resolve_resume_target(late)
    assert ckpt == late.resolve()
    assert run_dir == run.resolve()

    ckpt2, run_dir2 = resolve_resume_target(run)
    assert ckpt2 == late.resolve()
    assert run_dir2 == run.resolve()

    latest = run / "latest.zip"
    latest.symlink_to(Path("checkpoints") / early.name)
    ckpt3, _ = resolve_resume_target(run)
    assert ckpt3 == latest.resolve()


def test_apply_meta_to_scheduler():
    cfg = load_curriculum_aim_config("configs/train/curriculum_aim.yaml")
    sch = CurriculumScheduler(cfg)
    meta = CurriculumCheckpointMeta(
        stage_index=1,
        stage_name="stage2_linear",
        stage_timesteps=12_345,
        timesteps=200_000,
        bot_mode="linear",
        speed_scale=0.7,
        mean_straight_frames=180.0,
        turn_duration=25,
        frame_stack=16,
    )
    apply_meta_to_scheduler(sch, meta)
    assert sch.stage_index == 1
    assert sch.stage_timesteps == 12_345
    assert sch.bot.mode == "linear"
    assert abs(sch.bot.speed_scale - 0.7) < 1e-6


def test_advance_scheduler_from_promotion(tmp_path: Path):
    from tank_rl.checkpoint_meta import advance_scheduler_from_promotion

    cfg = load_curriculum_aim_config("configs/train/curriculum_aim.yaml")
    sch = CurriculumScheduler(cfg)
    apply_meta_to_scheduler(
        sch,
        CurriculumCheckpointMeta(
            stage_index=0,
            stage_name="stage1_static",
            stage_timesteps=1000,
            timesteps=1000,
            bot_mode="static",
            speed_scale=0.0,
            mean_straight_frames=180.0,
            turn_duration=30,
            frame_stack=3,
        ),
    )
    ckpt = tmp_path / "promo.zip"
    ckpt.write_bytes(b"x")
    save_promotion_sidecar(
        ckpt,
        metrics=EvalMetrics(
            kill_rate=0.5, hit_rate=0.3, median_ttk=100.0, n_episodes=10
        ),
        completed_stage_index=0,
        completed_stage_name="stage1_static",
        next_stage_index=1,
        next_stage_name="stage2_linear",
        eval_timesteps=1000,
    )
    prom = advance_scheduler_from_promotion(sch, ckpt)
    assert prom is not None
    assert sch.stage_index == 1
    assert sch.stage.name == "stage2_linear"
    assert sch.stage_timesteps == 0
    assert sch.bot.mode == "linear"
    assert sch.bot.speed_scale == pytest.approx(0.3)


def test_missing_meta_returns_none(tmp_path: Path):
    ckpt = tmp_path / "lonely.zip"
    ckpt.write_bytes(b"x")
    assert load_curriculum_meta(ckpt) is None
