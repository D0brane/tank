"""训练 checkpoint 旁路的课程元数据（阶段 / 退火后对手参数）。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CurriculumCheckpointMeta:
    """与某个 .zip 权重对应的课程状态快照。"""

    stage_index: int
    stage_name: str
    stage_timesteps: int
    timesteps: int
    bot_mode: str
    speed_scale: float
    mean_straight_frames: float
    turn_duration: int
    frame_stack: int
    frame_stride: int = 1
    stack_action_mean: bool = False
    rotate_phase: int = 0
    schema: int = 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CurriculumCheckpointMeta:
        bot = raw.get("bot") or {}
        return cls(
            schema=int(raw.get("schema", 1)),
            stage_index=int(raw["stage_index"]),
            stage_name=str(raw["stage_name"]),
            stage_timesteps=int(raw.get("stage_timesteps", 0)),
            timesteps=int(raw.get("timesteps", 0)),
            bot_mode=str(bot.get("mode", raw.get("bot_mode", "static"))),
            speed_scale=float(bot.get("speed_scale", raw.get("speed_scale", 0.0))),
            mean_straight_frames=float(
                bot.get("mean_straight_frames", raw.get("mean_straight_frames", 180.0))
            ),
            turn_duration=int(bot.get("turn_duration", raw.get("turn_duration", 30))),
            frame_stack=int(raw.get("frame_stack", 16)),
            frame_stride=int(raw.get("frame_stride", 1)),
            stack_action_mean=bool(raw.get("stack_action_mean", False)),
            rotate_phase=int(raw.get("rotate_phase", 0)),
        )


def meta_path_for_checkpoint(checkpoint: str | Path) -> Path:
    """``model.zip`` → ``model.curriculum.json``。"""
    p = Path(checkpoint)
    return p.with_name(p.stem + ".curriculum.json")


def save_curriculum_meta(checkpoint: str | Path, meta: CurriculumCheckpointMeta) -> Path:
    path = meta_path_for_checkpoint(checkpoint)
    payload = {
        "schema": meta.schema,
        "timesteps": meta.timesteps,
        "stage_index": meta.stage_index,
        "stage_name": meta.stage_name,
        "stage_timesteps": meta.stage_timesteps,
        "frame_stack": meta.frame_stack,
        "frame_stride": meta.frame_stride,
        "stack_action_mean": meta.stack_action_mean,
        "rotate_phase": meta.rotate_phase,
        "bot": {
            "mode": meta.bot_mode,
            "speed_scale": meta.speed_scale,
            "mean_straight_frames": meta.mean_straight_frames,
            "turn_duration": meta.turn_duration,
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_curriculum_meta(checkpoint: str | Path) -> CurriculumCheckpointMeta | None:
    path = meta_path_for_checkpoint(checkpoint)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CurriculumCheckpointMeta.from_dict(raw)


_STAGE_IN_NAME = re.compile(r"_stage(\d+)_")


def parse_stage_index_from_filename(checkpoint: str | Path) -> int | None:
    """从 ``model_t*_stage{N}_name.zip`` 解析阶段下标（旧存盘无 meta 时的回退）。"""
    m = _STAGE_IN_NAME.search(Path(checkpoint).name)
    if not m:
        return None
    return int(m.group(1))


_TIMESTEPS_IN_NAME = re.compile(r"_t(\d+)(?:_|\.|$)")


def parse_timesteps_from_filename(checkpoint: str | Path) -> int | None:
    """从 ``model_t114688_....zip`` / ``..._t9912320.zip`` 解析步数。"""
    m = _TIMESTEPS_IN_NAME.search(Path(checkpoint).name)
    if not m:
        return None
    return int(m.group(1))


def resolve_resume_target(path: str | Path) -> tuple[Path, Path]:
    """
    解析续训目标。

    ``path`` 可为：
    - 某个 ``.zip`` 权重
    - 某次 run 目录（选 checkpoints 内步数最大的 zip，否则 ``final_model.zip`` / ``latest.zip``）

    返回 ``(checkpoint.zip, run_dir)``。
    """
    p = Path(path).expanduser().resolve()
    if p.is_file():
        if p.suffix != ".zip":
            raise FileNotFoundError(f"续训路径不是 .zip: {p}")
        run_dir = _infer_run_dir_from_checkpoint(p)
        return p, run_dir
    if not p.is_dir():
        raise FileNotFoundError(f"续训路径不存在: {p}")

    # 目录：优先 latest.zip，再 checkpoints 最大 t，再 final_model.zip
    latest = p / "latest.zip"
    if latest.is_file():
        return latest.resolve(), p

    ckpt_dir = p / "checkpoints"
    candidates: list[Path] = []
    if ckpt_dir.is_dir():
        candidates.extend(ckpt_dir.glob("*.zip"))
    final = p / "final_model.zip"
    if final.is_file():
        candidates.append(final)

    if not candidates:
        raise FileNotFoundError(
            f"run 目录下找不到可续训权重（期望 checkpoints/*.zip 或 final_model.zip）: {p}"
        )

    def _key(c: Path) -> tuple[int, float]:
        meta = load_curriculum_meta(c)
        if meta is not None and meta.timesteps > 0:
            return meta.timesteps, c.stat().st_mtime
        parsed = parse_timesteps_from_filename(c)
        if parsed is not None:
            return parsed, c.stat().st_mtime
        return -1, c.stat().st_mtime

    best = max(candidates, key=_key)
    return best.resolve(), p


def _infer_run_dir_from_checkpoint(ckpt: Path) -> Path:
    """``.../run/checkpoints|promotions|named/x.zip`` → ``.../run``。"""
    parent = ckpt.parent
    if parent.name in ("checkpoints", "promotions", "named"):
        return parent.parent
    return parent


def apply_meta_to_scheduler(scheduler, meta: CurriculumCheckpointMeta) -> None:
    """把存盘课程状态写回 CurriculumScheduler（阶段 + 退火后对手）。"""
    if meta.stage_index < 0 or meta.stage_index >= len(scheduler.cfg.stages):
        raise ValueError(
            f"meta.stage_index={meta.stage_index} 越界（共 {len(scheduler.cfg.stages)} 阶段）"
        )
    scheduler.stage_index = int(meta.stage_index)
    scheduler.stage_timesteps = int(meta.stage_timesteps)
    scheduler.rotate_phase = int(getattr(meta, "rotate_phase", 0))
    scheduler.bot.configure(
        mode=meta.bot_mode,  # type: ignore[arg-type]
        speed_scale=meta.speed_scale,
        mean_straight_frames=meta.mean_straight_frames,
        turn_duration=meta.turn_duration,
    )


def advance_scheduler_from_promotion(
    scheduler, checkpoint: str | Path
) -> dict[str, Any] | None:
    """
    若旁路存在 ``*.promotion.json``：晋级权重的 curriculum meta 记的是刚完成阶段，
    续训应跳到 ``next_stage`` 并从该阶段退火起点开始。
    """
    path = Path(checkpoint).with_name(Path(checkpoint).stem + ".promotion.json")
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    next_idx = int(payload["next_stage_index"])
    if next_idx < 0 or next_idx >= len(scheduler.cfg.stages):
        raise ValueError(
            f"promotion next_stage_index={next_idx} 越界"
            f"（共 {len(scheduler.cfg.stages)} 阶段）"
        )
    scheduler.stage_index = next_idx
    scheduler.stage_timesteps = 0
    scheduler._apply_stage_bot(scheduler.stage, progress=0.0)
    return payload


def build_meta_snapshot(
    *,
    stage_index: int,
    stage_name: str,
    stage_timesteps: int,
    timesteps: int,
    frame_stack: int,
    bot_mode: str,
    speed_scale: float,
    mean_straight_frames: float,
    turn_duration: int,
    frame_stride: int = 1,
    stack_action_mean: bool = False,
    rotate_phase: int = 0,
) -> CurriculumCheckpointMeta:
    """固定快照（晋级存盘：记录刚完成阶段的对手参数）。"""
    return CurriculumCheckpointMeta(
        stage_index=int(stage_index),
        stage_name=str(stage_name),
        stage_timesteps=int(stage_timesteps),
        timesteps=int(timesteps),
        bot_mode=str(bot_mode),
        speed_scale=float(speed_scale),
        mean_straight_frames=float(mean_straight_frames),
        turn_duration=int(turn_duration),
        frame_stack=int(frame_stack),
        frame_stride=int(frame_stride),
        stack_action_mean=bool(stack_action_mean),
        rotate_phase=int(rotate_phase),
    )


def save_promotion_sidecar(
    checkpoint: str | Path,
    *,
    metrics,
    completed_stage_index: int,
    completed_stage_name: str,
    next_stage_index: int,
    next_stage_name: str,
    eval_timesteps: int,
) -> Path:
    """晋级权重旁路：本局评测指标与阶段跳转。"""
    path = Path(checkpoint).with_name(Path(checkpoint).stem + ".promotion.json")
    payload = {
        "schema": 1,
        "eval_timesteps": int(eval_timesteps),
        "completed_stage_index": int(completed_stage_index),
        "completed_stage_name": str(completed_stage_name),
        "next_stage_index": int(next_stage_index),
        "next_stage_name": str(next_stage_name),
        "kill_rate": float(metrics.kill_rate),
        "hit_rate": float(metrics.hit_rate),
        "median_ttk": metrics.median_ttk,
        "n_episodes": int(metrics.n_episodes),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_meta_from_scheduler(
    scheduler,
    *,
    timesteps: int,
    frame_stack: int,
    frame_stride: int = 1,
    stack_action_mean: bool = False,
) -> CurriculumCheckpointMeta:
    """从当前 CurriculumScheduler 快照对手与阶段。"""
    stage = scheduler.stage
    bot = scheduler.bot
    return CurriculumCheckpointMeta(
        stage_index=int(scheduler.stage_index),
        stage_name=str(stage.name),
        stage_timesteps=int(scheduler.stage_timesteps),
        timesteps=int(timesteps),
        bot_mode=str(bot.mode),
        speed_scale=float(bot.speed_scale),
        mean_straight_frames=float(bot.mean_straight_frames),
        turn_duration=int(bot.turn_duration),
        frame_stack=int(frame_stack),
        frame_stride=int(frame_stride),
        stack_action_mean=bool(stack_action_mean),
        rotate_phase=int(getattr(scheduler, "rotate_phase", 0)),
    )
