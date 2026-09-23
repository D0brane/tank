"""瞄准课程配置加载。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class PromoteConfig:
    metric: Literal["kill_rate", "hit_rate"]
    threshold: float
    max_median_ttk: float | None = None
    require_speed_scale: float | None = None
    require_mean_straight_frames: float | None = None


@dataclass
class StageBotConfig:
    mode: Literal["static", "linear", "turn_cruise", "turret"]
    speed_scale: float = 1.0
    speed_scale_end: float | None = None
    mean_straight_frames: float = 180.0
    mean_straight_frames_end: float | None = None
    turn_duration: int = 30
    anneal_timesteps: int = 0


@dataclass
class StageConfig:
    name: str
    bot: StageBotConfig
    reward: dict[str, Any]
    promote: PromoteConfig


@dataclass
class ArenaConfig:
    """课程空场：random_open=每局随机尺寸（有外框、无内墙）；map=用 map_path。"""

    mode: Literal["map", "random_open"] = "map"
    cols_min: int = 8
    cols_max: int = 12
    rows_min: int = 4
    rows_max: int = 6

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "cols_min": self.cols_min,
            "cols_max": self.cols_max,
            "rows_min": self.rows_min,
            "rows_max": self.rows_max,
        }


@dataclass
class RotatePenaltySchedule:
    """按评测 hit_rate 阶梯下调转向惩罚；最后一档再达标则停训。"""

    levels: list[float]
    threshold: float = 0.3
    # True: hit_rate > threshold；False: hit_rate >= threshold
    strict_gt: bool = True
    stop_after_last: bool = True


@dataclass
class CurriculumAimConfig:
    seed: int
    total_timesteps: int
    env_config: str
    map_path: str
    n_envs: int
    frame_stack: int
    frame_stride: int
    stack_action_mean: bool
    agent_side: str
    random_spawn: bool
    min_spawn_dist: float
    max_spawn_dist: float
    ppo: dict[str, Any]
    eval_n_episodes: int
    eval_every_timesteps: int
    eval_max_episode_steps: int
    checkpoint_every_timesteps: int
    stages: list[StageConfig] = field(default_factory=list)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    rotate_penalty_schedule: RotatePenaltySchedule | None = None


def load_curriculum_aim_config(path: str | Path) -> CurriculumAimConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    env = raw.get("env", {})
    ev = raw.get("eval", {})
    ck = raw.get("checkpoint", {})
    stages: list[StageConfig] = []
    for s in raw["stages"]:
        bot = s["bot"]
        pr = s["promote"]
        stages.append(
            StageConfig(
                name=str(s["name"]),
                bot=StageBotConfig(
                    mode=bot["mode"],
                    speed_scale=float(bot.get("speed_scale", 1.0)),
                    speed_scale_end=(
                        float(bot["speed_scale_end"])
                        if bot.get("speed_scale_end") is not None
                        else None
                    ),
                    mean_straight_frames=float(bot.get("mean_straight_frames", 180.0)),
                    mean_straight_frames_end=(
                        float(bot["mean_straight_frames_end"])
                        if bot.get("mean_straight_frames_end") is not None
                        else None
                    ),
                    turn_duration=int(bot.get("turn_duration", 30)),
                    anneal_timesteps=int(bot.get("anneal_timesteps", 0)),
                ),
                reward=dict(s.get("reward", {})),
                promote=PromoteConfig(
                    metric=pr["metric"],
                    threshold=float(pr["threshold"]),
                    max_median_ttk=(
                        float(pr["max_median_ttk"])
                        if pr.get("max_median_ttk") is not None
                        else None
                    ),
                    require_speed_scale=(
                        float(pr["require_speed_scale"])
                        if pr.get("require_speed_scale") is not None
                        else None
                    ),
                    require_mean_straight_frames=(
                        float(pr["require_mean_straight_frames"])
                        if pr.get("require_mean_straight_frames") is not None
                        else None
                    ),
                ),
            )
        )
    return CurriculumAimConfig(
        seed=int(raw.get("seed", 0)),
        total_timesteps=int(raw["total_timesteps"]),
        env_config=str(raw["env_config"]),
        map_path=str(raw["map_path"]),
        n_envs=int(env.get("n_envs", 8)),
        frame_stack=int(env.get("frame_stack", 16)),
        frame_stride=int(env.get("frame_stride", 1)),
        stack_action_mean=bool(env.get("stack_action_mean", False)),
        agent_side=str(env.get("agent_side", "red")),
        random_spawn=bool(env.get("random_spawn", True)),
        min_spawn_dist=float(env.get("min_spawn_dist", 120.0)),
        max_spawn_dist=float(env.get("max_spawn_dist", 360.0)),
        ppo=dict(raw.get("ppo", {})),
        eval_n_episodes=int(ev.get("n_episodes", 40)),
        eval_every_timesteps=int(ev.get("every_timesteps", 50_000)),
        eval_max_episode_steps=int(ev.get("max_episode_steps", 600)),
        checkpoint_every_timesteps=int(ck.get("every_timesteps", 100_000)),
        stages=stages,
        arena=_load_arena(raw.get("arena")),
        rotate_penalty_schedule=_load_rotate_schedule(raw.get("rotate_penalty_schedule")),
    )


def _load_rotate_schedule(raw: Any) -> RotatePenaltySchedule | None:
    if not isinstance(raw, dict):
        return None
    if raw.get("enabled", True) is False:
        return None
    levels_raw = raw.get("levels")
    if not levels_raw:
        return None
    levels = [float(x) for x in levels_raw]
    if not levels:
        return None
    compare = str(raw.get("compare", "gt")).lower()
    return RotatePenaltySchedule(
        levels=levels,
        threshold=float(raw.get("threshold", 0.3)),
        strict_gt=compare != "ge",
        stop_after_last=bool(raw.get("stop_after_last", True)),
    )


def _load_arena(raw: Any) -> ArenaConfig:
    if not isinstance(raw, dict):
        return ArenaConfig()
    mode = str(raw.get("mode", "map"))
    if mode not in ("map", "random_open"):
        mode = "map"
    cols = raw.get("cols", [8, 12])
    rows = raw.get("rows", [4, 6])
    if isinstance(cols, (list, tuple)) and len(cols) >= 2:
        cols_min, cols_max = int(cols[0]), int(cols[1])
    else:
        cols_min = int(raw.get("cols_min", 8))
        cols_max = int(raw.get("cols_max", 12))
    if isinstance(rows, (list, tuple)) and len(rows) >= 2:
        rows_min, rows_max = int(rows[0]), int(rows[1])
    else:
        rows_min = int(raw.get("rows_min", 4))
        rows_max = int(raw.get("rows_max", 6))
    return ArenaConfig(
        mode=mode,  # type: ignore[arg-type]
        cols_min=cols_min,
        cols_max=cols_max,
        rows_min=rows_min,
        rows_max=rows_max,
    )
