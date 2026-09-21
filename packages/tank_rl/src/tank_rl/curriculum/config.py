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
    mode: Literal["static", "linear", "turn_cruise"]
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
class CurriculumAimConfig:
    seed: int
    total_timesteps: int
    env_config: str
    map_path: str
    n_envs: int
    frame_stack: int
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
    )
