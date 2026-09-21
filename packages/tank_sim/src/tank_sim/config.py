"""从 YAML 加载仿真配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tank_sim.observation.spec import expected_obs_dim


@dataclass(frozen=True)
class TankConfig:
    width: float
    height: float
    speed_forward: float
    angular_speed: float
    collision_radius: float


@dataclass(frozen=True)
class BulletConfig:
    speed: float
    radius: float
    substeps: int
    # 存活帧数；正版约 10s（OliverBryan / simple-tank）
    lifetime_frames: int
    # 每车同时在场子弹上限；经典 TT 为 5
    max_active_per_tank: int


@dataclass(frozen=True)
class SimConfig:
    fps: int
    tank: TankConfig
    bullet: BulletConfig
    fire_cooldown_frames: int
    max_episode_steps: int


@dataclass(frozen=True)
class MapConfig:
    cell_px: float
    wall_thickness: float
    default: str

    @property
    def tile_px(self) -> int:
        """兼容旧名。"""
        return int(self.cell_px)


@dataclass(frozen=True)
class ObsConfig:
    version: int
    dim: int
    bullet_slots: int
    local_grid: int


@dataclass(frozen=True)
class PlannerConfig:
    refresh_every_steps: int
    replan_distance_threshold: float


@dataclass(frozen=True)
class RewardConfig:
    kill: float
    death: float
    survive_per_step: float
    bullet_near_enemy: float
    bullet_threat_self: float
    fire_penalty: float
    path_delta_scale: float
    # 瞄准塑形：0 关闭；current=对准现位，lead=超前拦截点
    aim_align_scale: float = 0.0
    aim_mode: str = "current"
    # 冷却未结束仍按开火时的惩罚（意图开火且未射出且仍在 CD）
    fire_on_cd_penalty: float = 0.0


@dataclass(frozen=True)
class EnvConfig:
    sim: SimConfig
    map: MapConfig
    obs: ObsConfig
    planner: PlannerConfig
    reward: RewardConfig


def _get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _load_obs_config(raw_obs: dict[str, Any]) -> ObsConfig:
    """解析 obs 段并校验 dim 与 bullet_slots / local_grid 一致。"""
    version = int(raw_obs["version"])
    dim = int(raw_obs["dim"])
    bullet_slots = int(raw_obs["bullet_slots"])
    local_grid = int(raw_obs["local_grid"])
    expected = expected_obs_dim(bullet_slots, local_grid)
    if dim != expected:
        raise ValueError(
            f"obs.dim={dim} 与规格不符，期望 {expected} "
            f"(bullet_slots={bullet_slots}, local_grid={local_grid})"
        )
    return ObsConfig(
        version=version,
        dim=dim,
        bullet_slots=bullet_slots,
        local_grid=local_grid,
    )


def load_env_config(path: str | Path) -> EnvConfig:
    """读取合并后的环境配置文件。"""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    sim_raw = raw["sim"]
    tank_raw = sim_raw["tank"]
    bullet_raw = sim_raw["bullet"]

    return EnvConfig(
        sim=SimConfig(
            fps=int(sim_raw["fps"]),
            tank=TankConfig(
                width=float(tank_raw["width"]),
                height=float(tank_raw["height"]),
                speed_forward=float(tank_raw["speed_forward"]),
                angular_speed=float(tank_raw["angular_speed"]),
                collision_radius=float(tank_raw["collision_radius"]),
            ),
            bullet=BulletConfig(
                speed=float(bullet_raw["speed"]),
                radius=float(bullet_raw["radius"]),
                substeps=int(bullet_raw["substeps"]),
                lifetime_frames=int(
                    bullet_raw.get("lifetime_frames", sim_raw.get("fps", 60) * 10)
                ),
                max_active_per_tank=int(bullet_raw.get("max_active_per_tank", 5)),
            ),
            fire_cooldown_frames=int(sim_raw["fire_cooldown_frames"]),
            max_episode_steps=int(sim_raw["max_episode_steps"]),
        ),
        map=MapConfig(
            cell_px=float(
                raw["map"].get("cell_px", raw["map"].get("tile_px", 60))
            ),
            wall_thickness=float(raw["map"].get("wall_thickness", 4)),
            default=str(raw["map"]["default"]),
        ),
        obs=_load_obs_config(raw["obs"]),
        planner=PlannerConfig(
            refresh_every_steps=int(raw["planner"]["refresh_every_steps"]),
            replan_distance_threshold=float(raw["planner"]["replan_distance_threshold"]),
        ),
        reward=RewardConfig(
            kill=float(raw["reward"]["kill"]),
            death=float(raw["reward"]["death"]),
            survive_per_step=float(raw["reward"]["survive_per_step"]),
            bullet_near_enemy=float(raw["reward"]["bullet_near_enemy"]),
            bullet_threat_self=float(raw["reward"]["bullet_threat_self"]),
            fire_penalty=float(raw["reward"]["fire_penalty"]),
            path_delta_scale=float(raw["reward"]["path_delta_scale"]),
            aim_align_scale=float(raw["reward"].get("aim_align_scale", 0.0)),
            aim_mode=str(raw["reward"].get("aim_mode", "current")),
            fire_on_cd_penalty=float(raw["reward"].get("fire_on_cd_penalty", 0.0)),
        ),
    )


def default_config_path() -> Path:
    """仓库内默认配置路径（相对当前工作目录）。"""
    return Path("configs/env/sim_p0_tt2_classic.yaml")
