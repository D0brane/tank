"""组装单帧上帝观测（维数由 ObsConfig.dim 决定；v3 有雷达 58 / 无雷达 50）。"""

from __future__ import annotations

import copy

import numpy as np

from tank_sim.config import EnvConfig
from tank_sim.core.types import TankState, WorldState
from tank_sim.observation.bullet_features import BulletSlotAssigner, build_bullet_features
from tank_sim.observation.enemy_features import build_enemy_features
from tank_sim.observation.self_features import build_self_features
from tank_sim.observation.wall_radar import build_wall_radar
from tank_sim.planning.planner_features import PlannerCache, compute_planner_features


class ObservationBuilder:
    """带规划缓存、子弹固定槽与上一帧敌人状态的观测构建器。"""

    def __init__(self, cfg: EnvConfig) -> None:
        self._obs_cfg = cfg.obs
        self._planner_cfg = cfg.planner
        self._sim = cfg.sim
        if cfg.obs.bullet_slots % 2 != 0:
            raise ValueError("obs.bullet_slots 必须为偶数（己/敌各半）")
        slots_per_side = cfg.obs.bullet_slots // 2
        self._red_cache = PlannerCache()
        self._blue_cache = PlannerCache()
        self._red_slots = BulletSlotAssigner(slots_per_side)
        self._blue_slots = BulletSlotAssigner(slots_per_side)
        self._prev_red: TankState | None = None
        self._prev_blue: TankState | None = None

    def reset(self) -> None:
        """对局重置时清空缓存。"""
        self._red_cache = PlannerCache()
        self._blue_cache = PlannerCache()
        self._red_slots.reset()
        self._blue_slots.reset()
        self._prev_red = None
        self._prev_blue = None

    def build(self, state: WorldState, observer: str) -> np.ndarray:
        """observer: 'red' | 'blue'"""
        red, blue = state.tanks
        if observer == "red":
            obs_t, en_t, prev_en, cache, assigner = (
                red,
                blue,
                self._prev_blue,
                self._red_cache,
                self._red_slots,
            )
        else:
            obs_t, en_t, prev_en, cache, assigner = (
                blue,
                red,
                self._prev_red,
                self._blue_cache,
                self._blue_slots,
            )

        cd_frames = self._sim.fire_cooldown_frames
        parts: list[float] = []
        parts.extend(build_self_features(obs_t, cd_frames))
        parts.extend(
            build_enemy_features(
                obs_t,
                en_t,
                prev_en,
                speed_forward=self._sim.tank.speed_forward,
                angular_speed=self._sim.tank.angular_speed,
                fire_cooldown_frames=cd_frames,
            )
        )
        slot_list = assigner.assign(obs_t.owner, state.bullets)
        parts.extend(build_bullet_features(obs_t, slot_list))
        n_rays = int(self._obs_cfg.wall_radar_rays)
        if n_rays > 0:
            parts.extend(build_wall_radar(obs_t, state.game_map, n_rays=n_rays))
        parts.extend(
            list(
                compute_planner_features(
                    obs_t,
                    en_t,
                    state.game_map,
                    self._planner_cfg,
                    state.step,
                    cache,
                )
            )
        )

        vec = np.asarray(parts, dtype=np.float32)
        if vec.shape[0] != self._obs_cfg.dim:
            raise RuntimeError(
                f"观测维度错误: 期望 {self._obs_cfg.dim}, 得到 {vec.shape[0]}"
            )
        return vec

    def on_step_end(self, state: WorldState) -> None:
        """步进后更新上一帧坦克状态（供进退/转向推断）。"""
        red, blue = state.tanks
        self._prev_red = copy.copy(red)
        self._prev_blue = copy.copy(blue)
