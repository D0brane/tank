"""瞄准课程对手：静止 / 直线 / 直行+偶发转弯（不开火）。"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from tank_sim.core.types import WorldState

CurriculumMode = Literal["static", "linear", "turn_cruise"]


class CurriculumBot:
    """
    课程靶子 Bot。

    - static: 不动
    - linear: 定速沿航向直行，贴边则航向翻转 180°
    - turn_cruise: 直行若干帧后短暂转向，再直行；转弯间隔可退火
    """

    def __init__(
        self,
        mode: CurriculumMode = "static",
        *,
        speed_scale: float = 1.0,
        mean_straight_frames: float = 180.0,
        turn_duration: int = 30,
        seed: int | None = None,
    ) -> None:
        self.mode: CurriculumMode = mode
        self.speed_scale = float(np.clip(speed_scale, 0.0, 1.0))
        self.mean_straight_frames = max(1.0, float(mean_straight_frames))
        self.turn_duration = max(1, int(turn_duration))
        self._rng = np.random.default_rng(seed)
        self._desired_theta: float | None = None
        self._straight_left = 0
        self._turn_left = 0
        self._turn_sign = 1.0
        self._margin_px = 40.0

    def configure(
        self,
        *,
        mode: CurriculumMode | None = None,
        speed_scale: float | None = None,
        mean_straight_frames: float | None = None,
        turn_duration: int | None = None,
    ) -> None:
        """训练课程退火时热更新参数。"""
        if mode is not None:
            self.mode = mode
        if speed_scale is not None:
            self.speed_scale = float(np.clip(speed_scale, 0.0, 1.0))
        if mean_straight_frames is not None:
            self.mean_straight_frames = max(1.0, float(mean_straight_frames))
        if turn_duration is not None:
            self.turn_duration = max(1, int(turn_duration))

    def reset(self, seed: int | None = None) -> None:
        """每局开始时重置内部航向 / 转弯计时。"""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._desired_theta = float(self._rng.uniform(0.0, 2.0 * math.pi))
        self._schedule_next_straight()
        self._turn_left = 0
        self._turn_sign = 1.0 if self._rng.random() < 0.5 else -1.0

    def act(self, obs: np.ndarray, state: WorldState, side: str) -> np.ndarray:
        del obs  # 课程 Bot 只用世界状态
        red, blue = state.tanks
        me = red if side == "red" else blue
        if not me.alive or self.mode == "static" or self.speed_scale <= 1e-6:
            return np.zeros(3, dtype=np.float32)

        if self._desired_theta is None:
            self._desired_theta = me.theta

        if self.mode == "turn_cruise":
            return self._act_turn_cruise(me, state)
        return self._act_linear(me, state)

    def _act_linear(self, me, state: WorldState) -> np.ndarray:
        assert self._desired_theta is not None
        if self._near_boundary(me, state):
            self._desired_theta = _wrap(self._desired_theta + math.pi)
        return self._drive_toward(me, self._desired_theta, turn_priority=True)

    def _act_turn_cruise(self, me, state: WorldState) -> np.ndarray:
        assert self._desired_theta is not None
        if self._turn_left > 0:
            self._turn_left -= 1
            # 转向阶段：原地转，不前进
            w = 0.9 * self._turn_sign
            if self._turn_left == 0:
                self._desired_theta = me.theta
                self._schedule_next_straight()
            return np.array([0.0, w, -1.0], dtype=np.float32)

        if self._near_boundary(me, state):
            self._desired_theta = _wrap(self._desired_theta + math.pi)

        self._straight_left -= 1
        if self._straight_left <= 0:
            self._turn_sign = 1.0 if self._rng.random() < 0.5 else -1.0
            self._turn_left = self.turn_duration
            return np.array([0.0, 0.9 * self._turn_sign, -1.0], dtype=np.float32)

        return self._drive_toward(me, self._desired_theta, turn_priority=True)

    def _drive_toward(self, me, target_theta: float, *, turn_priority: bool) -> np.ndarray:
        diff = _wrap(target_theta - me.theta)
        aim_tol = 0.12
        if abs(diff) > aim_tol:
            w = 0.9 if diff > 0 else -0.9
            v = 0.0 if turn_priority else 0.35 * self.speed_scale
            return np.array([v, w, -1.0], dtype=np.float32)
        v = float(np.clip(self.speed_scale, 0.0, 1.0))
        # 连续动作：>0.2 前进；用 0.25+scale 映射到稳定前进
        v_cmd = 0.25 + 0.75 * v if v > 0 else 0.0
        return np.array([v_cmd, 0.0, -1.0], dtype=np.float32)

    def _near_boundary(self, me, state: WorldState) -> bool:
        gm = state.game_map
        w = gm.cols * gm.cell_px
        h = gm.rows * gm.cell_px
        m = self._margin_px
        return me.x < m or me.y < m or me.x > w - m or me.y > h - m

    def _schedule_next_straight(self) -> None:
        # 指数间隔，均值 mean_straight_frames
        gap = int(self._rng.exponential(self.mean_straight_frames))
        self._straight_left = max(self.turn_duration, gap)


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a
