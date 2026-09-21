"""规则 Bot v1：偏保守，适合人玩陪练；训练冷启动也可复用。"""

from __future__ import annotations

import math

import numpy as np

from tank_sim.core.types import WorldState


class RuleBotV1:
    """
    启发式对手。

    设计目标：不要开局秒杀；先转向、再逼近、瞄准够准才开火。
    """

    def __init__(
        self,
        *,
        fire_warmup_steps: int = 90,
        aim_tol: float = 0.08,
        fire_range: float = 180.0,
        approach_range: float = 220.0,
    ) -> None:
        self.fire_warmup_steps = fire_warmup_steps
        self.aim_tol = aim_tol
        self.fire_range = fire_range
        self.approach_range = approach_range

    def act(self, obs: np.ndarray, state: WorldState, side: str) -> np.ndarray:
        red, blue = state.tanks
        me = red if side == "red" else blue
        enemy = blue if side == "red" else red

        if not me.alive:
            return np.zeros(3, dtype=np.float32)

        dx = enemy.x - me.x
        dy = enemy.y - me.y
        dist = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        diff = _wrap_angle(target_angle - me.theta)

        v = 0.0
        w = 0.0
        fire = -1.0

        # 优先躲近弹
        for b in state.bullets:
            if b.owner == me.owner:
                continue
            if math.hypot(b.x - me.x, b.y - me.y) < 80.0:
                return np.array([-0.85, 0.9, -1.0], dtype=np.float32)

        # 先转向对准
        if abs(diff) > self.aim_tol:
            w = 0.9 if diff > 0 else -0.9
            # 未对准时缓慢挪动，避免直线冲锋秒杀
            if dist > self.approach_range:
                v = 0.35
            return np.array([v, w, fire], dtype=np.float32)

        # 已大致对准：保持距离，不要贴脸
        if dist > self.approach_range:
            v = 0.55
        elif dist < self.fire_range * 0.45:
            v = -0.45
        else:
            v = 0.0

        # 开局预热后再开火，且距离要够近
        can_fire = (
            state.step >= self.fire_warmup_steps
            and dist < self.fire_range
            and abs(diff) <= self.aim_tol
        )
        if can_fire:
            fire = 0.85

        return np.array([v, w, fire], dtype=np.float32)


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
