"""连续动作 [-1,1]^3 → 离散控制意图（对齐原版定速 + 开火阈值）。"""

from __future__ import annotations

import numpy as np

from tank_sim.core.types import ControlIntent, MoveIntent, RotateIntent

# 死区边界：与项目定版一致
DEAD_LOW = -0.2
DEAD_HIGH = 0.2
# 动作第三维 ∈ [-1,1]：严格大于 0 开火（=0 与负数均不开）
FIRE_THRESHOLD = 0.0


def continuous_to_intent(action: np.ndarray | list[float]) -> ControlIntent:
    """
    将 3 维连续向量映射为控制意图。

    维度：[v_pred, omega_pred, fire_pred]
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] != 3:
        raise ValueError(f"动作维度应为 3，得到 {a.shape}")

    v, w, fire = float(a[0]), float(a[1]), float(a[2])

    if v < DEAD_LOW:
        move = MoveIntent.BACKWARD
    elif v > DEAD_HIGH:
        move = MoveIntent.FORWARD
    else:
        move = MoveIntent.STOP

    if w < DEAD_LOW:
        rotate = RotateIntent.LEFT
    elif w > DEAD_HIGH:
        rotate = RotateIntent.RIGHT
    else:
        rotate = RotateIntent.STOP

    return ControlIntent(
        move=move,
        rotate=rotate,
        fire=fire > FIRE_THRESHOLD,
    )
