"""开火阶段编码：冷却 [-1,0]，就绪后计时 (0,1]。"""

from __future__ import annotations

from tank_sim.core.types import TankState


def encode_fire_phase(tank: TankState, fire_cooldown_frames: int) -> float:
    """
    - 冷却中：剩余/满冷却 → [-1, 0)（刚开火≈-1，将好≈0）
    - 已可开火：就绪经过/满冷却 → [0, 1]（刚就绪=0，满 1.5s 后=1）
    """
    denom = float(max(1, fire_cooldown_frames))
    if tank.fire_cooldown > 0:
        return -max(0, tank.fire_cooldown) / denom
    return min(1.0, max(0, tank.fire_ready_age) / denom)
