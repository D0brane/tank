"""己方 1 维特征（开火阶段）。"""

from __future__ import annotations

from tank_sim.core.types import TankState
from tank_sim.observation.fire_phase import encode_fire_phase


def build_self_features(
    observer: TankState,
    fire_cooldown_frames: int,
) -> list[float]:
    """开火阶段 ∈ [-1, 1]：冷却为负，就绪后 1.5s 计时为正。"""
    return [encode_fire_phase(observer, fire_cooldown_frames)]
