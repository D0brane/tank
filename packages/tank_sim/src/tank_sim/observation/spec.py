"""观测向量维度规格（与 configs/env 中 obs 段一致）。"""

from __future__ import annotations

# 各子块固定维数（obs_version=3，墙块为雷达）
SELF_DIM = 1
ENEMY_DIM = 6
BULLET_FEATURES_PER_SLOT = 4
PLANNER_DIM = 3


def expected_obs_dim(bullet_slots: int, wall_radar_rays: int) -> int:
    """
    单帧观测总维度。

    公式：1 + 6 + 4×bullet_slots + wall_radar_rays + 3
    """
    return (
        SELF_DIM
        + ENEMY_DIM
        + BULLET_FEATURES_PER_SLOT * bullet_slots
        + wall_radar_rays
        + PLANNER_DIM
    )
