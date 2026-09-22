"""观测向量维度规格（与 configs/env 中 obs 段一致）。"""

from __future__ import annotations

# 各子块固定维数（obs_version=3，墙块为雷达）
SELF_DIM = 1
ENEMY_DIM = 6
BULLET_FEATURES_PER_SLOT = 4
PLANNER_DIM = 3

# 瞄准课程默认：10 弹槽、8 向雷达 → 单帧 58
DEFAULT_BULLET_SLOTS = 10
DEFAULT_WALL_RADAR_RAYS = 8


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


def obs_block_ranges(
    bullet_slots: int = DEFAULT_BULLET_SLOTS,
    wall_radar_rays: int = DEFAULT_WALL_RADAR_RAYS,
) -> dict[str, tuple[int, int]]:
    """各子块半开区间 [start, end)。"""
    bullets = BULLET_FEATURES_PER_SLOT * bullet_slots
    i0 = 0
    i1 = i0 + SELF_DIM
    i2 = i1 + ENEMY_DIM
    i3 = i2 + bullets
    i4 = i3 + wall_radar_rays
    i5 = i4 + PLANNER_DIM
    assert i5 == expected_obs_dim(bullet_slots, wall_radar_rays)
    return {
        "self": (i0, i1),
        "enemy": (i1, i2),
        "bullets": (i2, i3),
        "walls": (i3, i4),
        "planner": (i4, i5),
    }


def stackable_slices(
    bullet_slots: int = DEFAULT_BULLET_SLOTS,
    wall_radar_rays: int = DEFAULT_WALL_RADAR_RAYS,
) -> tuple[tuple[int, int], ...]:
    """历史堆叠的子块：敌方 + 墙雷达 + A*。"""
    r = obs_block_ranges(bullet_slots, wall_radar_rays)
    return (r["enemy"], r["walls"], r["planner"])


def current_only_slices(
    bullet_slots: int = DEFAULT_BULLET_SLOTS,
    wall_radar_rays: int = DEFAULT_WALL_RADAR_RAYS,
) -> tuple[tuple[int, int], ...]:
    """仅保留当前帧：己方开火阶段 + 子弹槽。"""
    r = obs_block_ranges(bullet_slots, wall_radar_rays)
    return (r["self"], r["bullets"])


def stackable_dim(
    bullet_slots: int = DEFAULT_BULLET_SLOTS,
    wall_radar_rays: int = DEFAULT_WALL_RADAR_RAYS,
) -> int:
    return sum(b - a for a, b in stackable_slices(bullet_slots, wall_radar_rays))


def current_only_dim(
    bullet_slots: int = DEFAULT_BULLET_SLOTS,
    wall_radar_rays: int = DEFAULT_WALL_RADAR_RAYS,
) -> int:
    return sum(b - a for a, b in current_only_slices(bullet_slots, wall_radar_rays))


def selective_stacked_dim(
    n_stack: int,
    *,
    bullet_slots: int = DEFAULT_BULLET_SLOTS,
    wall_radar_rays: int = DEFAULT_WALL_RADAR_RAYS,
) -> int:
    """
    选择性稀疏堆叠输出维：
    stackable × n_stack + self_fire + bullets（仅当前帧）。
    """
    k = max(1, int(n_stack))
    return stackable_dim(bullet_slots, wall_radar_rays) * k + current_only_dim(
        bullet_slots, wall_radar_rays
    )


# 默认 58 维规格下的便利常量
STACKABLE_SLICES = stackable_slices()
CURRENT_ONLY_SLICES = current_only_slices()
