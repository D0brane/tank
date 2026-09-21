"""OBB / 圆几何原语（无 pygame，供物理与测试复用）。"""

from __future__ import annotations

import math
from typing import Sequence


def tank_half_extents(width: float, height: float) -> tuple[float, float]:
    """全宽全高 → 半宽半高。"""
    return width * 0.5, height * 0.5


def tank_effective_radius(width: float, height: float) -> float:
    """外接圆半径，仅用于观测威胁粗算与 Bot 阈值。"""
    return 0.5 * math.hypot(width, height)


def tank_corners(
    x: float,
    y: float,
    theta: float,
    half_w: float,
    half_h: float,
) -> list[tuple[float, float]]:
    """
    返回旋转矩形四顶点（中心为原点，局部 +x 为车头）。

    顶点顺序：右前、左前、左后、右后（局部坐标）。
    """
    c, s = math.cos(theta), math.sin(theta)
    local = (
        (half_h, half_w),
        (half_h, -half_w),
        (-half_h, -half_w),
        (-half_h, half_w),
    )
    out: list[tuple[float, float]] = []
    for lx, ly in local:
        wx = x + lx * c - ly * s
        wy = y + lx * s + ly * c
        out.append((wx, wy))
    return out


def _project(points: Sequence[tuple[float, float]], axis: tuple[float, float]) -> tuple[float, float]:
    """点集在轴上投影区间 [min, max]。"""
    ax, ay = axis
    dots = [px * ax + py * ay for px, py in points]
    return min(dots), max(dots)


def _axes_from_obb(corners: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """OBB 两根边法线（归一化）。"""
    axes: list[tuple[float, float]] = []
    for i in range(2):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % 4]
        ex, ey = x1 - x0, y1 - y0
        # 边的法线
        nx, ny = -ey, ex
        length = math.hypot(nx, ny)
        if length < 1e-12:
            continue
        axes.append((nx / length, ny / length))
    return axes


def _aabb_corners(cx: float, cy: float, half: float) -> list[tuple[float, float]]:
    """轴对齐正方形四角（中心 + 半边长）。"""
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]


def obb_overlaps_obb(
    a_corners: Sequence[tuple[float, float]],
    b_corners: Sequence[tuple[float, float]],
) -> bool:
    """SAT：两 OBB（由顶点给出）是否重叠。"""
    for axis in _axes_from_obb(a_corners) + _axes_from_obb(b_corners):
        amin, amax = _project(a_corners, axis)
        bmin, bmax = _project(b_corners, axis)
        if amax < bmin or bmax < amin:
            return False
    return True


def obb_overlaps_aabb(
    corners: Sequence[tuple[float, float]],
    cx: float,
    cy: float,
    size: float,
) -> bool:
    """OBB 与轴对齐正方形墙块是否重叠。"""
    half = size * 0.5
    return obb_overlaps_rect(corners, cx - half, cy - half, cx + half, cy + half)


def obb_overlaps_rect(
    corners: Sequence[tuple[float, float]],
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> bool:
    """OBB 与任意轴对齐矩形是否重叠（边墙薄 AABB）。"""
    aabb = [
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    ]
    axes = _axes_from_obb(corners) + [(1.0, 0.0), (0.0, 1.0)]
    for axis in axes:
        amin, amax = _project(corners, axis)
        bmin, bmax = _project(aabb, axis)
        if amax < bmin or bmax < amin:
            return False
    return True


def circle_overlaps_rect(
    cx: float,
    cy: float,
    radius: float,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> bool:
    """圆与轴对齐矩形是否重叠。"""
    qx = max(left, min(cx, right))
    qy = max(top, min(cy, bottom))
    dx, dy = cx - qx, cy - qy
    return dx * dx + dy * dy <= radius * radius


def circle_overlaps_obb(
    cx: float,
    cy: float,
    radius: float,
    corners: Sequence[tuple[float, float]],
    center: tuple[float, float],
    theta: float,
    half_w: float,
    half_h: float,
) -> bool:
    """
    圆与 OBB 是否重叠。

    将圆心变换到 OBB 局部坐标，再与膨胀 AABB 比较。
    """
    dx = cx - center[0]
    dy = cy - center[1]
    c, s = math.cos(theta), math.sin(theta)
    # 世界 → 局部（+x 车头）
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    # 最近点在局部 AABB 上的钳制
    qx = max(-half_h, min(lx, half_h))
    qy = max(-half_w, min(ly, half_w))
    ddx = lx - qx
    ddy = ly - qy
    return ddx * ddx + ddy * ddy <= radius * radius


def closest_point_on_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float, float]:
    """
    点到线段最近点，返回 (qx, qy, t)。

    t ∈ [0,1] 为参数。
    """
    abx, aby = bx - ax, by - ay
    len2 = abx * abx + aby * aby
    if len2 < 1e-12:
        return ax, ay, 0.0
    t = ((px - ax) * abx + (py - ay) * aby) / len2
    t = max(0.0, min(1.0, t))
    return ax + t * abx, ay + t * aby, t


def segment_all_aabb_edge_hits(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> list[tuple[float, float, float, float, float]]:
    """
    线段与 AABB 四边的全部合法交点。

    返回列表元素：(t, ix, iy, nx, ny)，t ∈ [0,1]。
    """
    dx = x1 - x0
    dy = y1 - y0
    hits: list[tuple[float, float, float, float, float]] = []
    eps = 1e-5

    def consider(t: float, nx: float, ny: float) -> None:
        if t < -eps or t > 1.0 + eps:
            return
        t = max(0.0, min(1.0, t))
        ix = x0 + t * dx
        iy = y0 + t * dy
        if nx != 0.0:
            if iy < top - eps or iy > bottom + eps:
                return
        if ny != 0.0:
            if ix < left - eps or ix > right + eps:
                return
        hits.append((t, ix, iy, nx, ny))

    if abs(dx) > 1e-12:
        consider((left - x0) / dx, -1.0, 0.0)
        consider((right - x0) / dx, 1.0, 0.0)
    if abs(dy) > 1e-12:
        consider((top - y0) / dy, 0.0, -1.0)
        consider((bottom - y0) / dy, 0.0, 1.0)
    return hits


def segment_intersect_aabb_edges(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[float, float, float, float] | None:
    """最早交点 (ix, iy, nx, ny)；无交则 None。"""
    hits = segment_all_aabb_edge_hits(x0, y0, x1, y1, left, top, right, bottom)
    if not hits:
        return None
    hits.sort(key=lambda h: h[0])
    _, ix, iy, nx, ny = hits[0]
    return ix, iy, nx, ny


def reflect_velocity(vx: float, vy: float, nx: float, ny: float) -> tuple[float, float]:
    """镜面反射：v' = v - 2(v·n)n，n 为单位法线。"""
    dot = vx * nx + vy * ny
    return vx - 2.0 * dot * nx, vy - 2.0 * dot * ny
