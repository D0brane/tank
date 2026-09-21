"""OBB 几何单测。"""

from tank_sim.core.geometry import (
    circle_overlaps_obb,
    obb_overlaps_aabb,
    obb_overlaps_obb,
    reflect_velocity,
    tank_corners,
    tank_half_extents,
)


def test_tank_corners_axis_aligned():
    hw, hh = tank_half_extents(20.0, 28.0)
    corners = tank_corners(100.0, 100.0, 0.0, hw, hh)
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    assert abs(min(xs) - (100 - 14)) < 1e-6
    assert abs(max(xs) - (100 + 14)) < 1e-6
    assert abs(min(ys) - (100 - 10)) < 1e-6
    assert abs(max(ys) - (100 + 10)) < 1e-6


def test_obb_aabb_overlap_clear():
    hw, hh = tank_half_extents(20.0, 28.0)
    corners = tank_corners(50.0, 50.0, 0.0, hw, hh)
    assert not obb_overlaps_aabb(corners, 200.0, 200.0, 16.0)
    assert obb_overlaps_aabb(corners, 50.0, 50.0, 16.0)


def test_obb_obb_overlap():
    hw, hh = tank_half_extents(20.0, 28.0)
    a = tank_corners(0.0, 0.0, 0.0, hw, hh)
    b = tank_corners(5.0, 0.0, 0.0, hw, hh)
    c = tank_corners(100.0, 0.0, 0.0, hw, hh)
    assert obb_overlaps_obb(a, b)
    assert not obb_overlaps_obb(a, c)


def test_circle_obb_hit():
    hw, hh = tank_half_extents(20.0, 28.0)
    center = (0.0, 0.0)
    theta = 0.0
    corners = tank_corners(*center, theta, hw, hh)
    assert circle_overlaps_obb(0.0, 0.0, 3.0, corners, center, theta, hw, hh)
    assert not circle_overlaps_obb(100.0, 0.0, 3.0, corners, center, theta, hw, hh)
    assert circle_overlaps_obb(hh + 2.0, 0.0, 3.0, corners, center, theta, hw, hh)


def test_reflect_velocity_vertical_wall():
    vx, vy = reflect_velocity(10.0, 5.0, -1.0, 0.0)
    assert abs(vx - (-10.0)) < 1e-9
    assert abs(vy - 5.0) < 1e-9


def test_reflect_velocity_horizontal_wall():
    vx, vy = reflect_velocity(3.0, -8.0, 0.0, 1.0)
    assert abs(vx - 3.0) < 1e-9
    assert abs(vy - 8.0) < 1e-9
