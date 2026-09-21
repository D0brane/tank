"""展示渲染测试。"""

import math

import pytest

pygame = pytest.importorskip("pygame")

from tank_sim.config import default_config_path, load_env_config
from tank_sim.core.geometry import tank_corners, tank_half_extents
from tank_sim.rendering.factory import make_renderer
from tank_sim.rendering.showcase import palette as pal


def test_owner_palette_is_red_blue():
    assert "body" in pal.TANK_RED
    assert "body" in pal.TANK_BLUE
    assert pal.OWNER_PALETTE["blue"] is pal.TANK_BLUE
    # 蓝车不是黄
    assert pal.TANK_BLUE["body"][2] > pal.TANK_BLUE["body"][0]


def test_showcase_renderer_builds_and_rgb():
    cfg = load_env_config(default_config_path())
    from tank_sim.core.world import create_initial_state

    state = create_initial_state(cfg, "assets/maps/empty.txt")
    rend = make_renderer(cfg, style="showcase", scale=1.0)
    arr = rend.render(state, mode="rgb_array")
    assert arr is not None
    assert arr.ndim == 3 and arr.shape[2] == 3
    assert float(arr.mean()) > 120
    rend.close()


def test_obb_corners_span_matches_tank_config():
    tw, th = 20.0, 28.0
    hw, hh = tank_half_extents(tw, th)
    corners = tank_corners(0.0, 0.0, 0.0, hw, hh)
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    assert math.isclose(max(xs) - min(xs), th, abs_tol=1e-6)
    assert math.isclose(max(ys) - min(ys), tw, abs_tol=1e-6)


def test_lite_renderer_still_works():
    cfg = load_env_config(default_config_path())
    from tank_sim.core.world import create_initial_state

    state = create_initial_state(cfg, "assets/maps/empty.txt")
    rend = make_renderer(cfg, style="lite", scale=1)
    arr = rend.render(state, mode="rgb_array")
    assert arr is not None
    rend.close()
