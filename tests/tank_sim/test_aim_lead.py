"""精确拦截 lead 目标点。"""

import math

import pytest

from tank_sim.core.types import TankState
from tank_sim.reward.shaping import _aim_target, _smallest_positive_root


def _tank(x: float, y: float, *, owner: str = "blue", alive: bool = True) -> TankState:
    return TankState(x=x, y=y, theta=0.0, alive=alive, owner=owner)  # type: ignore[arg-type]


def test_lead_stationary_enemy_is_current_position():
    me = _tank(0.0, 0.0, owner="red")
    enemy = _tank(100.0, 0.0)
    prev = _tank(100.0, 0.0)
    tx, ty = _aim_target(me, enemy, prev, "lead", bullet_speed=10.0)
    assert tx == pytest.approx(100.0)
    assert ty == pytest.approx(0.0)


def test_lead_exact_intercept_lateral_motion():
    """敌方横向匀速：拦截点应使弹道与相遇时间一致。"""
    me = _tank(0.0, 0.0, owner="red")
    # 敌在 (0, 100)，每 tick 向 +x 移 2
    enemy = _tank(0.0, 100.0)
    prev = _tank(-2.0, 100.0)
    vb = 5.0
    tx, ty = _aim_target(me, enemy, prev, "lead", bullet_speed=vb)
    # r=(0,100), ve=(2,0): (4-25)t^2 + 10000 = 0 → t = 100/sqrt(21)
    t = 100.0 / math.sqrt(21.0)
    assert tx == pytest.approx(2.0 * t)
    assert ty == pytest.approx(100.0)
    dist = math.hypot(tx - me.x, ty - me.y)
    assert dist / vb == pytest.approx(t)


def test_lead_falls_back_when_uninterceptable():
    me = _tank(0.0, 0.0, owner="red")
    # 敌远离且比弹快：无正根 → 回退现位
    enemy = _tank(10.0, 0.0)
    prev = _tank(0.0, 0.0)  # ve = +10
    tx, ty = _aim_target(me, enemy, prev, "lead", bullet_speed=3.0)
    assert tx == pytest.approx(10.0)
    assert ty == pytest.approx(0.0)


def test_smallest_positive_root_linear_and_quadratic():
    # t^2 - 5t + 6 = 0 → 2, 3
    assert _smallest_positive_root(1.0, -5.0, 6.0) == pytest.approx(2.0)
    # 2t - 4 = 0 → t=2（a≈0）
    assert _smallest_positive_root(0.0, 2.0, -4.0) == pytest.approx(2.0)
    assert _smallest_positive_root(1.0, 0.0, 1.0) is None
