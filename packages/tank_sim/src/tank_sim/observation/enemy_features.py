"""敌方坦克 6 维特征（定速意图 + 开火冷却）。"""

from __future__ import annotations

import math

from tank_sim.core.types import TankState
from tank_sim.observation.fire_phase import encode_fire_phase


def build_enemy_features(
    observer: TankState,
    enemy: TankState,
    prev_enemy: TankState | None,
    speed_forward: float,
    angular_speed: float,
    fire_cooldown_frames: int,
    scale: float = 500.0,
) -> list[float]:
    """
    局部相对位置、朝向差、推断的进退/转向标量 ∈ {-1,0,1}、开火阶段 ∈ [-1,1]。

    线速度沿车头定速，角速度定速，故不必再写 vx,vy。
    """
    dx = enemy.x - observer.x
    dy = enemy.y - observer.y
    lx, ly = _world_to_local(dx, dy, observer.theta)
    dtheta = _wrap_angle(enemy.theta - observer.theta)

    move_sign = 0.0
    rotate_sign = 0.0
    if prev_enemy is not None and enemy.alive:
        move_sign = float(
            _infer_move_sign(
                prev_enemy.x,
                prev_enemy.y,
                enemy.x,
                enemy.y,
                enemy.theta,
                speed_forward,
            )
        )
        rotate_sign = float(
            _infer_rotate_sign(prev_enemy.theta, enemy.theta, angular_speed)
        )

    fire_phase = 0.0 if not enemy.alive else encode_fire_phase(enemy, fire_cooldown_frames)

    return [
        lx / scale,
        ly / scale,
        dtheta / math.pi,
        move_sign,
        rotate_sign,
        fire_phase,
    ]


def _infer_move_sign(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    theta: float,
    speed_forward: float,
) -> int:
    """位移在车头方向上的投影 → 进(+1)/停(0)/退(-1)。"""
    fwd = math.cos(theta) * (x1 - x0) + math.sin(theta) * (y1 - y0)
    thr = max(1e-6, 0.3 * abs(speed_forward))
    if fwd > thr:
        return 1
    if fwd < -thr:
        return -1
    return 0


def _infer_rotate_sign(theta0: float, theta1: float, angular_speed: float) -> int:
    """朝向增量 → 左(+1)/停(0)/右(-1)（与 RotateIntent.LEFT 增角一致）。"""
    dth = _wrap_angle(theta1 - theta0)
    thr = max(1e-6, 0.3 * abs(angular_speed))
    if dth > thr:
        return 1
    if dth < -thr:
        return -1
    return 0


def _world_to_local(dx: float, dy: float, theta: float) -> tuple[float, float]:
    c, s = math.cos(theta), math.sin(theta)
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    return lx, ly


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
