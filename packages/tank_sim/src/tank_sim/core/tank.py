"""坦克运动与开火（OBB 碰撞）。"""

from __future__ import annotations

import math

from tank_sim.config import SimConfig
from tank_sim.core.collision import tank_overlaps_wall, tanks_overlap
from tank_sim.core.geometry import tank_effective_radius
from tank_sim.core.types import BulletState, ControlIntent, MoveIntent, RotateIntent, TankState
from tank_sim.core.types import GameMap


def apply_rotation_checked(
    tank: TankState,
    intent: ControlIntent,
    sim: SimConfig,
    game_map: GameMap,
    other: TankState | None = None,
) -> None:
    """旋转；若嵌墙或与对方重叠则撤销。"""
    if not tank.alive:
        return
    if intent.rotate == RotateIntent.STOP:
        return
    old_theta = tank.theta
    if intent.rotate == RotateIntent.LEFT:
        tank.theta += sim.tank.angular_speed
    else:
        tank.theta -= sim.tank.angular_speed
    if tank_overlaps_wall(tank, sim.tank, game_map):
        tank.theta = old_theta
        return
    if other is not None and tanks_overlap(tank, other, sim.tank):
        tank.theta = old_theta


def try_axis_move(
    tank: TankState,
    dx: float,
    dy: float,
    sim: SimConfig,
    game_map: GameMap,
    other: TankState | None,
) -> bool:
    """
    尝试位移 (dx,dy)；失败则试仅 x / 仅 y（轴对齐滑动）。

    返回是否发生了任意有效位移。
    """
    if not tank.alive:
        return False
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return False

    old_x, old_y = tank.x, tank.y

    def valid() -> bool:
        if tank_overlaps_wall(tank, sim.tank, game_map):
            return False
        if other is not None and tanks_overlap(tank, other, sim.tank):
            return False
        return True

    tank.x, tank.y = old_x + dx, old_y + dy
    if valid():
        return True

    # 仅 x
    tank.x, tank.y = old_x + dx, old_y
    if valid():
        return True

    # 仅 y
    tank.x, tank.y = old_x, old_y + dy
    if valid():
        return True

    tank.x, tank.y = old_x, old_y
    return False


def apply_translation(
    tank: TankState,
    intent: ControlIntent,
    sim: SimConfig,
    game_map: GameMap,
    other: TankState | None = None,
) -> None:
    """按定速平移，遇墙/对方则轴向滑动或回退。"""
    if not tank.alive:
        return
    if intent.move == MoveIntent.STOP:
        return

    dist = sim.tank.speed_forward
    if intent.move == MoveIntent.BACKWARD:
        dist = -dist

    dx = math.cos(tank.theta) * dist
    dy = math.sin(tank.theta) * dist
    try_axis_move(tank, dx, dy, sim, game_map, other)


def tick_cooldown(tank: TankState) -> None:
    """开火冷却递减；归零后累加 fire_ready_age（供观测双段编码）。"""
    if tank.fire_cooldown > 0:
        tank.fire_cooldown -= 1
        tank.fire_ready_age = 0
        return
    if tank.fire_ready_age < 10**9:
        tank.fire_ready_age += 1


def try_fire(
    tank: TankState,
    intent: ControlIntent,
    sim: SimConfig,
    active_bullets: int = 0,
) -> BulletState | None:
    """若允许则生成新子弹（冷却 + 同时在场上限）。"""
    if not tank.alive or not intent.fire:
        return None
    if tank.fire_cooldown > 0:
        return None
    if active_bullets >= sim.bullet.max_active_per_tank:
        return None

    # 炮口在车头方向，半长 + 弹半径
    half_len = sim.tank.height * 0.5
    muzzle_dist = half_len + sim.bullet.radius + 2.0
    bx = tank.x + math.cos(tank.theta) * muzzle_dist
    by = tank.y + math.sin(tank.theta) * muzzle_dist
    vx = math.cos(tank.theta) * sim.bullet.speed
    vy = math.sin(tank.theta) * sim.bullet.speed
    tank.fire_cooldown = sim.fire_cooldown_frames
    tank.fire_ready_age = 0
    return BulletState(
        x=bx,
        y=by,
        vx=vx,
        vy=vy,
        owner=tank.owner,
        radius=sim.bullet.radius,
        age=0,
    )


def resolve_tank_tank_overlap(
    red: TankState,
    blue: TankState,
    sim: SimConfig,
) -> None:
    """
    若两车仍重叠，将蓝方沿红→蓝方向推出（确定性优先级）。

    用于平移后仍可能因旋转等造成的残差。
    """
    if not tanks_overlap(red, blue, sim.tank):
        return
    dx = blue.x - red.x
    dy = blue.y - red.y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        dx, dy, dist = 1.0, 0.0, 1.0
    # 推出至外接圆之和再略多
    need = tank_effective_radius(sim.tank.width, sim.tank.height) * 2.0 + 1.0
    scale = need / dist
    blue.x = red.x + dx * scale
    blue.y = red.y + dy * scale
