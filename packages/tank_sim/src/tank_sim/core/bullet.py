"""子弹步进与命中。"""

from __future__ import annotations

from tank_sim.config import SimConfig
from tank_sim.core.collision import bullet_hits_tank, resolve_bullet_wall_step
from tank_sim.core.types import BulletState, HitEvent, TankState, WorldState


def step_bullets(state: WorldState, sim: SimConfig) -> None:
    """
    更新所有子弹位置、墙反射、命中判定；超时移除。

    命中任意存活坦克：扣 1 hp 并立刻移除该弹。
    hp 扣至 ≤0 时置死，并标记 winner / killed_by / 致死弹信息。
    """
    red, blue = state.tanks
    survivors: list[BulletState] = []
    lifetime = max(1, sim.bullet.lifetime_frames)

    for b in state.bullets:
        if state.terminated:
            nb = _integrate_one(b, sim, state.game_map)
            if nb.age < lifetime:
                survivors.append(nb)
            continue

        nb = _integrate_one(b, sim, state.game_map)
        if nb.age >= lifetime:
            continue

        hit_red = bullet_hits_tank(nb, red, sim.tank)
        hit_blue = bullet_hits_tank(nb, blue, sim.tank)

        if hit_red and red.alive:
            _apply_hit(state, victim=red, bullet=nb)
            continue
        if hit_blue and blue.alive:
            _apply_hit(state, victim=blue, bullet=nb)
            continue

        survivors.append(nb)

    state.bullets = survivors


def _apply_hit(state: WorldState, *, victim: TankState, bullet: BulletState) -> None:
    """扣血；记录命中事件；致命时写终局字段。子弹由调用方丢弃。"""
    state.hit_events.append(
        HitEvent(attacker=bullet.owner, victim=victim.owner, bounces=bullet.bounces)
    )
    victim.hp -= 1
    if victim.hp > 0:
        return
    victim.alive = False
    victim.hp = 0
    state.kill_bullet_bounces = bullet.bounces
    state.kill_bullet_owner = bullet.owner
    if victim.owner == "red":
        if bullet.owner == "red":
            state.red_killed_by = "self"
        else:
            state.red_killed_by = "enemy"
        state.winner = "blue"
    else:
        if bullet.owner == "blue":
            state.blue_killed_by = "self"
        else:
            state.blue_killed_by = "enemy"
        state.winner = "red"


def _integrate_one(bullet: BulletState, sim: SimConfig, game_map) -> BulletState:
    """单子步进，含子步防穿墙；累计本帧墙反射次数。"""
    x, y, vx, vy = bullet.x, bullet.y, bullet.vx, bullet.vy
    bounces = bullet.bounces
    substeps = max(1, sim.bullet.substeps)
    dt = 1.0 / substeps

    for _ in range(substeps):
        x, y, vx, vy, bounced = resolve_bullet_wall_step(
            x, y, vx, vy, bullet.radius, game_map, dt=dt
        )
        if bounced:
            bounces += 1

    return BulletState(
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        owner=bullet.owner,
        radius=bullet.radius,
        age=bullet.age + 1,
        id=bullet.id,
        bounces=bounces,
    )
