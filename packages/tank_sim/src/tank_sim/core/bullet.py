"""子弹步进与命中。"""

from __future__ import annotations

from tank_sim.config import SimConfig
from tank_sim.core.collision import bullet_hits_tank, resolve_bullet_wall_step
from tank_sim.core.types import BulletState, WorldState


def step_bullets(state: WorldState, sim: SimConfig) -> None:
    """
    更新所有子弹位置、墙反射、命中判定；超时移除。

    命中后标记 winner 与 killed_by，并移除该子弹。
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
            red.alive = False
            if nb.owner == "red":
                state.red_killed_by = "self"
                state.winner = "blue"
            else:
                state.red_killed_by = "enemy"
                state.winner = "blue"
            continue
        if hit_blue and blue.alive:
            blue.alive = False
            if nb.owner == "blue":
                state.blue_killed_by = "self"
                state.winner = "red"
            else:
                state.blue_killed_by = "enemy"
                state.winner = "red"
            continue

        survivors.append(nb)

    state.bullets = survivors


def _integrate_one(bullet: BulletState, sim: SimConfig, game_map) -> BulletState:
    """单子步进，含子步防穿墙。"""
    x, y, vx, vy = bullet.x, bullet.y, bullet.vx, bullet.vy
    substeps = max(1, sim.bullet.substeps)
    dt = 1.0 / substeps

    for _ in range(substeps):
        x, y, vx, vy = resolve_bullet_wall_step(
            x, y, vx, vy, bullet.radius, game_map, dt=dt
        )

    return BulletState(
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        owner=bullet.owner,
        radius=bullet.radius,
        age=bullet.age + 1,
        id=bullet.id,
    )
