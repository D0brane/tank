"""世界初始化与单步推进。"""

from __future__ import annotations

import copy

from tank_sim.config import EnvConfig, SimConfig
from tank_sim.core.bullet import step_bullets
from tank_sim.core.collision import clamp_tank_to_map, tank_overlaps_wall, tanks_overlap
from tank_sim.core.map_loader import load_map
from tank_sim.core.tank import (
    apply_rotation_checked,
    apply_translation,
    resolve_tank_tank_overlap,
    tick_cooldown,
    try_fire,
)
from tank_sim.core.types import BulletState, ControlIntent, GameMap, TankState, WorldState
from tank_sim.core.spawn import DualSpawn


def pop_oldest_bullet(bullets: list[BulletState], owner: str) -> bool:
    """移除指定方最早的一发（age 最大，并列取 id 最小）。成功返回 True。"""
    best_i: int | None = None
    best_key: tuple[int, int] | None = None
    for i, b in enumerate(bullets):
        if b.owner != owner:
            continue
        # age 降序优先；同 age 时 id 升序（更早发放）
        key = (b.age, -b.id)
        if best_key is None or key > best_key:
            best_key = key
            best_i = i
    if best_i is None:
        return False
    bullets.pop(best_i)
    return True


def _maybe_evict_for_fire(
    bullets: list[BulletState],
    tank: TankState,
    intent: ControlIntent,
    sim: SimConfig,
    owner: str,
    active: int,
) -> int:
    """满弹且冷却就绪、意图开火时挤掉最早己方弹；返回更新后的己方弹数。"""
    if (
        tank.alive
        and intent.fire
        and tank.fire_cooldown <= 0
        and active >= sim.bullet.max_active_per_tank
    ):
        if pop_oldest_bullet(bullets, owner):
            active -= 1
    return active


def create_initial_state(
    cfg: EnvConfig,
    map_path: str | None = None,
    game_map: GameMap | None = None,
    spawn: DualSpawn | None = None,
) -> WorldState:
    """
    根据配置创建对局初始状态。

    game_map 优先；否则从 map_path / 默认路径加载。
    spawn 若给定则覆盖地图默认出生点与朝向。
    """
    if game_map is None:
        mpath = map_path or cfg.map.default
        game_map = load_map(mpath, cfg.map.cell_px, cfg.map.wall_thickness)
    if spawn is None:
        red = TankState(
            x=game_map.spawn_red[0],
            y=game_map.spawn_red[1],
            theta=0.0,
            owner="red",
            hp=cfg.sim.tank.hits_to_die,
        )
        blue = TankState(
            x=game_map.spawn_blue[0],
            y=game_map.spawn_blue[1],
            theta=3.141592653589793,
            owner="blue",
            hp=cfg.sim.tank.hits_to_die,
        )
    else:
        red = TankState(
            x=spawn.red_x,
            y=spawn.red_y,
            theta=spawn.red_theta,
            owner="red",
            hp=cfg.sim.tank.hits_to_die,
        )
        blue = TankState(
            x=spawn.blue_x,
            y=spawn.blue_y,
            theta=spawn.blue_theta,
            owner="blue",
            hp=cfg.sim.tank.hits_to_die,
        )
    return WorldState(
        step=0,
        game_map=game_map,
        tanks=(red, blue),
        bullets=[],
    )


def step_world(
    state: WorldState,
    intent_red: ControlIntent,
    intent_blue: ControlIntent,
    sim: SimConfig,
) -> WorldState:
    """
    推进一帧逻辑（确定性顺序）。

    顺序：冷却 → 旋转 → 平移（含车-墙/车-车）→ 开火 → 子弹 → 步数+1
    """
    # 浅拷贝：共享不可变 game_map，避免每步 deepcopy 墙表（量大时极慢）
    s = copy.copy(state)
    s.tanks = (copy.copy(state.tanks[0]), copy.copy(state.tanks[1]))
    s.bullets = [copy.copy(b) for b in state.bullets]
    s.red_killed_by = "none"
    s.blue_killed_by = "none"
    s.red_fired = False
    s.blue_fired = False
    s.kill_bullet_bounces = 0
    s.kill_bullet_owner = "none"
    s.hit_events = []

    if s.terminated:
        return s

    red, blue = s.tanks

    tick_cooldown(red)
    tick_cooldown(blue)

    # 旋转：红先蓝后，均可引用对方当前位置
    apply_rotation_checked(red, intent_red, sim, s.game_map, other=blue)
    apply_rotation_checked(blue, intent_blue, sim, s.game_map, other=red)

    # 平移：红先蓝后，蓝方移动时把红当作障碍
    apply_translation(red, intent_red, sim, s.game_map, other=blue)
    apply_translation(blue, intent_blue, sim, s.game_map, other=red)

    # 残差互撞：蓝方被推出
    if tanks_overlap(red, blue, sim.tank):
        resolve_tank_tank_overlap(red, blue, sim)
        # 推出后若蓝嵌墙，再尝试把蓝拉回空地（简单：回退到半程）
        if tank_overlaps_wall(blue, sim.tank, s.game_map):
            blue.x = (red.x + blue.x) * 0.5
            blue.y = (red.y + blue.y) * 0.5

    # 无墙空场：软钳制防开出世界边界
    clamp_tank_to_map(red, sim.tank, s.game_map)
    clamp_tank_to_map(blue, sim.tank, s.game_map)

    n_red = sum(1 for b in s.bullets if b.owner == "red")
    n_blue = sum(1 for b in s.bullets if b.owner == "blue")
    n_red = _maybe_evict_for_fire(
        s.bullets, red, intent_red, sim, "red", n_red
    )
    b_red = try_fire(red, intent_red, sim, active_bullets=n_red)
    if b_red:
        b_red.id = s.next_bullet_id
        s.next_bullet_id += 1
        s.bullets.append(b_red)
        s.red_fired = True
        n_red += 1
    n_blue = _maybe_evict_for_fire(
        s.bullets, blue, intent_blue, sim, "blue", n_blue
    )
    b_blue = try_fire(blue, intent_blue, sim, active_bullets=n_blue)
    if b_blue:
        b_blue.id = s.next_bullet_id
        s.next_bullet_id += 1
        s.bullets.append(b_blue)
        s.blue_fired = True

    step_bullets(s, sim)

    s.step += 1
    return s
