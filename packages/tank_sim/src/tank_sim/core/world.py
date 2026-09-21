"""世界初始化与单步推进。"""

from __future__ import annotations

import copy

from tank_sim.config import EnvConfig, SimConfig
from tank_sim.core.bullet import step_bullets
from tank_sim.core.collision import tank_overlaps_wall, tanks_overlap
from tank_sim.core.map_loader import load_map
from tank_sim.core.tank import (
    apply_rotation_checked,
    apply_translation,
    resolve_tank_tank_overlap,
    tick_cooldown,
    try_fire,
)
from tank_sim.core.types import ControlIntent, GameMap, TankState, WorldState
from tank_sim.core.spawn import DualSpawn


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
        )
        blue = TankState(
            x=game_map.spawn_blue[0],
            y=game_map.spawn_blue[1],
            theta=3.141592653589793,
            owner="blue",
        )
    else:
        red = TankState(
            x=spawn.red_x,
            y=spawn.red_y,
            theta=spawn.red_theta,
            owner="red",
        )
        blue = TankState(
            x=spawn.blue_x,
            y=spawn.blue_y,
            theta=spawn.blue_theta,
            owner="blue",
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
    s = copy.deepcopy(state)
    s.red_killed_by = "none"
    s.blue_killed_by = "none"
    s.red_fired = False
    s.blue_fired = False

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

    n_red = sum(1 for b in s.bullets if b.owner == "red")
    n_blue = sum(1 for b in s.bullets if b.owner == "blue")
    b_red = try_fire(red, intent_red, sim, active_bullets=n_red)
    if b_red:
        b_red.id = s.next_bullet_id
        s.next_bullet_id += 1
        s.bullets.append(b_red)
        s.red_fired = True
        n_red += 1
    b_blue = try_fire(blue, intent_blue, sim, active_bullets=n_blue)
    if b_blue:
        b_blue.id = s.next_bullet_id
        s.next_bullet_id += 1
        s.bullets.append(b_blue)
        s.blue_fired = True

    step_bullets(s, sim)

    s.step += 1
    return s
