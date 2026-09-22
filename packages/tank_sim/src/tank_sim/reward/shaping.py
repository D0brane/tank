"""奖励塑形（稠密 + 稀疏）。"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Literal

from tank_sim.config import RewardConfig
from tank_sim.core.types import MoveIntent, RotateIntent, TankState, WorldState
from tank_sim.observation.wall_radar import min_wall_distance_px
from tank_sim.planning.astar import astar


@dataclass
class RewardState:
    """跨步缓存（路径距离、上一帧平移意图等）。"""

    prev_path_len_red: float | None = None
    prev_path_len_blue: float | None = None
    prev_move_red: MoveIntent = MoveIntent.STOP
    prev_move_blue: MoveIntent = MoveIntent.STOP


@dataclass
class RewardBreakdown:
    """单步各奖励分项（便于 TensorBoard 分别记录）。"""

    total: float = 0.0
    survive: float = 0.0
    fire: float = 0.0
    fire_on_cd: float = 0.0
    move: float = 0.0
    rotate: float = 0.0
    move_switch: float = 0.0
    bullet_near_enemy: float = 0.0
    bullet_threat_self: float = 0.0
    aim_align: float = 0.0
    path_delta: float = 0.0
    kill: float = 0.0
    kill_bounce: float = 0.0
    death: float = 0.0
    wall_proximity: float = 0.0
    enemy_proximity: float = 0.0

    def as_parts_dict(self) -> dict[str, float]:
        """不含 total 的分项字典（写入 info / TB）。"""
        return {f.name: float(getattr(self, f.name)) for f in fields(self) if f.name != "total"}


def compute_reward_for_side(
    prev: WorldState,
    curr: WorldState,
    side: str,
    cfg: RewardConfig,
    rstate: RewardState,
    *,
    bullet_speed: float = 3.77,
    fire_intent: bool = False,
    move_intent: MoveIntent = MoveIntent.STOP,
    rotate_intent: RotateIntent = RotateIntent.STOP,
) -> float:
    """计算单侧 agent 的本步奖励（合计）。"""
    return compute_reward_breakdown(
        prev,
        curr,
        side,
        cfg,
        rstate,
        bullet_speed=bullet_speed,
        fire_intent=fire_intent,
        move_intent=move_intent,
        rotate_intent=rotate_intent,
    ).total


def compute_reward_breakdown(
    prev: WorldState,
    curr: WorldState,
    side: str,
    cfg: RewardConfig,
    rstate: RewardState,
    *,
    bullet_speed: float = 3.77,
    fire_intent: bool = False,
    move_intent: MoveIntent = MoveIntent.STOP,
    rotate_intent: RotateIntent = RotateIntent.STOP,
) -> RewardBreakdown:
    """计算单侧 agent 的本步奖励分项。"""
    red, blue = curr.tanks
    me: TankState = red if side == "red" else blue
    enemy: TankState = blue if side == "red" else red
    prev_enemy = prev.tanks[1] if side == "red" else prev.tanks[0]

    parts = RewardBreakdown()
    parts.survive = float(cfg.survive_per_step)

    fired = (side == "red" and curr.red_fired) or (side == "blue" and curr.blue_fired)
    if fired:
        parts.fire = float(cfg.fire_penalty)
    elif fire_intent and me.fire_cooldown > 0 and cfg.fire_on_cd_penalty != 0.0:
        parts.fire_on_cd = float(cfg.fire_on_cd_penalty)

    if me.alive and move_intent != MoveIntent.STOP and cfg.move_penalty != 0.0:
        parts.move = float(cfg.move_penalty)
    if me.alive and rotate_intent != RotateIntent.STOP and cfg.rotate_penalty != 0.0:
        parts.rotate = float(cfg.rotate_penalty)

    prev_move = rstate.prev_move_red if side == "red" else rstate.prev_move_blue
    if (
        me.alive
        and cfg.move_switch_penalty != 0.0
        and _is_forward_backward_flip(prev_move, move_intent)
    ):
        parts.move_switch = float(cfg.move_switch_penalty)
    if side == "red":
        rstate.prev_move_red = move_intent
    else:
        rstate.prev_move_blue = move_intent

    near, threat = _bullet_shaping_parts(me, curr, cfg)
    parts.bullet_near_enemy = near
    parts.bullet_threat_self = threat
    parts.aim_align = _aim_align(me, enemy, prev_enemy, cfg, bullet_speed)
    parts.path_delta = _path_delta(me, enemy, curr, side, cfg, rstate)
    parts.wall_proximity = _wall_proximity(me, curr, cfg)
    parts.enemy_proximity = _enemy_proximity(me, enemy, cfg)
    _apply_hit_scores(parts, curr, side, cfg)

    parts.total = (
        parts.survive
        + parts.fire
        + parts.fire_on_cd
        + parts.move
        + parts.rotate
        + parts.move_switch
        + parts.bullet_near_enemy
        + parts.bullet_threat_self
        + parts.aim_align
        + parts.path_delta
        + parts.kill
        + parts.kill_bounce
        + parts.death
        + parts.wall_proximity
        + parts.enemy_proximity
    )
    return parts


def _apply_hit_scores(
    parts: RewardBreakdown, curr: WorldState, side: str, cfg: RewardConfig
) -> None:
    """每次命中计分：打中敌方 → kill/kill_bounce；被击中 → death（友伤只记 death）。"""
    for ev in curr.hit_events:
        if ev.victim == side:
            parts.death += float(cfg.death)
        if ev.attacker == side and ev.victim != side:
            if ev.bounces > 0:
                parts.kill_bounce += float(cfg.kill_bounce)
            else:
                parts.kill += float(cfg.kill)


def _proximity_penalty(d: float, scale: float, margin: float, power: float) -> float:
    """通用近距惩罚：scale × (1 - d/margin)^power；d≥margin 为 0。"""
    if scale == 0.0:
        return 0.0
    m = max(1e-6, float(margin))
    if d >= m:
        return 0.0
    p = max(1.0, float(power))
    return float(scale) * ((1.0 - d / m) ** p)


def _wall_proximity(me: TankState, curr: WorldState, cfg: RewardConfig) -> float:
    """贴墙惩罚：scale × (1 - d/margin)^power，d 为车心到最近墙像素距离。"""
    if cfg.wall_proximity_scale == 0.0 or not me.alive:
        return 0.0
    d = min_wall_distance_px(me.x, me.y, curr.game_map)
    return _proximity_penalty(
        d,
        cfg.wall_proximity_scale,
        cfg.wall_proximity_margin,
        cfg.wall_proximity_power,
    )


def _enemy_proximity(me: TankState, enemy: TankState, cfg: RewardConfig) -> float:
    """贴敌惩罚：scale × (1 - d/margin)^power，d 为两车心距。"""
    if cfg.enemy_proximity_scale == 0.0 or not me.alive:
        return 0.0
    d = math.hypot(me.x - enemy.x, me.y - enemy.y)
    return _proximity_penalty(
        d,
        cfg.enemy_proximity_scale,
        cfg.enemy_proximity_margin,
        cfg.enemy_proximity_power,
    )


def _is_forward_backward_flip(prev: MoveIntent, curr: MoveIntent) -> bool:
    """仅前进↔后退瞬时对拉；经 STOP 不触发。"""
    pair = {prev, curr}
    return pair == {MoveIntent.FORWARD, MoveIntent.BACKWARD}


def _aim_align(
    me: TankState,
    enemy: TankState,
    prev_enemy: TankState,
    cfg: RewardConfig,
    bullet_speed: float,
) -> float:
    """
    车头对准目标点的塑形：scale × max(0, cos(Δθ))^power。

    power 越大，峰值越尖（只在接近 0° 时拿满奖）；背对（cos≤0）为 0。
    """
    if cfg.aim_align_scale == 0.0 or not me.alive:
        return 0.0
    tx, ty = _aim_target(me, enemy, prev_enemy, cfg.aim_mode, bullet_speed)
    target_angle = math.atan2(ty - me.y, tx - me.x)
    delta = _wrap_angle(target_angle - me.theta)
    cos_pos = max(0.0, math.cos(delta))
    power = max(1.0, float(cfg.aim_align_power))
    return cfg.aim_align_scale * (cos_pos**power)


def _aim_target(
    me: TankState,
    enemy: TankState,
    prev_enemy: TankState,
    mode: Literal["current", "lead"],
    bullet_speed: float,
) -> tuple[float, float]:
    if mode != "lead" or not enemy.alive:
        return enemy.x, enemy.y
    evx = enemy.x - prev_enemy.x
    evy = enemy.y - prev_enemy.y
    dist = math.hypot(enemy.x - me.x, enemy.y - me.y)
    speed = max(1e-6, bullet_speed)
    t = dist / speed
    return enemy.x + evx * t, enemy.y + evy * t


def _bullet_shaping_parts(
    me: TankState, curr: WorldState, cfg: RewardConfig
) -> tuple[float, float]:
    """返回 (bullet_near_enemy, bullet_threat_self)。"""
    near = 0.0
    threat = 0.0
    for b in curr.bullets:
        if b.owner != me.owner:
            continue
        enemy = curr.tanks[1] if me.owner == "red" else curr.tanks[0]
        de = math.hypot(b.x - enemy.x, b.y - enemy.y)
        if de < 80:
            near += cfg.bullet_near_enemy * (1.0 - de / 80.0)
        ds = math.hypot(b.x - me.x, b.y - me.y)
        if ds < 80:
            threat += cfg.bullet_threat_self * (1.0 - ds / 80.0)
    return near, threat


def _path_delta(
    me: TankState,
    enemy: TankState,
    curr: WorldState,
    side: str,
    cfg: RewardConfig,
    rstate: RewardState,
) -> float:
    """A* 路径长度变化奖励。"""
    if cfg.path_delta_scale == 0.0:
        return 0.0
    path, ok = astar(curr.game_map, (me.x, me.y), (enemy.x, enemy.y))
    if not ok:
        return 0.0
    length = float(len(path))
    key_prev = rstate.prev_path_len_red if side == "red" else rstate.prev_path_len_blue
    if key_prev is not None:
        delta = key_prev - length
        out = cfg.path_delta_scale * delta
    else:
        out = 0.0
    if side == "red":
        rstate.prev_path_len_red = length
    else:
        rstate.prev_path_len_blue = length
    return out


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a
