"""奖励塑形（稠密 + 稀疏）。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from tank_sim.config import RewardConfig
from tank_sim.core.types import TankState, WorldState
from tank_sim.planning.astar import astar


@dataclass
class RewardState:
    """跨步缓存（路径距离等）。"""

    prev_path_len_red: float | None = None
    prev_path_len_blue: float | None = None


def compute_reward_for_side(
    prev: WorldState,
    curr: WorldState,
    side: str,
    cfg: RewardConfig,
    rstate: RewardState,
    *,
    bullet_speed: float = 3.77,
    fire_intent: bool = False,
) -> float:
    """计算单侧 agent 的本步奖励。"""
    red, blue = curr.tanks
    me: TankState = red if side == "red" else blue
    enemy: TankState = blue if side == "red" else red
    prev_enemy = prev.tanks[1] if side == "red" else prev.tanks[0]

    reward = cfg.survive_per_step

    fired = (side == "red" and curr.red_fired) or (side == "blue" and curr.blue_fired)
    if fired:
        reward += cfg.fire_penalty
    elif fire_intent and me.fire_cooldown > 0 and cfg.fire_on_cd_penalty != 0.0:
        # 意图开火但因冷却未射出（仍在 CD）；满弹未射且 CD=0 不走此项
        reward += cfg.fire_on_cd_penalty

    reward += _bullet_shaping(me, curr, cfg)
    reward += _aim_align(me, enemy, prev_enemy, cfg, bullet_speed)
    reward += _path_delta(me, enemy, curr, side, cfg, rstate)

    # 终局
    if curr.winner == side:
        reward += cfg.kill
    elif curr.winner != "none" and not me.alive:
        reward += cfg.death

    return reward


def _aim_align(
    me: TankState,
    enemy: TankState,
    prev_enemy: TankState,
    cfg: RewardConfig,
    bullet_speed: float,
) -> float:
    """车头对准目标点（当前位置或超前拦截点）的 cos 塑形。"""
    if cfg.aim_align_scale == 0.0 or not me.alive:
        return 0.0
    tx, ty = _aim_target(me, enemy, prev_enemy, cfg.aim_mode, bullet_speed)
    target_angle = math.atan2(ty - me.y, tx - me.x)
    delta = _wrap_angle(target_angle - me.theta)
    return cfg.aim_align_scale * math.cos(delta)


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


def _bullet_shaping(me: TankState, curr: WorldState, cfg: RewardConfig) -> float:
    """子弹逼近敌/己的稠密项。"""
    bonus = 0.0
    for b in curr.bullets:
        if b.owner != me.owner:
            continue
        # 己方子弹靠近敌人
        enemy = curr.tanks[1] if me.owner == "red" else curr.tanks[0]
        de = math.hypot(b.x - enemy.x, b.y - enemy.y)
        if de < 80:
            bonus += cfg.bullet_near_enemy * (1.0 - de / 80.0)
        # 任意子弹威胁自己
        ds = math.hypot(b.x - me.x, b.y - me.y)
        if ds < 80:
            bonus += cfg.bullet_threat_self * (1.0 - ds / 80.0)
    return bonus


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
