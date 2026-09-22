"""仿真核心数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal


class MoveIntent(IntEnum):
    """平移意图（定速）。"""

    STOP = 0
    FORWARD = 1
    BACKWARD = 2


class RotateIntent(IntEnum):
    """旋转意图（定速）。"""

    STOP = 0
    LEFT = 1
    RIGHT = 2


@dataclass
class ControlIntent:
    """一帧内的离散控制意图（由连续动作或键盘映射而来）。"""

    move: MoveIntent = MoveIntent.STOP
    rotate: RotateIntent = RotateIntent.STOP
    fire: bool = False


@dataclass
class TankState:
    """坦克状态。"""

    x: float
    y: float
    theta: float
    alive: bool = True
    # 剩余命中次数；开局由 hits_to_die 写入
    hp: int = 5
    fire_cooldown: int = 0
    # 冷却归零后已过去的帧数；开局取很大值，观测上视为已就绪满一个 CD 周期
    fire_ready_age: int = 10**9
    owner: Literal["red", "blue"] = "red"


@dataclass
class BulletState:
    """子弹状态。"""

    x: float
    y: float
    vx: float
    vy: float
    owner: Literal["red", "blue"]
    radius: float = 2.5
    age: int = 0
    # 全局递增 id，供观测槽位在存活期内保持稳定
    id: int = 0
    # 撞墙反射次数（用于区分直击 / 反弹击杀奖励）
    bounces: int = 0


@dataclass(frozen=True)
class WallRect:
    """轴对齐薄墙矩形（像素）。"""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.left + self.right)

    @property
    def cy(self) -> float:
        return 0.5 * (self.top + self.bottom)


@dataclass
class GameMap:
    """
    边墙迷宫：格子全部可走（除非 blocked），墙是格子间的薄分割线。

    - h_walls[r][c]：水平墙，位于 y = r * cell_px，横跨格列 c（r∈[0,rows]）
    - v_walls[r][c]：竖直墙，位于 x = c * cell_px，跨越格行 r（c∈[0,cols]）
    - wall_rects：碰撞用的薄 AABB 列表（由边墙生成）
    """

    cols: int
    rows: int
    cell_px: float
    wall_thickness: float
    h_walls: list[list[bool]]
    v_walls: list[list[bool]]
    blocked: list[list[bool]]
    wall_rects: list[WallRect]
    spawn_red: tuple[float, float]
    spawn_blue: tuple[float, float]

    @property
    def tile_px(self) -> int:
        """兼容旧代码：格距取整。"""
        return int(self.cell_px)

    @property
    def width(self) -> int:
        return self.cols

    @property
    def height(self) -> int:
        return self.rows

    @property
    def grid(self) -> list[list[int]]:
        """兼容：1=不可走格，0=可走（边墙不体现在 grid）。"""
        return [
            [1 if self.blocked[y][x] else 0 for x in range(self.cols)]
            for y in range(self.rows)
        ]


Winner = Literal["none", "red", "blue"]


@dataclass
class HitEvent:
    """本步一次子弹命中（含非致命），供 kill/death 逐步计分。"""

    attacker: Literal["red", "blue"]
    victim: Literal["red", "blue"]
    bounces: int = 0


@dataclass
class WorldState:
    """完整仿真状态（确定性步进）。"""

    step: int
    game_map: GameMap
    tanks: tuple[TankState, TankState]
    bullets: list[BulletState] = field(default_factory=list)
    winner: Winner = "none"
    # 下一发子弹的 id（从 1 起；0 表示未分配）
    next_bullet_id: int = 1
    # 本步事件标记，供奖励与 info 使用
    red_killed_by: Literal["none", "self", "enemy"] = "none"
    blue_killed_by: Literal["none", "self", "enemy"] = "none"
    red_fired: bool = False
    blue_fired: bool = False
    # 本步致死弹信息（供评测区分直击 / 反弹）
    kill_bullet_bounces: int = 0
    kill_bullet_owner: Literal["none", "red", "blue"] = "none"
    # 本步所有命中（每击一条；kill/death 按此计分）
    hit_events: list[HitEvent] = field(default_factory=list)

    @property
    def terminated(self) -> bool:
        return self.winner != "none"
