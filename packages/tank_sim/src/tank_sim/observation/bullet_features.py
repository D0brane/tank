"""子弹槽位观测：己/敌分库 + 存活期内 id 固定占槽；每槽航向角代替 vx,vy。"""

from __future__ import annotations

import math
from typing import Literal

from tank_sim.core.types import BulletState, TankState


class BulletSlotAssigner:
    """
    槽 0..N-1 己弹，N..2N-1 敌弹（N = slots_per_side）。

    存活子弹保持原槽；消亡后空槽可被新弹复用。
    """

    def __init__(self, slots_per_side: int = 5) -> None:
        if slots_per_side < 1:
            raise ValueError("slots_per_side 必须 ≥ 1")
        self.slots_per_side = slots_per_side
        self._own: list[int | None] = [None] * slots_per_side
        self._enemy: list[int | None] = [None] * slots_per_side

    def reset(self) -> None:
        self._own = [None] * self.slots_per_side
        self._enemy = [None] * self.slots_per_side

    def assign(
        self,
        observer_owner: Literal["red", "blue"],
        bullets: list[BulletState],
    ) -> list[BulletState | None]:
        """返回长度 2N 的槽位列表（空槽为 None）。"""
        own = [b for b in bullets if b.owner == observer_owner]
        enemy = [b for b in bullets if b.owner != observer_owner]
        self._sync_bank(self._own, own)
        self._sync_bank(self._enemy, enemy)
        by_id = {b.id: b for b in bullets if b.id != 0}
        # 无 id 的弹（测试手工塞入）按出现顺序临时占空槽，不写入持久映射
        out: list[BulletState | None] = []
        out.extend(self._bank_to_bullets(self._own, by_id, own))
        out.extend(self._bank_to_bullets(self._enemy, by_id, enemy))
        return out

    def _sync_bank(self, bank: list[int | None], live: list[BulletState]) -> None:
        live_ids = {b.id for b in live if b.id != 0}
        for i, bid in enumerate(bank):
            if bid is not None and bid not in live_ids:
                bank[i] = None
        assigned = {bid for bid in bank if bid is not None}
        for b in live:
            if b.id == 0 or b.id in assigned:
                continue
            for i, bid in enumerate(bank):
                if bid is None:
                    bank[i] = b.id
                    assigned.add(b.id)
                    break

    def _bank_to_bullets(
        self,
        bank: list[int | None],
        by_id: dict[int, BulletState],
        live: list[BulletState],
    ) -> list[BulletState | None]:
        slots: list[BulletState | None] = [
            by_id.get(bid) if bid is not None else None for bid in bank
        ]
        # id==0 的弹填尚未占用的空槽（仅本帧）
        free = [i for i, s in enumerate(slots) if s is None]
        for b in live:
            if b.id != 0:
                continue
            if not free:
                break
            slots[free.pop(0)] = b
        return slots


def build_bullet_features(
    observer: TankState,
    slots: list[BulletState | None],
    scale: float = 500.0,
) -> list[float]:
    """
    每槽 4 维：局部 dx, dy, φ/π, valid。

    φ 为局部系弹速航向；定速故不必存 |v| 或 (vx,vy)。
    """
    feats: list[float] = []
    for b in slots:
        if b is None:
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue
        dx, dy = _world_to_local(b.x - observer.x, b.y - observer.y, observer.theta)
        lvx, lvy = _world_to_local(b.vx, b.vy, observer.theta)
        phi = math.atan2(lvy, lvx)
        feats.extend(
            [
                dx / scale,
                dy / scale,
                phi / math.pi,
                1.0,
            ]
        )
    return feats


def _world_to_local(dx: float, dy: float, theta: float) -> tuple[float, float]:
    """世界向量 → 观察者局部坐标（+x 为车头）。"""
    c, s = math.cos(theta), math.sin(theta)
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    return lx, ly
