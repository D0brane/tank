"""键盘 → 连续动作 [-1,1]^3（供人玩与手感调试）。"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

Scheme = Literal["wasd", "tt2_p1", "tt2_p2"]


class KeyState(Protocol):
    """与 pygame.key.get_pressed() 兼容的按下表。"""

    def __getitem__(self, key: int) -> bool: ...


def keys_to_action(keys: KeyState, scheme: Scheme) -> np.ndarray:
    """
    读取当前帧键盘，输出 PPO 使用的 3 维连续动作。

    经 action_mapping 后会变为定速前进/后退/旋转与开火。
    """
    if scheme == "wasd":
        return _scheme_wasd(keys)
    if scheme == "tt2_p1":
        return _scheme_tt2_p1(keys)
    if scheme == "tt2_p2":
        return _scheme_tt2_p2(keys)
    raise ValueError(f"未知键位方案: {scheme}")


def _scheme_wasd(keys: KeyState) -> np.ndarray:
    """WASD + Space（调试常用）。"""
    import pygame

    v, w, fire = 0.0, 0.0, -1.0
    if keys[pygame.K_w]:
        v = 0.85
    if keys[pygame.K_s]:
        v = -0.85
    if keys[pygame.K_a]:
        w = 0.9
    if keys[pygame.K_d]:
        w = -0.9
    if keys[pygame.K_SPACE]:
        fire = 0.85
    return np.array([v, w, fire], dtype=np.float32)


def _scheme_tt2_p1(keys: KeyState) -> np.ndarray:
    """Tank Trouble 2 多人默认：玩家1 — E/S/D/F 移动，Q 开火。"""
    import pygame

    v, w, fire = 0.0, 0.0, -1.0
    if keys[pygame.K_e]:
        v = 0.85
    if keys[pygame.K_d]:
        v = -0.85
    if keys[pygame.K_s]:
        w = 0.9
    if keys[pygame.K_f]:
        w = -0.9
    if keys[pygame.K_q]:
        fire = 0.85
    return np.array([v, w, fire], dtype=np.float32)


def _scheme_tt2_p2(keys: KeyState) -> np.ndarray:
    """Tank Trouble 2 多人默认：玩家2 — 方向键移动，M 开火。"""
    import pygame

    v, w, fire = 0.0, 0.0, -1.0
    if keys[pygame.K_UP]:
        v = 0.85
    if keys[pygame.K_DOWN]:
        v = -0.85
    if keys[pygame.K_LEFT]:
        w = 0.9
    if keys[pygame.K_RIGHT]:
        w = -0.9
    if keys[pygame.K_m]:
        fire = 0.85
    return np.array([v, w, fire], dtype=np.float32)


def scheme_for_side(side: Literal["red", "blue"], layout: Literal["wasd", "tt2"]) -> Scheme:
    """根据阵营与布局选择键位方案。"""
    if layout == "wasd":
        return "wasd"
    return "tt2_p1" if side == "red" else "tt2_p2"
