"""Tank Trouble 经典视觉调色板（展示层专用）。

参考：simple-tank / mglyn / OliverBryan — 浅灰地、深灰墙、黑弹；1v1 红 vs 蓝。
"""

from __future__ import annotations

FLOOR_A = (231, 231, 231)
FLOOR_B = (210, 210, 210)

WALL_FACE = (77, 77, 77)  # #4D4D4D

BULLET = (25, 25, 28)

TANK_RED = {
    "body": (220, 60, 50),
    "body_dark": (175, 42, 38),
    "turret": (190, 48, 42),
    "barrel": (115, 115, 120),
}

TANK_BLUE = {
    "body": (55, 115, 220),
    "body_dark": (35, 80, 175),
    "turret": (45, 95, 195),
    "barrel": (115, 115, 120),
}

OWNER_PALETTE = {
    "red": TANK_RED,
    "blue": TANK_BLUE,
}

HUD_TEXT = (40, 40, 45)
HUD_SHADOW = (255, 255, 255)
