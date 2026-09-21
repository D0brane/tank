"""展示层绘制辅助：地面 / 墙（坦克改为几何直接画，避免贴图旋转晕边）。"""

from __future__ import annotations

from typing import Any

from tank_sim.rendering.showcase import palette as pal


def draw_checkered_floor(
    screen: Any,
    pygame: Any,
    cols: int,
    rows: int,
    cell_px: float,
    scale: float,
) -> None:
    """棋盘格地面。"""
    s = scale
    for ty in range(rows):
        for tx in range(cols):
            color = pal.FLOOR_A if (tx + ty) % 2 == 0 else pal.FLOOR_B
            rect = pygame.Rect(
                int(tx * cell_px * s),
                int(ty * cell_px * s),
                int(cell_px * s) + 1,
                int(cell_px * s) + 1,
            )
            pygame.draw.rect(screen, color, rect)


def draw_wall_rect(
    screen: Any,
    pygame: Any,
    left: float,
    top: float,
    right: float,
    bottom: float,
    scale: float,
) -> None:
    """与碰撞 AABB 完全一致的墙块；纯色填充，无描边。"""
    s = scale
    rect = pygame.Rect(
        int(left * s),
        int(top * s),
        max(1, int((right - left) * s)),
        max(1, int((bottom - top) * s)),
    )
    pygame.draw.rect(screen, pal.WALL_FACE, rect)
