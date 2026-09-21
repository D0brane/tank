"""轻量 Pygame 渲染（训练 / 调试）：纯色几何，无贴图。

人玩展示请用 `rendering.showcase.ShowcaseRenderer`（factory style=showcase）。
"""

from __future__ import annotations

import math

import numpy as np

from tank_sim.config import EnvConfig
from tank_sim.core.geometry import tank_corners, tank_half_extents
from tank_sim.core.types import WorldState


class PygameRenderer:
    """将 WorldState 绘制到窗口或返回 rgb_array。"""

    def __init__(self, cfg: EnvConfig, scale: int = 2) -> None:
        import pygame

        if not pygame.get_init():
            pygame.init()
        self._pygame = pygame
        self._cfg = cfg
        self._scale = scale
        self._screen = None
        self._clock = pygame.time.Clock()
        self._surface_size: tuple[int, int] | None = None
        self.show_debug_hud = False
        self._hud_extra: list[str] = []

    def set_hud_extra(self, lines: list[str]) -> None:
        """追加 HUD 说明行（例如当前控制模式）。"""
        self._hud_extra = lines

    def render(self, state: WorldState, mode: str = "human") -> np.ndarray | None:
        import pygame

        gm = state.game_map
        w = int(gm.cols * gm.cell_px * self._scale)
        h = int(gm.rows * gm.cell_px * self._scale)
        if self._surface_size != (w, h):
            self._surface_size = (w, h)
            if mode == "human":
                self._screen = pygame.display.set_mode((w, h))
                pygame.display.set_caption("Tank 对战")
            else:
                self._screen = pygame.Surface((w, h))

        assert self._screen is not None
        # 地面
        self._screen.fill((210, 210, 215))

        # 淡格线（可选氛围）
        s = self._scale
        for c in range(gm.cols + 1):
            x = int(c * gm.cell_px * s)
            pygame.draw.line(self._screen, (200, 200, 205), (x, 0), (x, h), 1)
        for r in range(gm.rows + 1):
            y = int(r * gm.cell_px * s)
            pygame.draw.line(self._screen, (200, 200, 205), (0, y), (w, y), 1)

        # 边墙：细分割线
        wall_color = (55, 55, 60)
        for wr in gm.wall_rects:
            rect = pygame.Rect(
                int(wr.left * s),
                int(wr.top * s),
                max(1, int((wr.right - wr.left) * s)),
                max(1, int((wr.bottom - wr.top) * s)),
            )
            pygame.draw.rect(self._screen, wall_color, rect)

        for b in state.bullets:
            self._draw_circle(b.x, b.y, b.radius, (240, 220, 80))

        red, blue = state.tanks
        if red.alive:
            self._draw_tank(red.x, red.y, red.theta, (220, 70, 70))
        if blue.alive:
            self._draw_tank(blue.x, blue.y, blue.theta, (70, 120, 220))

        if self.show_debug_hud and self._screen is not None:
            from tank_sim.rendering.hud import draw_debug_hud

            draw_debug_hud(self._screen, self._cfg, state, self._hud_extra)

        if mode == "human":
            # 只翻页；事件与帧率由 apps.play / showcase 主循环负责，
            # 此处若调用 event.get() 会吞掉 R/Esc 等按键。
            pygame.display.flip()
        else:
            arr = pygame.surfarray.array3d(self._screen)
            return np.transpose(arr, (1, 0, 2))

        return None

    def _draw_circle(self, x: float, y: float, r: float, color) -> None:
        pygame = self._pygame
        s = self._scale
        pygame.draw.circle(
            self._screen,
            color,
            (int(x * s), int(y * s)),
            max(2, int(r * s)),
        )

    def _draw_tank(self, x: float, y: float, theta: float, color) -> None:
        """绘制与物理 OBB 一致的旋转矩形 + 炮管。"""
        pygame = self._pygame
        s = self._scale
        tw = self._cfg.sim.tank.width
        th = self._cfg.sim.tank.height
        hw, hh = tank_half_extents(tw, th)
        corners = tank_corners(x, y, theta, hw, hh)
        pts = [(int(px * s), int(py * s)) for px, py in corners]
        pygame.draw.polygon(self._screen, color, pts)
        pygame.draw.polygon(self._screen, (240, 240, 240), pts, 1)
        # 炮管沿车头
        cx, cy = int(x * s), int(y * s)
        barrel = int(hh * s * 1.15)
        lx = cx + int(math.cos(theta) * barrel)
        ly = cy + int(math.sin(theta) * barrel)
        pygame.draw.line(self._screen, (240, 240, 240), (cx, cy), (lx, ly), 2)

    def close(self) -> None:
        if self._screen is not None:
            self._pygame.display.quit()
        self._screen = None
