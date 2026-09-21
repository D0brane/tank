"""展示专用渲染：正版配色；坦克按 OBB 几何绘制（无贴图旋转晕边）。训练请用 lite。"""

from __future__ import annotations

import math

import numpy as np

from tank_sim.config import EnvConfig
from tank_sim.core.geometry import tank_corners, tank_half_extents
from tank_sim.core.types import WorldState
from tank_sim.rendering.showcase import palette as pal
from tank_sim.rendering.showcase.sprites import draw_checkered_floor, draw_wall_rect


class ShowcaseRenderer:
    """
    人玩 / 观战展示渲染。

    - 地面棋盘格、深灰边墙、红/蓝坦克、黑色圆弹
    - 坦克 = 物理 OBB 填充 + 炮塔 + 炮管（与碰撞重合，无贴图晕边）
    """

    def __init__(self, cfg: EnvConfig, scale: float = 2.0) -> None:
        import pygame

        if not pygame.get_init():
            pygame.init()
        self._pygame = pygame
        self._cfg = cfg
        self._scale = float(scale)
        self._screen = None
        self._surface_size: tuple[int, int] | None = None
        self.show_debug_hud = False
        self._hud_extra: list[str] = []

    def set_hud_extra(self, lines: list[str]) -> None:
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
                pygame.display.set_caption("Tank Trouble — 展示")
            else:
                self._screen = pygame.Surface((w, h))

        assert self._screen is not None
        draw_checkered_floor(
            self._screen, pygame, gm.cols, gm.rows, gm.cell_px, self._scale
        )

        for wr in gm.wall_rects:
            draw_wall_rect(
                self._screen, pygame, wr.left, wr.top, wr.right, wr.bottom, self._scale
            )

        for b in state.bullets:
            self._draw_bullet(b.x, b.y, b.radius)

        for tank in state.tanks:
            if tank.alive:
                self._draw_tank(tank.x, tank.y, tank.theta, tank.owner)

        if self.show_debug_hud:
            self._draw_hud(state)

        if mode == "human":
            pygame.display.flip()
            return None

        arr = pygame.surfarray.array3d(self._screen)
        return np.transpose(arr, (1, 0, 2))

    def _draw_bullet(self, x: float, y: float, radius: float) -> None:
        pygame = self._pygame
        s = self._scale
        pygame.draw.circle(
            self._screen,
            pal.BULLET,
            (int(x * s), int(y * s)),
            max(2, int(round(radius * s))),
        )

    def _draw_tank(self, x: float, y: float, theta: float, owner: str) -> None:
        """物理 OBB 填色 + 炮塔圆 + 多边形炮管（避免厚 line 的横竖端盖伪影）。"""
        pygame = self._pygame
        s = self._scale
        colors = pal.OWNER_PALETTE.get(owner, pal.TANK_RED)
        tw = self._cfg.sim.tank.width
        th = self._cfg.sim.tank.height
        hw, hh = tank_half_extents(tw, th)
        corners = tank_corners(x, y, theta, hw, hh)
        pts = [(int(px * s), int(py * s)) for px, py in corners]

        pygame.draw.polygon(self._screen, colors["body"], pts)

        cx, cy = x * s, y * s
        tr = max(2, int(round(min(tw, th) * 0.28 * s)))
        pygame.draw.circle(self._screen, colors["turret"], (int(cx), int(cy)), tr)

        # 炮管：从炮塔外缘伸到车头附近，用矩形四边形（无 draw.line）
        barrel_half_w = max(1.0, tw * 0.09 * s)
        tip_dist = hh * 0.95 * s
        root_dist = tr * 0.55  # 从炮塔内部接出，盖住接缝
        c, sn = math.cos(theta), math.sin(theta)
        # 侧向单位向量
        px, py = -sn, c

        root_x, root_y = cx + c * root_dist, cy + sn * root_dist
        tip_x, tip_y = cx + c * tip_dist, cy + sn * tip_dist
        barrel_pts = [
            (int(root_x + px * barrel_half_w), int(root_y + py * barrel_half_w)),
            (int(tip_x + px * barrel_half_w), int(tip_y + py * barrel_half_w)),
            (int(tip_x - px * barrel_half_w), int(tip_y - py * barrel_half_w)),
            (int(root_x - px * barrel_half_w), int(root_y - py * barrel_half_w)),
        ]
        pygame.draw.polygon(self._screen, colors["barrel"], barrel_pts)
        # 炮口圆帽
        pygame.draw.circle(
            self._screen,
            colors["barrel"],
            (int(tip_x), int(tip_y)),
            max(1, int(round(barrel_half_w))),
        )
        # 炮塔再画一层，盖住根部接缝
        pygame.draw.circle(self._screen, colors["turret"], (int(cx), int(cy)), tr)

    def _draw_hud(self, state: WorldState) -> None:
        import pygame

        font = pygame.font.SysFont("sans-serif", 14)
        red, blue = state.tanks
        lines = [
            f"step {state.step}",
            f"子弹 {len(state.bullets)}",
            f"红 CD {red.fire_cooldown}  蓝 CD {blue.fire_cooldown}",
            f"OBB {self._cfg.sim.tank.width:.0f}×{self._cfg.sim.tank.height:.0f}  "
            f"墙厚 {state.game_map.wall_thickness:.0f}",
            f"移速 {self._cfg.sim.tank.speed_forward}/帧  "
            f"弹速 {self._cfg.sim.bullet.speed}/帧",
        ]
        lines.extend(self._hud_extra)
        y = 4
        for line in lines:
            shadow = font.render(line, True, pal.HUD_SHADOW)
            text = font.render(line, True, pal.HUD_TEXT)
            self._screen.blit(shadow, (7, y + 1))
            self._screen.blit(text, (6, y))
            y += 16

    def close(self) -> None:
        if self._screen is not None:
            self._pygame.display.quit()
        self._screen = None
