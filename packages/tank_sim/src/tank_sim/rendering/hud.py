"""调试 HUD：手感检查时在画面上显示状态。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tank_sim.config import EnvConfig
    from tank_sim.core.types import WorldState


def draw_debug_hud(screen, cfg: "EnvConfig", state: "WorldState", lines_extra: list[str] | None = None) -> None:
    """在 pygame 画面上绘制文字信息。"""
    import pygame

    font = pygame.font.SysFont("sans-serif", 14)
    red, blue = state.tanks
    y = 4
    base = [
        f"step {state.step}",
        f"子弹数 {len(state.bullets)}",
        f"红 CD {red.fire_cooldown}  蓝 CD {blue.fire_cooldown}",
        f"车体 OBB {cfg.sim.tank.width:.0f}x{cfg.sim.tank.height:.0f}",
        f"移速 {cfg.sim.tank.speed_forward}/帧  弹速 {cfg.sim.bullet.speed}/帧  CD {cfg.sim.fire_cooldown_frames}帧",
    ]
    if lines_extra:
        base.extend(lines_extra)
    for line in base:
        surf = font.render(line, True, (230, 230, 230))
        screen.blit(surf, (6, y))
        y += 16
