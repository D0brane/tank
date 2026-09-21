"""渲染器工厂：lite（训练） / showcase（展示）。"""

from __future__ import annotations

from typing import Any, Literal

from tank_sim.config import EnvConfig

RenderStyle = Literal["lite", "showcase"]


def make_renderer(
    cfg: EnvConfig,
    style: RenderStyle = "lite",
    scale: float | None = None,
) -> Any:
    """
    创建渲染器。

    - lite: 纯色几何，轻量，适合训练 rgb_array / 调试
    - showcase: 正版配色贴图，几何对齐碰撞，适合人玩与观战
    """
    if style == "showcase":
        from tank_sim.rendering.showcase import ShowcaseRenderer

        return ShowcaseRenderer(cfg, scale=2.0 if scale is None else scale)

    from tank_sim.rendering.pygame_backend import PygameRenderer

    # 轻量版 scale 保持 int 兼容
    sc = 2 if scale is None else int(scale)
    return PygameRenderer(cfg, scale=sc)
