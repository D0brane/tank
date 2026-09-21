"""车-车阻挡与窄道 OBB 测试。"""

from tank_sim.config import default_config_path, load_env_config
from tank_sim.control.action_mapping import continuous_to_intent
from tank_sim.core.collision import tanks_overlap
from tank_sim.core.types import ControlIntent, MoveIntent, RotateIntent
from tank_sim.core.world import create_initial_state, step_world
from tank_sim.core.tank import apply_translation
import numpy as np


def test_tanks_do_not_overlap_when_charging():
    """双方向对方冲锋后不应重叠。"""
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    # 红朝右、蓝朝左（默认），双方前进
    forward = continuous_to_intent(np.array([0.9, 0.0, -1.0]))
    for _ in range(400):
        state = step_world(state, forward, forward, cfg.sim)
        red, blue = state.tanks
        assert not tanks_overlap(red, blue, cfg.sim.tank)
        if abs(red.x - blue.x) < cfg.sim.tank.height:
            # 已顶住，再多走几步确认仍不穿
            for _ in range(20):
                state = step_world(state, forward, forward, cfg.sim)
                assert not tanks_overlap(state.tanks[0], state.tanks[1], cfg.sim.tank)
            break


def test_tank_blocked_by_wall_not_through():
    """贴墙前进不应穿入墙格。"""
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/maze_small.txt")
    from tank_sim.core.collision import tank_overlaps_wall

    red = state.tanks[0]
    # 朝左上墙顶
    red.theta = -1.5708  # 向上
    intent = ControlIntent(move=MoveIntent.FORWARD, rotate=RotateIntent.STOP, fire=False)
    for _ in range(200):
        apply_translation(red, intent, cfg.sim, state.game_map, other=state.tanks[1])
        assert not tank_overlaps_wall(red, cfg.sim.tank, state.game_map)
