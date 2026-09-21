"""物理步进冒烟测试与重叠不变量。"""

import numpy as np

from tank_sim.config import default_config_path, load_env_config
from tank_sim.control.action_mapping import continuous_to_intent
from tank_sim.core.collision import tank_overlaps_wall, tanks_overlap
from tank_sim.core.world import create_initial_state, step_world


def test_random_steps_no_crash():
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    rng = np.random.default_rng(0)
    for _ in range(200):
        ir = continuous_to_intent(rng.uniform(-1, 1, size=3))
        ib = continuous_to_intent(rng.uniform(-1, 1, size=3))
        state = step_world(state, ir, ib, cfg.sim)
        if state.terminated:
            break
    assert state.step > 0


def test_invariants_no_overlap_maze():
    """随机动作下：车不嵌墙、车车不重叠、坐标有限。"""
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/maze_small.txt")
    rng = np.random.default_rng(42)
    for _ in range(300):
        ir = continuous_to_intent(rng.uniform(-1, 1, size=3))
        ib = continuous_to_intent(rng.uniform(-1, 1, size=3))
        state = step_world(state, ir, ib, cfg.sim)
        red, blue = state.tanks
        for t in (red, blue):
            if t.alive:
                assert abs(t.x) < 1e6 and abs(t.y) < 1e6
                assert not tank_overlaps_wall(t, cfg.sim.tank, state.game_map)
        if red.alive and blue.alive:
            assert not tanks_overlap(red, blue, cfg.sim.tank)
        if state.terminated:
            break
