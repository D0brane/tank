"""子弹寿命与同时在场上限。"""

from tank_sim.config import default_config_path, load_env_config
from tank_sim.core.types import BulletState, ControlIntent
from tank_sim.core.world import create_initial_state, step_world
from tank_sim.control.action_mapping import continuous_to_intent
import numpy as np


def test_bullet_expires_after_lifetime():
    """正版约 10s @60 → 600 帧后子弹应消失。"""
    cfg = load_env_config(default_config_path())
    assert cfg.sim.bullet.lifetime_frames == 600
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    # 塞入一发已接近寿命的弹
    life = cfg.sim.bullet.lifetime_frames
    state.bullets.append(
        BulletState(
            x=state.tanks[0].x + 40,
            y=state.tanks[0].y,
            vx=0.0,
            vy=0.0,
            owner="red",
            radius=cfg.sim.bullet.radius,
            age=life - 2,
        )
    )
    idle = continuous_to_intent(np.array([0.0, 0.0, -1.0]))
    state = step_world(state, idle, idle, cfg.sim)
    assert len(state.bullets) == 1
    assert state.bullets[0].age == life - 1
    state = step_world(state, idle, idle, cfg.sim)
    assert len(state.bullets) == 0


def test_max_active_bullets_per_tank():
    """每车同时最多 5 发；满弹再开火时挤掉最早己方弹（FIFO）。"""
    cfg = load_env_config(default_config_path())
    assert cfg.sim.bullet.max_active_per_tank == 5
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    # 预先放满 5 发：age/id 递增，最早为 id=1 age=50
    for i in range(5):
        state.bullets.append(
            BulletState(
                x=state.tanks[0].x + 20 + i,
                y=state.tanks[0].y,
                vx=0.1,
                vy=0.0,
                owner="red",
                radius=cfg.sim.bullet.radius,
                age=50 - i * 10,
                id=i + 1,
            )
        )
    oldest_id = 1
    state.next_bullet_id = 100
    fire = ControlIntent(fire=True)
    idle = ControlIntent()
    state.tanks[0].fire_cooldown = 0
    state = step_world(state, fire, idle, cfg.sim)
    red = [b for b in state.bullets if b.owner == "red"]
    assert state.red_fired
    assert len(red) == 5
    assert all(b.id != oldest_id for b in red)
    # step_bullets 后新弹 age 至少为 1；id 为发放时的 next_bullet_id
    assert any(b.id == 100 for b in red)
