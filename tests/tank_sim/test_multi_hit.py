"""多命中死亡与命中消弹。"""

from tank_sim.config import default_config_path, load_env_config
from tank_sim.core.bullet import step_bullets
from tank_sim.core.types import BulletState, ControlIntent
from tank_sim.core.world import create_initial_state, step_world


def _point_blank_bullet(
    *,
    target_x: float,
    target_y: float,
    owner: str,
    bounces: int = 0,
    bid: int = 1,
) -> BulletState:
    return BulletState(
        x=target_x - 8.0,
        y=target_y,
        vx=10.0,
        vy=0.0,
        owner=owner,  # type: ignore[arg-type]
        radius=2.5,
        bounces=bounces,
        id=bid,
    )


def test_hits_to_die_default_is_five():
    cfg = load_env_config(default_config_path())
    assert cfg.sim.tank.hits_to_die == 5
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    assert state.tanks[0].hp == 5
    assert state.tanks[1].hp == 5


def test_four_hits_survive_fifth_kills():
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    red, blue = state.tanks
    idle = ControlIntent()

    for i in range(4):
        state.bullets = [
            _point_blank_bullet(
                target_x=blue.x, target_y=blue.y, owner="red", bid=i + 1
            )
        ]
        state = step_world(state, idle, idle, cfg.sim)
        red, blue = state.tanks
        assert blue.alive
        assert blue.hp == 5 - (i + 1)
        assert state.winner == "none"
        assert state.bullets == []

    state.bullets = [
        _point_blank_bullet(target_x=blue.x, target_y=blue.y, owner="red", bid=99)
    ]
    state = step_world(state, idle, idle, cfg.sim)
    assert not state.tanks[1].alive
    assert state.tanks[1].hp == 0
    assert state.winner == "red"
    assert state.kill_bullet_owner == "red"
    assert state.bullets == []


def test_hit_removes_bullet_immediately():
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    blue = state.tanks[1]
    state.bullets = [
        _point_blank_bullet(target_x=blue.x, target_y=blue.y, owner="red", bid=1)
    ]
    step_bullets(state, cfg.sim)
    assert state.bullets == []
    assert state.tanks[1].alive
    assert state.tanks[1].hp == 4


def test_friendly_fire_damages_self():
    cfg = load_env_config(default_config_path())
    state = create_initial_state(cfg, "assets/maps/empty.txt")
    red = state.tanks[0]
    state.bullets = [
        _point_blank_bullet(target_x=red.x, target_y=red.y, owner="red", bid=1)
    ]
    # 贴脸从左侧打红：子弹在 red.x-8，朝 +x，会撞到红
    step_bullets(state, cfg.sim)
    assert state.tanks[0].hp == 4
    assert state.tanks[0].alive
    assert state.winner == "none"
    assert state.bullets == []

    # 连扣至死 → 友伤自杀，蓝胜
    for i in range(4):
        state.bullets = [
            _point_blank_bullet(
                target_x=state.tanks[0].x,
                target_y=state.tanks[0].y,
                owner="red",
                bid=i + 2,
            )
        ]
        step_bullets(state, cfg.sim)
    assert not state.tanks[0].alive
    assert state.winner == "blue"
    assert state.red_killed_by == "self"
