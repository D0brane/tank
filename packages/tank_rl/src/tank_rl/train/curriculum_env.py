"""课程 DuelEnv 工厂（避免 train ↔ eval 循环导入）。"""

from __future__ import annotations

from tank_sim.envs.duel_env import DuelEnv

from tank_rl.curriculum.config import CurriculumAimConfig
from tank_rl.curriculum.scheduler import CurriculumScheduler


def make_curriculum_env(cfg: CurriculumAimConfig, scheduler: CurriculumScheduler) -> DuelEnv:
    open_arena = None
    if cfg.arena.mode == "random_open":
        open_arena = cfg.arena.as_dict()
    env = DuelEnv(
        config_path=cfg.env_config,
        map_path=cfg.map_path,
        agent_side=cfg.agent_side,  # type: ignore[arg-type]
        opponent="curriculum",
        curriculum_bot=scheduler.bot,
        random_spawn=cfg.random_spawn,
        min_spawn_dist=cfg.min_spawn_dist,
        max_spawn_dist=cfg.max_spawn_dist,
        reward_overrides=scheduler.stage.reward,
        render_mode=None,
        open_arena=open_arena,
    )
    scheduler.attach_env(env)
    env._curriculum_scheduler = scheduler  # type: ignore[attr-defined]
    return env
