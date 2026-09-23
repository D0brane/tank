"""与训练评测一致的 StridedStack + DuelEnv 构造。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from tank_rl.curriculum.config import CurriculumAimConfig
from tank_rl.curriculum.scheduler import CurriculumScheduler
from tank_rl.train.curriculum_env import make_curriculum_env
from tank_rl.train.policy_mean import policy_mean_and_action
from tank_rl.vec_env.strided_stack import StridedStackEnv


def make_strided_duel_eval_env(
    cfg: CurriculumAimConfig,
    scheduler: CurriculumScheduler,
    *,
    render_mode: str | None = None,
    render_style: str | None = None,
    stack_action_mean: bool | None = None,
    max_episode_steps: int | None = None,
) -> StridedStackEnv:
    """训练 ``_eval_current`` 与观战共用：单帧 DuelEnv + 稀疏堆叠。"""
    env = make_curriculum_env(cfg, scheduler)
    if render_mode is not None:
        env.render_mode = render_mode
    if render_style is not None:
        env.render_style = render_style
    steps = max_episode_steps
    if steps is None and render_mode is None:
        # 无渲染的训练内评测：用课程 eval 步限；观战默认不改 sim
        steps = int(cfg.eval_max_episode_steps)
    if steps is not None and steps > 0:
        sim = replace(env.cfg.sim, max_episode_steps=int(steps))
        env.cfg = replace(env.cfg, sim=sim)
    use_mean = stack_action_mean if stack_action_mean is not None else cfg.stack_action_mean
    adim = 3 if use_mean else 0
    return StridedStackEnv(
        env,
        cfg.frame_stack,
        cfg.frame_stride,
        action_dim=adim,
    )


def sb3_predict_fn(model, obs: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
    """与训练评测 ``_eval_current`` 相同的 predict 路径。"""
    action, _ = model.predict(obs, deterministic=deterministic)
    return np.asarray(action, dtype=np.float32)


def make_eval_predict_fn(model, stack_env: StridedStackEnv):
    """评测/观战：单次前向取 deterministic mean（写入堆叠）与动作。"""

    def predict_fn(obs: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        if stack_env._action_dim > 0:
            mean, action = policy_mean_and_action(
                model, obs, deterministic=deterministic
            )
            stack_env.record_action_mean(mean)
            return np.asarray(action, dtype=np.float32)
        return sb3_predict_fn(model, obs, deterministic=deterministic)

    return predict_fn
