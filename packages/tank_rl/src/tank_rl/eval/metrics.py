"""评测 kill_rate / hit_rate / median_ttk。"""

from __future__ import annotations

from typing import Any

import numpy as np

from tank_rl.curriculum.scheduler import EvalMetrics


def evaluate_duel_policy(
    env,
    predict_fn,
    *,
    n_episodes: int = 200,
    deterministic: bool = True,
) -> EvalMetrics:
    """
    对已配置好的 DuelEnv（或 VecEnv 的单环境）跑评测。

    predict_fn(obs) -> action
    """
    wins = 0
    hits = 0
    ttks: list[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        info: dict[str, Any] = {}
        while not done:
            action = predict_fn(obs, deterministic=deterministic)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
        if info.get("agent_won"):
            wins += 1
            if info.get("ttk") is not None:
                ttks.append(float(info["ttk"]))
        if info.get("hit_enemy"):
            hits += 1

    median_ttk = float(np.median(ttks)) if ttks else None
    return EvalMetrics(
        kill_rate=wins / max(1, n_episodes),
        hit_rate=hits / max(1, n_episodes),
        median_ttk=median_ttk,
        n_episodes=n_episodes,
    )
