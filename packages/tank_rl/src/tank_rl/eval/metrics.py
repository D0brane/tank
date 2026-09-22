"""评测 kill_rate（总命中） / hit_rate（直击） / median_ttk。"""

from __future__ import annotations

from typing import Any

import numpy as np

from tank_rl.curriculum.scheduler import EvalMetrics


def evaluate_duel_policy(
    env,
    predict_fn,
    *,
    n_episodes: int = 40,
    deterministic: bool = True,
    progress_every: int = 10,
) -> EvalMetrics:
    """
    对已配置好的 DuelEnv（或 VecEnv 的单环境）跑评测。

    predict_fn(obs) -> action

    kill_rate = 总命中率：本局己弹至少命中敌方一次的局比例（info.hit_enemy；含反弹）。
    hit_rate = 直击率：本局至少一发己弹未反弹命中敌方（info.direct_hit；晋级默认看这个）。
    """
    hit_any = 0
    direct_hits = 0
    ttks: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        info: dict[str, Any] = {}
        while not done:
            action = predict_fn(obs, deterministic=deterministic)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
        if info.get("agent_won") and info.get("ttk") is not None:
            ttks.append(float(info["ttk"]))
        if info.get("hit_enemy"):
            hit_any += 1
        if info.get("direct_hit"):
            direct_hits += 1
        if progress_every > 0 and (ep + 1) % progress_every == 0:
            print(
                f"  [评测进度] {ep + 1}/{n_episodes}  "
                f"kill(总命中)={hit_any / (ep + 1):.2f}  "
                f"direct_hit={direct_hits / (ep + 1):.2f}"
            )

    median_ttk = float(np.median(ttks)) if ttks else None
    return EvalMetrics(
        kill_rate=hit_any / max(1, n_episodes),
        hit_rate=direct_hits / max(1, n_episodes),
        median_ttk=median_ttk,
        n_episodes=n_episodes,
    )
