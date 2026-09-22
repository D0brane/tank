"""评测 kill_rate / hit_rate（按己方开火计） / median_ttk。"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from tank_rl.curriculum.scheduler import EvalMetrics


def aggregate_bullet_rates(
    *,
    bullets_fired: int,
    hits_on_enemy: int,
    direct_hits_on_enemy: int,
) -> tuple[float, float]:
    """kill_rate=命中次数/开火数；hit_rate=直击次数/开火数（无开火则为 0）。"""
    n = max(0, int(bullets_fired))
    if n <= 0:
        return 0.0, 0.0
    kill_rate = float(hits_on_enemy) / float(n)
    hit_rate = float(direct_hits_on_enemy) / float(n)
    return kill_rate, hit_rate


def evaluate_duel_policy(
    env,
    predict_fn: Callable[..., np.ndarray],
    *,
    n_episodes: int = 40,
    deterministic: bool = True,
    progress_every: int = 10,
) -> EvalMetrics:
    """
    对已配置好的 DuelEnv（或 StridedStackEnv 等）跑评测。

    predict_fn(obs, deterministic=...) -> action

    kill_rate = 己方命中敌方次数 / 己方开火数（含反弹命中，按 hit 事件计）。
    hit_rate = 己方未反弹命中敌方次数 / 己方开火数。

    局内首次己方直击即停止；TTK = 该步 step。无直击则用该局结束 step（评测步限封顶）。
    """
    total_bullets = 0
    total_hits = 0
    total_direct = 0
    ttks: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        info: dict[str, Any] = {}
        while not done:
            action = predict_fn(obs, deterministic=deterministic)
            obs, _reward, terminated, truncated, info = env.step(action)
            if int(info.get("direct_hits_on_enemy", 0)) > 0:
                done = True
            else:
                done = bool(terminated or truncated)
        ttk_ep = float(info.get("step", 0))
        ttks.append(ttk_ep)
        total_bullets += int(info.get("bullets_fired", 0))
        total_hits += int(info.get("hits_on_enemy", 0))
        total_direct += int(info.get("direct_hits_on_enemy", 0))
        if progress_every > 0 and (ep + 1) % progress_every == 0:
            k, h = aggregate_bullet_rates(
                bullets_fired=total_bullets,
                hits_on_enemy=total_hits,
                direct_hits_on_enemy=total_direct,
            )
            print(
                f"  [评测进度] {ep + 1}/{n_episodes}  "
                f"kill(命中/开火)={k:.3f}  direct(直击/开火)={h:.3f}  "
                f"弹数={total_bullets}"
            )

    kill_rate, hit_rate = aggregate_bullet_rates(
        bullets_fired=total_bullets,
        hits_on_enemy=total_hits,
        direct_hits_on_enemy=total_direct,
    )
    median_ttk = float(np.median(ttks)) if ttks else None
    return EvalMetrics(
        kill_rate=kill_rate,
        hit_rate=hit_rate,
        median_ttk=median_ttk,
        n_episodes=n_episodes,
    )
