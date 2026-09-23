"""策略确定性动作均值（与 SB3 squash 后 [-1,1] 一致），供观测堆叠。"""

from __future__ import annotations

import numpy as np


def policy_deterministic_action_mean(model, obs: np.ndarray) -> np.ndarray:
    """
    对当前送入网络的观测，取 ``get_actions(deterministic=True)``。

    返回 shape ``(action_dim,)`` 或 ``(n_env, action_dim)``，与 ``obs`` 批维一致。
    """
    mean, _ = policy_mean_and_action(model, obs, deterministic=True)
    return mean


def policy_mean_and_action(
    model,
    obs: np.ndarray,
    *,
    deterministic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    单次前向：确定性 mean + 用于 env.step 的动作。

    deterministic=True 时两者相同；False 时动作为采样、mean 仍为确定性。
    """
    import torch as th
    from stable_baselines3.common.utils import obs_as_tensor

    from tank_rl.train.action_mean_rollout import forward_actions_values_logprob_and_mean

    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 1:
        obs_batch = obs_as_tensor(arr[None, ...], model.device)
        squeeze = True
    else:
        obs_batch = obs_as_tensor(arr, model.device)
        squeeze = False

    with th.no_grad():
        if deterministic:
            # 确定性评测：一次 get_distribution 即可（比完整 actor-critic 更轻）
            dist = model.policy.get_distribution(obs_batch)
            means = dist.get_actions(deterministic=True)
            actions = means
        else:
            actions_t, _values, _lp, means = forward_actions_values_logprob_and_mean(
                model.policy, obs_batch
            )
            actions = actions_t

    mean_np = means.detach().cpu().numpy().astype(np.float32)
    act_np = actions.detach().cpu().numpy().astype(np.float32)
    if squeeze and mean_np.shape[0] == 1:
        return mean_np[0], act_np[0]
    return mean_np, act_np
