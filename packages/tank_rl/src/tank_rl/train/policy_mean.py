"""策略确定性动作均值（与 SB3 squash 后 [-1,1] 一致），供观测堆叠。"""

from __future__ import annotations

import numpy as np


def policy_deterministic_action_mean(model, obs: np.ndarray) -> np.ndarray:
    """
    对当前送入网络的观测，取 ``get_actions(deterministic=True)``。

    返回 shape ``(action_dim,)`` 或 ``(n_env, action_dim)``，与 ``obs`` 批维一致。
    """
    import torch as th
    from stable_baselines3.common.utils import obs_as_tensor

    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 1:
        obs_batch = obs_as_tensor(arr[None, ...], model.device)
        squeeze = True
    else:
        obs_batch = obs_as_tensor(arr, model.device)
        squeeze = False
    with th.no_grad():
        dist = model.policy.get_distribution(obs_batch)
        means = dist.get_actions(deterministic=True)
    out = means.detach().cpu().numpy().astype(np.float32)
    if squeeze and out.shape[0] == 1:
        return out[0]
    return out
