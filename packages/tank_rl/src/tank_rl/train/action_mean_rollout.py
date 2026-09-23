"""在 SB3 rollout 中写入 VecEnv 的确定性动作均值缓冲。"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.utils import obs_as_tensor


def forward_actions_values_logprob_and_mean(
    policy, obs_tensor: th.Tensor
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """
    单次特征抽取：采样动作 + value + log_prob + 确定性动作。

    等价于 ``policy.forward`` + ``get_actions(deterministic=True)``，但只跑一遍网络。
    """
    features = policy.extract_features(obs_tensor)
    if policy.share_features_extractor:
        latent_pi, latent_vf = policy.mlp_extractor(features)
    else:
        pi_features, vf_features = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
        latent_vf = policy.mlp_extractor.forward_critic(vf_features)
    values = policy.value_net(latent_vf)
    distribution = policy._get_action_dist_from_latent(latent_pi)
    actions = distribution.get_actions(deterministic=False)
    log_prob = distribution.log_prob(actions)
    det = distribution.get_actions(deterministic=True)
    actions = actions.reshape((-1, *policy.action_space.shape))
    det = det.reshape((-1, *policy.action_space.shape))
    return actions, values, log_prob, det


def _collect_rollouts_with_action_means(
    self, env, callback, rollout_buffer, n_rollout_steps
):
    """与 SB3 OnPolicyAlgorithm.collect_rollouts 同步，并在 step 前记录确定性均值。"""
    assert self._last_obs is not None, "No previous observation was provided"
    self.policy.set_training_mode(False)

    n_steps = 0
    rollout_buffer.reset()
    if self.use_sde:
        self.policy.reset_noise(env.num_envs)

    callback.on_rollout_start()

    while n_steps < n_rollout_steps:
        if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
            self.policy.reset_noise(env.num_envs)

        with th.no_grad():
            obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
            actions, values, log_probs, det = forward_actions_values_logprob_and_mean(
                self.policy, obs_tensor
            )
            env.record_action_means(det.detach().cpu().numpy())
        actions = actions.cpu().numpy()

        clipped_actions = actions
        if isinstance(self.action_space, spaces.Box):
            if self.policy.squash_output:
                clipped_actions = self.policy.unscale_action(clipped_actions)
            else:
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        new_obs, rewards, dones, infos = env.step(clipped_actions)

        self.num_timesteps += env.num_envs

        callback.update_locals(locals())
        if not callback.on_step():
            return False

        self._update_info_buffer(infos, dones)
        n_steps += 1

        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.reshape(-1, 1)

        for idx, done in enumerate(dones):
            if (
                done
                and infos[idx].get("terminal_observation") is not None
                and infos[idx].get("TimeLimit.truncated", False)
            ):
                terminal_obs = self.policy.obs_to_tensor(
                    infos[idx]["terminal_observation"]
                )[0]
                with th.no_grad():
                    terminal_value = self.policy.predict_values(terminal_obs)[0]
                rewards[idx] += self.gamma * terminal_value

        rollout_buffer.add(
            self._last_obs,  # type: ignore[arg-type]
            actions,
            rewards,
            self._last_episode_starts,  # type: ignore[arg-type]
            values,
            log_probs,
        )
        self._last_obs = new_obs  # type: ignore[assignment]
        self._last_episode_starts = dones

    with th.no_grad():
        values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

    rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    callback.update_locals(locals())
    callback.on_rollout_end()

    return True


def bind_action_mean_rollout(model) -> None:
    """若训练 env 支持 ``record_action_means``，则 patch ``collect_rollouts``。"""
    env = model.get_env()
    if env is None or int(getattr(env, "_action_dim", 0)) <= 0:
        return
    model.collect_rollouts = types.MethodType(_collect_rollouts_with_action_means, model)
    model._tank_action_mean_rollout = True  # type: ignore[attr-defined]


def save_sb3_model(model: Any, path: str | Path) -> None:
    """
    存盘时排除实例上 patch 的 ``collect_rollouts`` 与 ``train``（否则 cloudpickle 会炸）。

    加载后训练入口会重新绑定这两处。
    """
    model.save(
        str(path),
        exclude=["collect_rollouts", "_tank_action_mean_rollout", "train"],
    )
