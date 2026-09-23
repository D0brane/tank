"""稀疏帧堆叠：按 stride 从连续历史中采样；观测为选择性子块堆叠。"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper

from tank_sim.observation.spec import (
    CURRENT_ONLY_SLICES,
    STACKABLE_SLICES,
    expected_obs_dim,
    selective_stacked_dim,
)


def stack_depth(n_stack: int, stride: int) -> int:
    """维持采样所需的最短连续历史长度。"""
    k = max(1, int(n_stack))
    s = max(1, int(stride))
    return (k - 1) * s + 1


def stack_lags(n_stack: int, stride: int) -> list[int]:
    """相对当前帧的 lag 列表，从旧到新，如 k=3,s=3 → [6,3,0]。"""
    k = max(1, int(n_stack))
    s = max(1, int(stride))
    return [s * (k - 1 - i) for i in range(k)]


def _slice_concat(frame: np.ndarray, slices: tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.concatenate([frame[a:b] for a, b in slices], axis=-1)


class StridedFrameBuffer:
    """单环境稀疏堆叠缓冲。

    ``selective=True``（默认用于 58 维观测）：堆叠 enemy+walls+planner，
    己方开火与子弹仅取当前帧。
    ``selective=False``：整帧拼接（用于动作均值历史）。
    """

    def __init__(
        self,
        n_stack: int,
        stride: int,
        frame_dim: int,
        *,
        selective: bool = False,
    ) -> None:
        self.n_stack = max(1, int(n_stack))
        self.stride = max(1, int(stride))
        self.frame_dim = int(frame_dim)
        self.selective = bool(selective)
        self.depth = stack_depth(self.n_stack, self.stride)
        self.lags = stack_lags(self.n_stack, self.stride)
        self._frames: list[np.ndarray] = []
        self._zero = np.zeros(self.frame_dim, dtype=np.float32)
        if self.selective:
            expect = expected_obs_dim(10, 8)
            if self.frame_dim != expect:
                raise ValueError(
                    f"选择性堆叠要求单帧维={expect}，得到 {self.frame_dim}"
                )
            self._out_dim = selective_stacked_dim(self.n_stack)
        else:
            self._out_dim = self.frame_dim * self.n_stack

    def clear(self) -> None:
        self._frames.clear()

    def reset(self, frame: np.ndarray) -> None:
        """新 episode 首帧（不足 lag 用零填充）。"""
        self._frames = [np.asarray(frame, dtype=np.float32).reshape(-1).copy()]

    def append(self, frame: np.ndarray) -> None:
        self._frames.append(np.asarray(frame, dtype=np.float32).reshape(-1).copy())
        if len(self._frames) > self.depth:
            self._frames = self._frames[-self.depth :]

    def _frame_at_lag(self, lag: int) -> np.ndarray:
        n = len(self._frames)
        if n > lag:
            return self._frames[-(lag + 1)]
        return self._zero

    def stacked(self) -> np.ndarray:
        if not self.selective:
            parts = [self._frame_at_lag(lag) for lag in self.lags]
            out = np.concatenate(parts, axis=-1)
        else:
            parts = [
                _slice_concat(self._frame_at_lag(lag), STACKABLE_SLICES)
                for lag in self.lags
            ]
            cur = self._frame_at_lag(0)
            parts.append(_slice_concat(cur, CURRENT_ONLY_SLICES))
            out = np.concatenate(parts, axis=-1)
        if out.size != self._out_dim:
            raise RuntimeError(f"堆叠维数错误: {out.size} != {self._out_dim}")
        return out.astype(np.float32)


def stacked_obs_dim(frame_dim: int, n_stack: int, *, action_dim: int = 0) -> int:
    """选择性稀疏堆叠维 + 可选动作均值堆叠维。"""
    k = max(1, int(n_stack))
    if int(frame_dim) == expected_obs_dim(10, 8):
        base = selective_stacked_dim(k)
    else:
        # 非标准单帧（测试假环境等）：整帧堆叠
        base = int(frame_dim) * k
    if action_dim > 0:
        base += int(action_dim) * k
    return base


def _selective_obs_bounds(
    low: np.ndarray, high: np.ndarray, n_stack: int
) -> tuple[np.ndarray, np.ndarray]:
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for _ in range(n_stack):
        for a, b in STACKABLE_SLICES:
            lows.append(low[a:b])
            highs.append(high[a:b])
    for a, b in CURRENT_ONLY_SLICES:
        lows.append(low[a:b])
        highs.append(high[a:b])
    return np.concatenate(lows), np.concatenate(highs)


class StridedStackEnv:
    """评测用：包装 DuelEnv，step/reset 返回堆叠观测（可选堆确定性动作均值）。"""

    def __init__(self, env, n_stack: int, stride: int, *, action_dim: int = 0) -> None:
        self.env = env
        self._action_dim = max(0, int(action_dim))
        frame_dim = int(np.prod(env.observation_space.shape))
        selective = frame_dim == expected_obs_dim(10, 8)
        self._buf = StridedFrameBuffer(
            n_stack, stride, frame_dim, selective=selective
        )
        self._mean_buf: StridedFrameBuffer | None = None
        self._pending_mean: np.ndarray | None = None
        if self._action_dim > 0:
            self._mean_buf = StridedFrameBuffer(
                n_stack, stride, self._action_dim, selective=False
            )

    def record_action_mean(self, mean: np.ndarray) -> None:
        """本步 ``step`` 前调用：写入与当前决策对应的确定性均值。"""
        if self._action_dim <= 0:
            return
        self._pending_mean = np.asarray(mean, dtype=np.float32).reshape(self._action_dim)

    def _compose_obs(self) -> np.ndarray:
        frames = self._buf.stacked()
        if self._mean_buf is None:
            return frames
        return np.concatenate([frames, self._mean_buf.stacked()], axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buf.reset(obs)
        self._pending_mean = None
        if self._mean_buf is not None:
            self._mean_buf.reset(np.zeros(self._action_dim, dtype=np.float32))
        return self._compose_obs(), info

    def step(self, action):
        if self._mean_buf is not None:
            if self._pending_mean is None:
                self._pending_mean = np.zeros(self._action_dim, dtype=np.float32)
            self._mean_buf.append(self._pending_mean)
            self._pending_mean = None
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buf.append(obs)
        return self._compose_obs(), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()


class VecStridedFrameStackWrapper(VecEnvWrapper):
    """SB3 VecEnv 稀疏/选择性堆叠（模块级类，便于 pickle）。"""

    def __init__(
        self,
        venv,
        n_stack: int,
        stride: int,
        *,
        action_dim: int = 0,
        observation_space: gym.Space,
        frame_dim: int,
        selective: bool,
    ) -> None:
        super().__init__(venv, observation_space=observation_space)
        self._n_stack = max(1, int(n_stack))
        self._stride = max(1, int(stride))
        self._action_dim = max(0, int(action_dim))
        self._frame_dim = int(frame_dim)
        self._selective = bool(selective)
        self.buffers = [
            StridedFrameBuffer(
                self._n_stack, self._stride, self._frame_dim, selective=self._selective
            )
            for _ in range(self.num_envs)
        ]
        self.mean_buffers: list[StridedFrameBuffer] | None = None
        if self._action_dim > 0:
            self.mean_buffers = [
                StridedFrameBuffer(
                    self._n_stack, self._stride, self._action_dim, selective=False
                )
                for _ in range(self.num_envs)
            ]
        self._pending_means = np.zeros(
            (self.num_envs, self._action_dim), dtype=np.float32
        )

    def record_action_means(self, means: np.ndarray) -> None:
        m = np.asarray(means, dtype=np.float32).reshape(self.num_envs, self._action_dim)
        self._pending_means = m

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        for i, o in enumerate(obs):
            self.buffers[i].reset(np.asarray(o))
            if self.mean_buffers is not None:
                self.mean_buffers[i].reset(
                    np.zeros(self._action_dim, dtype=np.float32)
                )
        self._pending_means = np.zeros(
            (self.num_envs, self._action_dim), dtype=np.float32
        )
        return self._stacked_obs()

    def step_wait(self) -> tuple:
        obs, rewards, dones, infos = self.venv.step_wait()
        for i, o in enumerate(obs):
            oa = np.asarray(o)
            if self.mean_buffers is not None:
                self.mean_buffers[i].append(self._pending_means[i])
            if dones[i]:
                if "terminal_observation" in infos[i]:
                    term = np.asarray(infos[i]["terminal_observation"], dtype=np.float32)
                    self.buffers[i].append(term.reshape(-1))
                    infos[i]["terminal_observation"] = self._compose(i).copy()
                self.buffers[i].reset(oa)
                if self.mean_buffers is not None:
                    self.mean_buffers[i].reset(
                        np.zeros(self._action_dim, dtype=np.float32)
                    )
            else:
                self.buffers[i].append(oa)
        self._pending_means = np.zeros(
            (self.num_envs, self._action_dim), dtype=np.float32
        )
        return self._stacked_obs(), rewards, dones, infos

    def _compose(self, i: int) -> np.ndarray:
        frames = self.buffers[i].stacked()
        if self.mean_buffers is None:
            return frames
        return np.concatenate([frames, self.mean_buffers[i].stacked()], axis=-1)

    def _stacked_obs(self) -> np.ndarray:
        return np.stack([self._compose(i) for i in range(self.num_envs)], axis=0)


def VecStridedFrameStack(
    venv,
    n_stack: int,
    stride: int,
    *,
    action_dim: int = 0,
) -> VecEnvWrapper:
    """SB3 VecEnv 稀疏堆叠（stride=1 时 lags 为连续 k 帧）。"""
    n_stack = max(1, int(n_stack))
    stride = max(1, int(stride))
    frame_shape = venv.observation_space.shape
    assert frame_shape is not None
    frame_dim = int(np.prod(frame_shape))
    adim = max(0, int(action_dim))
    selective = frame_dim == expected_obs_dim(10, 8)
    if selective:
        low, high = _selective_obs_bounds(
            np.asarray(venv.observation_space.low).reshape(-1),
            np.asarray(venv.observation_space.high).reshape(-1),
            n_stack,
        )
    else:
        low = np.repeat(np.asarray(venv.observation_space.low).reshape(-1), n_stack)
        high = np.repeat(np.asarray(venv.observation_space.high).reshape(-1), n_stack)
    if adim > 0:
        act_low = np.full((adim * n_stack,), -1.0, dtype=venv.observation_space.dtype)
        act_high = np.full((adim * n_stack,), 1.0, dtype=venv.observation_space.dtype)
        low = np.concatenate([low, act_low])
        high = np.concatenate([high, act_high])
    obs_space = gym.spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
    return VecStridedFrameStackWrapper(
        venv,
        n_stack,
        stride,
        action_dim=adim,
        observation_space=obs_space,
        frame_dim=frame_dim,
        selective=selective,
    )
