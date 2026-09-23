"""把已训 PPO 当作 DuelEnv 对手（己方坐标系单帧 → 101 维堆叠）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tank_rl.train.policy_mean import policy_mean_and_action
from tank_rl.vec_env.strided_stack import StridedFrameBuffer
from tank_sim.observation.spec import expected_obs_dim


class StackedPolicyOpponent:
    """``act(obs, state, side)``：选择性堆叠 + 确定性动作均值，与训练评测一致。"""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        frame_stack: int = 3,
        frame_stride: int = 3,
        stack_action_mean: bool = True,
        device: str = "auto",
    ) -> None:
        from stable_baselines3 import PPO

        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"找不到模型文件: {path}")
        self.checkpoint = path
        self.frame_stack = max(1, int(frame_stack))
        self.frame_stride = max(1, int(frame_stride))
        self.stack_action_mean = bool(stack_action_mean)
        self.model = PPO.load(str(path), device=device)
        self._buf: StridedFrameBuffer | None = None
        self._mean_buf: StridedFrameBuffer | None = None
        self._pending_mean: np.ndarray | None = None
        self._fresh = True

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._buf = None
        self._mean_buf = None
        self._pending_mean = None
        self._fresh = True

    def act(self, obs: np.ndarray, state, side: str) -> np.ndarray:
        del state, side
        frame = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self._buf is None:
            selective = frame.size == expected_obs_dim(10, 8)
            self._buf = StridedFrameBuffer(
                self.frame_stack,
                self.frame_stride,
                frame.size,
                selective=selective,
            )
            if self.stack_action_mean:
                self._mean_buf = StridedFrameBuffer(
                    self.frame_stack, self.frame_stride, 3, selective=False
                )
        assert self._buf is not None
        if self._fresh:
            self._buf.reset(frame)
            if self._mean_buf is not None:
                self._mean_buf.reset(np.zeros(3, dtype=np.float32))
            self._fresh = False
        else:
            if self._mean_buf is not None:
                mean = (
                    self._pending_mean
                    if self._pending_mean is not None
                    else np.zeros(3, dtype=np.float32)
                )
                self._mean_buf.append(mean)
                self._pending_mean = None
            self._buf.append(frame)
        stacked = self._buf.stacked()
        if self._mean_buf is not None:
            stacked = np.concatenate([stacked, self._mean_buf.stacked()], axis=-1)
        mean, action = policy_mean_and_action(
            self.model, stacked, deterministic=True
        )
        if self._mean_buf is not None:
            self._pending_mean = np.asarray(mean, dtype=np.float32).reshape(3)
        return np.asarray(action, dtype=np.float32).reshape(3)
