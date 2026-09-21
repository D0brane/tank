"""SB3 PPO checkpoint 加载，供观战 / 检查结果。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PolicyBundle:
    """封装 PPO 模型 + 帧堆叠缓冲，输入单帧 obs，输出 shape (3,) 连续动作。"""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        frame_stack: int = 16,
        device: str = "auto",
    ) -> None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"找不到模型文件: {path}")

        try:
            from stable_baselines3 import PPO
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "需要 stable-baselines3。请执行: pip install -e '.[rl]'"
            ) from e

        self.checkpoint = path
        self.frame_stack = max(1, int(frame_stack))
        self.model = PPO.load(str(path), device=device)
        self._buf: list[np.ndarray] = []

        # 校验堆叠后维数与策略输入一致（允许未堆叠的旧权重）
        want = int(np.prod(self.model.observation_space.shape))
        self._expect_flat = want
        self._single_obs_dim: int | None = None
        if self.frame_stack > 1 and want % self.frame_stack == 0:
            self._single_obs_dim = want // self.frame_stack

    def reset(self) -> None:
        """新对局开始时清空帧缓冲。"""
        self._buf.clear()

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """单帧 obs → 连续动作 (3,)。"""
        frame = np.asarray(obs, dtype=np.float32).reshape(-1)
        stacked = self._stack(frame)
        action, _ = self.model.predict(stacked, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32).reshape(3)

    def _stack(self, frame: np.ndarray) -> np.ndarray:
        if self._expect_flat == frame.size:
            # 权重按单帧训练，或已传入叠好的向量
            return frame
        if self.frame_stack <= 1:
            return frame
        if not self._buf:
            self._buf = [frame.copy() for _ in range(self.frame_stack)]
        else:
            self._buf.append(frame.copy())
            self._buf = self._buf[-self.frame_stack :]
        out = np.concatenate(self._buf, axis=-1)
        if out.size != self._expect_flat and self._single_obs_dim is not None:
            # 维数仍不对则尽量截断/填充，避免硬崩
            if out.size > self._expect_flat:
                out = out[-self._expect_flat :]
            else:
                pad = np.zeros(self._expect_flat - out.size, dtype=np.float32)
                out = np.concatenate([out, pad], axis=0)
        return out.astype(np.float32)
