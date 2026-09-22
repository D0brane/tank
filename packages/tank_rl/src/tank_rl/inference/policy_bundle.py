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
        expect_obs_dim: int | None = None,
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

        want = int(np.prod(self.model.observation_space.shape))
        self._expect_flat = want
        self._single_obs_dim: int | None = None
        if self.frame_stack > 1 and want % self.frame_stack == 0:
            self._single_obs_dim = want // self.frame_stack
        elif self.frame_stack <= 1:
            self._single_obs_dim = want

        if expect_obs_dim is not None:
            self.assert_obs_dim(expect_obs_dim)

    def assert_obs_dim(self, env_obs_dim: int) -> None:
        """环境单帧维数必须与权重一致（禁止静默 pad/截断）。"""
        if self._single_obs_dim is None:
            raise ValueError(
                f"无法从权重推断单帧维数：observation_space={self._expect_flat}, "
                f"frame_stack={self.frame_stack}"
            )
        if int(env_obs_dim) != int(self._single_obs_dim):
            raise ValueError(
                f"观测维数不匹配：环境单帧={env_obs_dim}，权重单帧={self._single_obs_dim} "
                f"（堆叠后权重输入={self._expect_flat}，frame_stack={self.frame_stack}）。"
                f"请加载与当前 env 同版本的 checkpoint（例如 50 维无雷达 / 58 维有雷达），"
                f"勿用旧的维数不匹配权重。"
            )

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
            return frame
        if self.frame_stack <= 1:
            if frame.size != self._expect_flat:
                raise ValueError(
                    f"单帧观测维数 {frame.size} ≠ 权重输入 {self._expect_flat}"
                )
            return frame
        if self._single_obs_dim is not None and frame.size != self._single_obs_dim:
            raise ValueError(
                f"单帧观测维数 {frame.size} ≠ 期望 {self._single_obs_dim}"
            )
        if not self._buf:
            # 对齐 SB3 VecFrameStack.reset：历史槽填 0，仅最新一帧写当前观测
            zeros = np.zeros_like(frame)
            self._buf = [zeros.copy() for _ in range(self.frame_stack - 1)] + [
                frame.copy()
            ]
        else:
            self._buf.append(frame.copy())
            self._buf = self._buf[-self.frame_stack :]
        out = np.concatenate(self._buf, axis=-1)
        if out.size != self._expect_flat:
            raise ValueError(
                f"堆叠后维数 {out.size} ≠ 权重输入 {self._expect_flat}"
            )
        return out.astype(np.float32)
