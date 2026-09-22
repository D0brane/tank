"""SB3 PPO checkpoint 加载，供观战 / 检查结果。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tank_rl.vec_env.strided_stack import StridedFrameBuffer, stacked_obs_dim
from tank_sim.observation.spec import expected_obs_dim


class PolicyBundle:
    """封装 PPO 模型 + 帧堆叠缓冲，输入单帧 obs，输出 shape (3,) 连续动作。"""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        frame_stack: int = 16,
        frame_stride: int = 1,
        device: str = "auto",
        expect_obs_dim: int | None = None,
        stack_action_mean: bool = False,
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
        self.frame_stride = max(1, int(frame_stride))
        self.stack_action_mean = bool(stack_action_mean)
        self.model = PPO.load(str(path), device=device)
        self._buffer: StridedFrameBuffer | None = None
        self._fresh_episode = True

        want = int(np.prod(self.model.observation_space.shape))
        self._expect_flat = want
        self._single_obs_dim: int | None = None
        action_dim = 3 if self.stack_action_mean else 0
        # 选择性堆叠（58 维单帧）或整帧整除堆叠（旧权重）
        for cand in (expected_obs_dim(10, 8),):
            if stacked_obs_dim(cand, self.frame_stack, action_dim=action_dim) == want:
                self._single_obs_dim = cand
                break
        if self._single_obs_dim is None and self.frame_stack > 1 and want % self.frame_stack == 0:
            # 兼容旧整帧堆叠权重（可能含或不含 action mean）
            rem = want
            if self.stack_action_mean:
                rem = want - 3 * self.frame_stack
            if rem > 0 and rem % self.frame_stack == 0:
                self._single_obs_dim = rem // self.frame_stack
        elif self._single_obs_dim is None and self.frame_stack <= 1:
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
                f"请加载与当前 env 同版本的 checkpoint（例如 58 维含雷达观测），"
                f"勿用旧的 50 维无雷达权重。"
            )

    def reset(self) -> None:
        """新对局开始时清空帧缓冲。"""
        self._buffer = None
        self._fresh_episode = True

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """单帧 obs → 连续动作 (3,)。"""
        frame = np.asarray(obs, dtype=np.float32).reshape(-1)
        stacked = self._stack(frame)
        action, _ = self.model.predict(stacked, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32).reshape(3)

    def _stack(self, frame: np.ndarray) -> np.ndarray:
        if self._expect_flat == frame.size:
            return frame
        if self.frame_stack <= 1 and self.frame_stride <= 1:
            if frame.size != self._expect_flat:
                raise ValueError(
                    f"单帧观测维数 {frame.size} ≠ 权重输入 {self._expect_flat}"
                )
            return frame
        if self._single_obs_dim is not None and frame.size != self._single_obs_dim:
            raise ValueError(
                f"单帧观测维数 {frame.size} ≠ 期望 {self._single_obs_dim}"
            )
        selective = self._single_obs_dim == expected_obs_dim(10, 8)
        if self._buffer is None:
            self._buffer = StridedFrameBuffer(
                self.frame_stack,
                self.frame_stride,
                self._single_obs_dim or frame.size,
                selective=selective,
            )
        if self._fresh_episode:
            self._buffer.reset(frame)
            self._fresh_episode = False
        else:
            self._buffer.append(frame)
        out = self._buffer.stacked()
        # 旧 PolicyBundle 路径不堆动作均值；若权重含均值段则报错提示
        if out.size != self._expect_flat:
            raise ValueError(
                f"堆叠后维数 {out.size} ≠ 权重输入 {self._expect_flat}"
                + (
                    "（权重可能含 stack_action_mean；请用 eval_watch 的 StridedStackEnv 路径）"
                    if self.stack_action_mean or out.size < self._expect_flat
                    else ""
                )
            )
        return out
