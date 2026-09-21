"""课程阶段调度：退火对手参数 + 按评测指标晋级。"""

from __future__ import annotations

from dataclasses import dataclass

from tank_sim.bots.curriculum_bot import CurriculumBot
from tank_sim.envs.duel_env import DuelEnv

from tank_rl.curriculum.config import CurriculumAimConfig, StageConfig


@dataclass
class EvalMetrics:
    kill_rate: float
    hit_rate: float
    median_ttk: float | None
    n_episodes: int


class CurriculumScheduler:
    """管理当前阶段、Bot/奖励同步与晋级判定。"""

    def __init__(self, cfg: CurriculumAimConfig) -> None:
        if not cfg.stages:
            raise ValueError("curriculum stages 不能为空")
        self.cfg = cfg
        self.stage_index = 0
        self.stage_timesteps = 0
        self.bot = CurriculumBot(mode=cfg.stages[0].bot.mode)
        self._apply_stage_bot(cfg.stages[0], progress=0.0)

    @property
    def stage(self) -> StageConfig:
        return self.cfg.stages[self.stage_index]

    @property
    def finished(self) -> bool:
        return self.stage_index >= len(self.cfg.stages)

    def attach_env(self, env: DuelEnv) -> None:
        """把当前阶段奖励与 Bot 应用到单个 DuelEnv。"""
        env._curriculum_bot = self.bot
        env._opponent = "curriculum"
        env.set_reward_overrides(self.stage.reward)
        env.random_spawn = self.cfg.random_spawn
        env.min_spawn_dist = self.cfg.min_spawn_dist
        env.max_spawn_dist = self.cfg.max_spawn_dist

    def on_timesteps(self, delta: int) -> None:
        """推进阶段内计时并退火 Bot 参数。"""
        self.stage_timesteps += max(0, delta)
        progress = self._anneal_progress(self.stage)
        self._apply_stage_bot(self.stage, progress)

    def maybe_promote(self, metrics: EvalMetrics) -> bool:
        """若达标则晋级；已是最后一阶段则返回 False。"""
        if self.stage_index >= len(self.cfg.stages) - 1:
            return False
        if not self._passed(self.stage, metrics):
            return False
        self.stage_index += 1
        self.stage_timesteps = 0
        self._apply_stage_bot(self.stage, progress=0.0)
        return True

    def _passed(self, stage: StageConfig, metrics: EvalMetrics) -> bool:
        pr = stage.promote
        value = metrics.kill_rate if pr.metric == "kill_rate" else metrics.hit_rate
        if value < pr.threshold:
            return False
        if pr.max_median_ttk is not None:
            if metrics.median_ttk is None or metrics.median_ttk > pr.max_median_ttk:
                return False
        progress = self._anneal_progress(stage)
        if pr.require_speed_scale is not None:
            cur = self._lerp(
                stage.bot.speed_scale,
                stage.bot.speed_scale_end
                if stage.bot.speed_scale_end is not None
                else stage.bot.speed_scale,
                progress,
            )
            if cur + 1e-6 < pr.require_speed_scale:
                return False
        if pr.require_mean_straight_frames is not None:
            cur = self._lerp(
                stage.bot.mean_straight_frames,
                stage.bot.mean_straight_frames_end
                if stage.bot.mean_straight_frames_end is not None
                else stage.bot.mean_straight_frames,
                progress,
            )
            # 直行间隔越小越难；要求已退火到不超过该值
            if cur > pr.require_mean_straight_frames + 1e-6:
                return False
        return True

    def _anneal_progress(self, stage: StageConfig) -> float:
        n = stage.bot.anneal_timesteps
        if n <= 0:
            return 1.0
        return min(1.0, self.stage_timesteps / float(n))

    def _apply_stage_bot(self, stage: StageConfig, progress: float) -> None:
        speed = self._lerp(
            stage.bot.speed_scale,
            stage.bot.speed_scale_end
            if stage.bot.speed_scale_end is not None
            else stage.bot.speed_scale,
            progress,
        )
        straight = self._lerp(
            stage.bot.mean_straight_frames,
            stage.bot.mean_straight_frames_end
            if stage.bot.mean_straight_frames_end is not None
            else stage.bot.mean_straight_frames,
            progress,
        )
        self.bot.configure(
            mode=stage.bot.mode,
            speed_scale=speed,
            mean_straight_frames=straight,
            turn_duration=stage.bot.turn_duration,
        )

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t
