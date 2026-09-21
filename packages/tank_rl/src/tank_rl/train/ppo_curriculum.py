"""SB3 PPO + 瞄准课程训练循环（中文终端状态 + TensorBoard）。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tank_sim.envs.duel_env import DuelEnv

from tank_rl.curriculum.config import CurriculumAimConfig, load_curriculum_aim_config
from tank_rl.curriculum.scheduler import CurriculumScheduler, EvalMetrics
from tank_rl.eval.metrics import evaluate_duel_policy


def make_curriculum_env(cfg: CurriculumAimConfig, scheduler: CurriculumScheduler) -> DuelEnv:
    env = DuelEnv(
        config_path=cfg.env_config,
        map_path=cfg.map_path,
        agent_side=cfg.agent_side,  # type: ignore[arg-type]
        opponent="curriculum",
        curriculum_bot=scheduler.bot,
        random_spawn=cfg.random_spawn,
        min_spawn_dist=cfg.min_spawn_dist,
        max_spawn_dist=cfg.max_spawn_dist,
        reward_overrides=scheduler.stage.reward,
        render_mode=None,
    )
    scheduler.attach_env(env)
    env._curriculum_scheduler = scheduler  # type: ignore[attr-defined]
    return env


def train_curriculum_aim(
    config_path: str | Path = "configs/train/curriculum_aim.yaml",
    *,
    log_dir: str | Path = "runs/curriculum_aim",
    device: str = "auto",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    """
    运行瞄准课程 PPO 训练；返回 SB3 模型。

    需要可选依赖：pip install -e '.[rl]'
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "训练需要 stable-baselines3 / torch。请执行: pip install -e '.[rl]'"
        ) from e

    config_path = Path(config_path)
    cfg = load_curriculum_aim_config(config_path)
    scheduler = CurriculumScheduler(cfg)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    from tank_sim.config import load_env_config as _load_env

    obs_dim = _load_env(cfg.env_config).obs.dim
    _print_banner(config_path, log_dir, device, cfg, scheduler, obs_dim)

    def _thunk(rank: int):
        def _init():
            local = CurriculumScheduler(cfg)
            local.stage_index = scheduler.stage_index
            local.stage_timesteps = scheduler.stage_timesteps
            local._apply_stage_bot(local.stage, local._anneal_progress(local.stage))
            env = make_curriculum_env(cfg, local)
            env.reset(seed=cfg.seed + rank)
            return env

        return _init

    n_envs = max(1, cfg.n_envs)
    vec = DummyVecEnv([_thunk(i) for i in range(n_envs)])
    vec = VecFrameStack(vec, n_stack=cfg.frame_stack)

    net_arch = cfg.ppo.get("net_arch", [64, 64])
    model = PPO(
        "MlpPolicy",
        vec,
        learning_rate=float(cfg.ppo.get("learning_rate", 3e-4)),
        n_steps=int(cfg.ppo.get("n_steps", 2048)),
        batch_size=int(cfg.ppo.get("batch_size", 256)),
        gamma=float(cfg.ppo.get("gamma", 0.99)),
        ent_coef=float(cfg.ppo.get("ent_coef", 0.01)),
        policy_kwargs={"net_arch": net_arch},
        seed=cfg.seed,
        verbose=0,  # 课程回调自行打印中文状态
        device=device,
        tensorboard_log=str(log_dir),
    )

    class CurriculumCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self._last_eval = 0
            self._last_ckpt = 0
            self._t0 = time.perf_counter()
            self._steps0 = 0

        def _on_training_start(self) -> None:
            self._t0 = time.perf_counter()
            self._steps0 = 0

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            delta = int(cfg.ppo.get("n_steps", 2048)) * n_envs
            scheduler.on_timesteps(delta)
            _sync_dummy_schedulers(vec, scheduler)

            t = int(self.num_timesteps)
            elapsed = max(1e-6, time.perf_counter() - self._t0)
            fps = (t - self._steps0) / elapsed if t > self._steps0 else 0.0
            # 滑动窗口：每 rollout 重置计时起点，避免全程均值掩盖近期 FPS
            self._t0 = time.perf_counter()
            self._steps0 = t

            aim = float(scheduler.stage.reward.get("aim_align_scale", 0.0))
            _log_curriculum_tb(self.logger, scheduler, aim, promoted=None)
            self.logger.dump(t)

            pct = 100.0 * t / max(1, cfg.total_timesteps)
            print(
                f"[进度] {t:,}/{cfg.total_timesteps:,} ({pct:.1f}%)  "
                f"FPS≈{fps:.0f}  "
                f"阶段={scheduler.stage.name}  "
                f"速度={scheduler.bot.speed_scale:.2f}  "
                f"直行间隔={scheduler.bot.mean_straight_frames:.0f}  "
                f"aim={aim:.3f}/{scheduler.stage.reward.get('aim_mode', '?')}"
            )

            ckpt_every = max(0, int(cfg.checkpoint_every_timesteps))
            if ckpt_every > 0 and t - self._last_ckpt >= ckpt_every:
                self._last_ckpt = t
                path = log_dir / "checkpoints" / (
                    f"model_t{t}_stage{scheduler.stage_index}_{scheduler.stage.name}.zip"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(path))
                print(f"[存盘] {path}")

            if t - self._last_eval < cfg.eval_every_timesteps:
                return

            self._last_eval = t
            print(f"[评测] 开始（{cfg.eval_n_episodes} 局）…")
            stage_idx_before = scheduler.stage_index
            metrics = _eval_current(model, cfg, scheduler)
            prev_name = scheduler.stage.name
            promoted = scheduler.maybe_promote(metrics)
            if promoted:
                _sync_dummy_schedulers(vec, scheduler)

            _log_eval_tb(self.logger, metrics, promoted=promoted)
            _log_curriculum_tb(
                self.logger,
                scheduler,
                float(scheduler.stage.reward.get("aim_align_scale", 0.0)),
                promoted=promoted,
            )
            self.logger.dump(t)

            _print_eval_block(
                prev_name,
                scheduler,
                metrics,
                promoted,
                promote_stage_index=stage_idx_before,
            )
            payload = {
                "timesteps": t,
                "stage": scheduler.stage.name,
                "stage_index": scheduler.stage_index,
                "kill_rate": metrics.kill_rate,
                "hit_rate": metrics.hit_rate,
                "median_ttk": metrics.median_ttk,
                "promoted": promoted,
            }
            if progress_callback:
                progress_callback(payload)

    model.learn(total_timesteps=cfg.total_timesteps, callback=CurriculumCallback())
    ckpt = log_dir / "final_model.zip"
    model.save(str(ckpt))
    _print_finish(scheduler, ckpt, log_dir)
    return model


def _print_banner(
    config_path: Path,
    log_dir: Path,
    device: str,
    cfg: CurriculumAimConfig,
    scheduler: CurriculumScheduler,
    obs_dim: int,
) -> None:
    stage = scheduler.stage
    print("=" * 60)
    print("瞄准课程 PPO 训练")
    print("=" * 60)
    print(f"  配置文件     : {config_path}")
    print(f"  日志目录     : {log_dir}")
    print(f"  设备         : {device}")
    print(f"  总步数       : {cfg.total_timesteps:,}")
    print(f"  并行环境     : {cfg.n_envs}")
    print(f"  帧堆叠       : {cfg.frame_stack} → 输入维 {obs_dim * cfg.frame_stack}")
    print(f"  评测间隔     : 每 {cfg.eval_every_timesteps:,} 步 / {cfg.eval_n_episodes} 局")
    print(f"  存盘间隔     : 每 {cfg.checkpoint_every_timesteps:,} 步 → {log_dir}/checkpoints/")
    print("-" * 60)
    print(f"  当前阶段     : {stage.name} ({scheduler.stage_index + 1}/{len(cfg.stages)})")
    print(f"  对手         : mode={scheduler.bot.mode}  speed={scheduler.bot.speed_scale:.2f}")
    print(
        f"  奖励         : aim={stage.reward.get('aim_align_scale')} "
        f"mode={stage.reward.get('aim_mode')}  kill={stage.reward.get('kill')}"
    )
    print("-" * 60)
    print(f"  TensorBoard  : tensorboard --logdir {log_dir}")
    print(f"  浏览器打开   : http://localhost:6006")
    print("=" * 60)


def _print_eval_block(
    prev_name: str,
    scheduler: CurriculumScheduler,
    metrics: EvalMetrics,
    promoted: bool,
    *,
    promote_stage_index: int,
) -> None:
    pr = scheduler.cfg.stages[promote_stage_index].promote
    value = metrics.kill_rate if pr.metric == "kill_rate" else metrics.hit_rate
    ttk_s = f"{metrics.median_ttk:.0f}" if metrics.median_ttk is not None else "无"
    print("-" * 60)
    print(f"[评测] 阶段={prev_name}  局数={metrics.n_episodes}")
    print(
        f"  击杀率 kill_rate = {metrics.kill_rate:.3f}  "
        f"命中率 hit_rate = {metrics.hit_rate:.3f}  "
        f"中位TTK = {ttk_s}"
    )
    print(
        f"  晋级条件 {pr.metric}≥{pr.threshold:.2f}  "
        f"当前={value:.3f}  → {'达标' if value >= pr.threshold else '未达标'}"
    )
    if promoted:
        s = scheduler.stage
        print(f"  ★ 晋级成功：{prev_name} → {s.name}")
        print(
            f"    新对手 mode={scheduler.bot.mode}  "
            f"speed={scheduler.bot.speed_scale:.2f}  "
            f"直行间隔={scheduler.bot.mean_straight_frames:.0f}"
        )
        print(
            f"    新奖励 aim={s.reward.get('aim_align_scale')} "
            f"mode={s.reward.get('aim_mode')}"
        )
    else:
        print("  未晋级，继续当前阶段")
    print("-" * 60)


def _print_finish(scheduler: CurriculumScheduler, ckpt: Path, log_dir: Path) -> None:
    print("=" * 60)
    print("训练结束")
    print(f"  最终阶段 : {scheduler.stage.name} (#{scheduler.stage_index})")
    print(f"  模型保存 : {ckpt}")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")
    print("=" * 60)


def _log_curriculum_tb(logger, scheduler: CurriculumScheduler, aim: float, promoted: bool | None) -> None:
    logger.record("curriculum/stage_index", float(scheduler.stage_index))
    logger.record("curriculum/speed_scale", float(scheduler.bot.speed_scale))
    logger.record(
        "curriculum/mean_straight_frames",
        float(scheduler.bot.mean_straight_frames),
    )
    logger.record("curriculum/aim_align_scale", float(aim))
    if promoted is not None:
        logger.record("curriculum/promoted", 1.0 if promoted else 0.0)


def _log_eval_tb(logger, metrics: EvalMetrics, *, promoted: bool) -> None:
    logger.record("eval/kill_rate", float(metrics.kill_rate))
    logger.record("eval/hit_rate", float(metrics.hit_rate))
    logger.record(
        "eval/median_ttk",
        float(metrics.median_ttk) if metrics.median_ttk is not None else -1.0,
    )
    logger.record("curriculum/promoted", 1.0 if promoted else 0.0)


def _sync_dummy_schedulers(vec, master: CurriculumScheduler) -> None:
    base = vec
    while hasattr(base, "venv"):
        base = base.venv
    for e in getattr(base, "envs", []):
        raw = e
        while hasattr(raw, "env"):
            raw = raw.env
        sch = getattr(raw, "_curriculum_scheduler", None)
        if sch is None:
            continue
        sch.stage_index = master.stage_index
        sch.stage_timesteps = master.stage_timesteps
        sch.bot.configure(
            mode=master.bot.mode,
            speed_scale=master.bot.speed_scale,
            mean_straight_frames=master.bot.mean_straight_frames,
            turn_duration=master.bot.turn_duration,
        )
        sch.attach_env(raw)


def _eval_current(model, cfg: CurriculumAimConfig, scheduler: CurriculumScheduler):
    eval_sched = CurriculumScheduler(cfg)
    eval_sched.stage_index = scheduler.stage_index
    eval_sched.stage_timesteps = scheduler.stage_timesteps
    eval_sched._apply_stage_bot(eval_sched.stage, eval_sched._anneal_progress(eval_sched.stage))
    eval_sched.bot.configure(
        mode=scheduler.bot.mode,
        speed_scale=scheduler.bot.speed_scale,
        mean_straight_frames=scheduler.bot.mean_straight_frames,
        turn_duration=scheduler.bot.turn_duration,
    )
    env = make_curriculum_env(cfg, eval_sched)
    stacked = _ConcatFrameStack(env, cfg.frame_stack)

    def predict_fn(obs, deterministic=True):
        action, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    return evaluate_duel_policy(
        stacked,
        predict_fn,
        n_episodes=cfg.eval_n_episodes,
    )


class _ConcatFrameStack:
    """把最近 k 帧观测在最后一维拼接，对齐 SB3 VecFrameStack(1D)。"""

    def __init__(self, env: DuelEnv, k: int) -> None:
        self.env = env
        self.k = k
        self._buf: list[np.ndarray] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buf = [obs.copy() for _ in range(self.k)]
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buf.append(obs.copy())
        self._buf = self._buf[-self.k :]
        return self._stacked(), reward, terminated, truncated, info

    def _stacked(self) -> np.ndarray:
        return np.concatenate(self._buf, axis=-1).astype(np.float32)
