"""SB3 PPO + 瞄准课程训练循环（中文终端状态 + TensorBoard）。"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tank_rl.curriculum.config import CurriculumAimConfig, load_curriculum_aim_config
from tank_rl.curriculum.scheduler import CurriculumScheduler, EvalMetrics
from tank_rl.eval.duel_eval_env import make_eval_predict_fn, make_strided_duel_eval_env
from tank_rl.eval.metrics import evaluate_duel_policy
from tank_rl.train.action_mean_rollout import bind_action_mean_rollout, save_sb3_model
from tank_rl.train.curriculum_env import make_curriculum_env
from tank_rl.vec_env.strided_stack import VecStridedFrameStack, stacked_obs_dim


def train_curriculum_aim(
    config_path: str | Path = "configs/train/curriculum_aim.yaml",
    *,
    log_dir: str | Path = "runs/curriculum_aim",
    device: str = "auto",
    resume: str | Path | None = None,
    new_run: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    """
    运行瞄准课程 PPO 训练；返回 SB3 模型。

    ``log_dir`` 为实验根目录；新训时在其下新建 ``YYYYMMDD_HHMMSS/``
    存放 checkpoint 与 TensorBoard。

    ``resume`` 为权重 ``.zip`` 或某次 run 目录时：恢复权重与课程阶段
    （``reset_num_timesteps=False``，总步数目标仍为配置中的
    ``total_timesteps``）。默认写回原 run；``new_run=True`` 时在
    ``log_dir`` 下新建时间戳目录，并把起点权重存为 ``named/model_0``（0号模型）。

    需要可选依赖：pip install -e '.[rl]'
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.utils import FloatSchedule
        from stable_baselines3.common.vec_env import DummyVecEnv

    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "训练需要 stable-baselines3 / torch。请执行: pip install -e '.[rl]'"
        ) from e

    from tank_rl.checkpoint_meta import (
        apply_meta_to_scheduler,
        advance_scheduler_from_promotion,
        load_curriculum_meta,
        parse_stage_index_from_filename,
        parse_timesteps_from_filename,
        resolve_resume_target,
    )

    config_path = Path(config_path)
    cfg = load_curriculum_aim_config(config_path)
    scheduler = CurriculumScheduler(cfg)

    resume_ckpt: Path | None = None
    start_timesteps = 0
    if resume is not None:
        resume_ckpt, old_run_dir = resolve_resume_target(resume)
        meta = load_curriculum_meta(resume_ckpt)
        if meta is not None:
            apply_meta_to_scheduler(scheduler, meta)
            start_timesteps = int(meta.timesteps)
            if meta.frame_stack != cfg.frame_stack:
                raise ValueError(
                    f"续训 frame_stack 不匹配：权重 meta={meta.frame_stack}，"
                    f"配置={cfg.frame_stack}"
                )
            meta_stride = int(getattr(meta, "frame_stride", 1))
            if meta_stride != cfg.frame_stride:
                raise ValueError(
                    f"续训 frame_stride 不匹配：权重 meta={meta_stride}，"
                    f"配置={cfg.frame_stride}"
                )
            meta_mean = bool(getattr(meta, "stack_action_mean", False))
            if meta_mean != cfg.stack_action_mean:
                raise ValueError(
                    f"续训 stack_action_mean 不匹配：权重 meta={meta_mean}，"
                    f"配置={cfg.stack_action_mean}"
                )
            prom = advance_scheduler_from_promotion(scheduler, resume_ckpt)
            if prom is not None:
                print(
                    f"[续训] 晋级权重：{prom['completed_stage_name']} → "
                    f"{scheduler.stage.name}（退火起点）",
                    flush=True,
                )
        else:
            # 无 meta：尽量从文件名恢复阶段；对手用阶段退火进度近似
            stage_idx = parse_stage_index_from_filename(resume_ckpt)
            if stage_idx is not None and 0 <= stage_idx < len(cfg.stages):
                scheduler.stage_index = stage_idx
                scheduler.stage_timesteps = 0
                scheduler._apply_stage_bot(scheduler.stage, progress=1.0)
            parsed_t = parse_timesteps_from_filename(resume_ckpt)
            if parsed_t is not None:
                start_timesteps = parsed_t
            print(
                f"[续训] 警告：无 .curriculum.json，阶段/对手可能不准确 "
                f"（{resume_ckpt.name}）"
            )
        if new_run:
            run_dir = _make_timestamped_run_dir(Path(log_dir))
            seeded = _seed_named_model_zero(run_dir, resume_ckpt)
            resume_ckpt = seeded
            print(
                f"[续训] 新时间戳 run（自 {old_run_dir.name}）→ {run_dir.name}；"
                f"起点已存为 named/model_0（0号模型）",
                flush=True,
            )
        else:
            run_dir = old_run_dir
            run_dir.mkdir(parents=True, exist_ok=True)
    else:
        if new_run:
            print("[续训] 警告：未指定 --resume，--new-run 无效", flush=True)
        run_dir = _make_timestamped_run_dir(Path(log_dir))

    from tank_sim.config import load_env_config as _load_env

    obs_dim = _load_env(cfg.env_config).obs.dim
    _print_banner(
        config_path,
        run_dir,
        device,
        cfg,
        scheduler,
        obs_dim,
        resume_ckpt=resume_ckpt,
        start_timesteps=start_timesteps,
    )

    def _thunk(rank: int):
        def _init():
            local = CurriculumScheduler(cfg)
            local.stage_index = scheduler.stage_index
            local.stage_timesteps = scheduler.stage_timesteps
            local.bot.configure(
                mode=scheduler.bot.mode,
                speed_scale=scheduler.bot.speed_scale,
                mean_straight_frames=scheduler.bot.mean_straight_frames,
                turn_duration=scheduler.bot.turn_duration,
            )
            env = make_curriculum_env(cfg, local)
            env.reset(seed=cfg.seed + rank)
            return env

        return _init

    n_envs = max(1, cfg.n_envs)
    vec = DummyVecEnv([_thunk(i) for i in range(n_envs)])
    action_dim = 3 if cfg.stack_action_mean else 0
    vec = VecStridedFrameStack(
        vec, cfg.frame_stack, cfg.frame_stride, action_dim=action_dim
    )

    net_arch = cfg.ppo.get("net_arch", [64, 64])
    ent_start = float(cfg.ppo.get("ent_coef", 0.01))
    ent_end = float(cfg.ppo.get("ent_coef_end", ent_start))
    ent_decay = int(cfg.ppo.get("ent_coef_decay_timesteps", 0))
    log_std_min = float(cfg.ppo.get("log_std_min", -5.0))
    log_std_max = float(cfg.ppo.get("log_std_max", 0.0))
    # 熵衰减相对「本次开训/续训起点」计步
    ent_decay_t0 = start_timesteps

    if resume_ckpt is not None:
        model = PPO.load(str(resume_ckpt), env=vec, device=device)
        # 续训时 TensorBoard 仍写同一 run 目录
        model.tensorboard_log = str(run_dir)
        # 用当前 yaml 覆盖存盘超参（否则仍用 ckpt 内旧 ent_coef / lr）
        # SB3 实际用 lr_schedule，只改 learning_rate 属性不会生效
        lr = float(cfg.ppo.get("learning_rate", 3e-4))
        model.learning_rate = lr
        model.lr_schedule = FloatSchedule(lr)
        model.ent_coef = ent_start
        std_before = _policy_action_std(model)
        _clamp_policy_log_std(model, log_std_min, log_std_max)
        std_after = _policy_action_std(model)
        print(
            f"[续训] 覆盖超参 learning_rate={lr:g}  "
            f"ent_coef={ent_start:g}→{ent_end:g}（{ent_decay:,} 步衰减）  "
            f"log_std∈[{log_std_min:g},{log_std_max:g}]  "
            f"std {std_before:.3f}→{std_after:.3f}",
            flush=True,
        )
        # SB3 存盘通常已带 num_timesteps；若缺失则用 meta/文件名
        if int(getattr(model, "num_timesteps", 0) or 0) <= 0 and start_timesteps > 0:
            model.num_timesteps = start_timesteps
        start_timesteps = int(model.num_timesteps)
        ent_decay_t0 = start_timesteps
        if start_timesteps >= cfg.total_timesteps:
            raise ValueError(
                f"续训步数 t={start_timesteps:,} 已达到/超过配置 "
                f"total_timesteps={cfg.total_timesteps:,}；请增大 yaml 总步数"
            )
    else:
        model = PPO(
            "MlpPolicy",
            vec,
            learning_rate=float(cfg.ppo.get("learning_rate", 3e-4)),
            n_steps=int(cfg.ppo.get("n_steps", 2048)),
            batch_size=int(cfg.ppo.get("batch_size", 256)),
            gamma=float(cfg.ppo.get("gamma", 0.99)),
            ent_coef=ent_start,
            policy_kwargs={
                "net_arch": net_arch,
                "log_std_init": min(0.0, log_std_max),
            },
            seed=cfg.seed,
            verbose=0,  # 课程回调自行打印中文状态
            device=device,
            tensorboard_log=str(run_dir),
        )
        _clamp_policy_log_std(model, log_std_min, log_std_max)
        ent_decay_t0 = 0

    if cfg.stack_action_mean:
        bind_action_mean_rollout(model)

    # 每次 PPO 更新后钳制 log_std（更新过程中梯度可能顶破上限）
    _orig_train = model.train

    def _train_and_clamp(*args, **kwargs):
        _orig_train(*args, **kwargs)
        _clamp_policy_log_std(model, log_std_min, log_std_max)

    model.train = _train_and_clamp  # type: ignore[method-assign]

    class CurriculumCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self._last_eval = start_timesteps
            self._last_ckpt = start_timesteps
            self._t0 = time.perf_counter()
            self._steps0 = start_timesteps
            self._rew_sum: dict[str, float] = {}
            self._rew_count = 0
            self._stop_requested = False

        def _on_training_start(self) -> None:
            self._t0 = time.perf_counter()
            self._steps0 = int(self.num_timesteps)
            self._rew_sum.clear()
            self._rew_count = 0
            # 确保各 env 奖励与转向阶梯一致
            _sync_dummy_schedulers(vec, scheduler)

        def _on_step(self) -> bool:
            if self._stop_requested:
                return False
            infos = self.locals.get("infos")
            if not infos:
                return True
            for info in infos:
                if not isinstance(info, dict):
                    continue
                parts = info.get("reward_parts")
                if not isinstance(parts, dict):
                    continue
                self._rew_count += 1
                for k, v in parts.items():
                    self._rew_sum[k] = self._rew_sum.get(k, 0.0) + float(v)
            return True

        def _on_rollout_end(self) -> None:
            delta = int(cfg.ppo.get("n_steps", 2048)) * n_envs
            scheduler.on_timesteps(delta)
            _sync_dummy_schedulers(vec, scheduler)

            t = int(self.num_timesteps)
            # 熵线性衰减 + 每 rollout 钳制 log_std（防 std 爆炸）
            self.model.ent_coef = _scheduled_ent_coef(
                t, ent_start, ent_end, ent_decay_t0, ent_decay
            )
            _clamp_policy_log_std(self.model, log_std_min, log_std_max)

            elapsed = max(1e-6, time.perf_counter() - self._t0)
            fps = (t - self._steps0) / elapsed if t > self._steps0 else 0.0

            aim = float(scheduler.stage.reward.get("aim_align_scale", 0.0))
            _log_curriculum_tb(self.logger, scheduler, aim, promoted=None)
            self.logger.record("train/ent_coef", float(self.model.ent_coef))
            self.logger.record("train/action_std_clamped", _policy_action_std(self.model))
            _log_reward_parts_tb(self.logger, self._rew_sum, self._rew_count)
            self._rew_sum.clear()
            self._rew_count = 0
            self.logger.dump(t)

            pct = 100.0 * t / max(1, cfg.total_timesteps)
            print(
                f"[进度] {t:,}/{cfg.total_timesteps:,} ({pct:.1f}%)  "
                f"FPS≈{fps:.0f}  "
                f"阶段={scheduler.stage.name}  "
                f"速度={scheduler.bot.speed_scale:.2f}  "
                f"直行间隔={scheduler.bot.mean_straight_frames:.0f}  "
                f"aim={aim:.3f}/{scheduler.stage.reward.get('aim_mode', '?')}  "
                f"rotate={scheduler.rotate_penalty:g}  "
                f"ent={float(self.model.ent_coef):g}  "
                f"std={_policy_action_std(self.model):.3f}"
            )

            ckpt_every = max(0, int(cfg.checkpoint_every_timesteps))
            if ckpt_every > 0 and t - self._last_ckpt >= ckpt_every:
                self._last_ckpt = t
                path = run_dir / "checkpoints" / (
                    f"model_t{t}_stage{scheduler.stage_index}_{scheduler.stage.name}.zip"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                save_sb3_model(self.model, path)
                _save_ckpt_meta(
                    path,
                    scheduler,
                    timesteps=t,
                    frame_stack=cfg.frame_stack,
                    frame_stride=cfg.frame_stride,
                    stack_action_mean=cfg.stack_action_mean,
                )
                _update_latest_symlink(run_dir, path)
                print(f"[存盘] {path}")

            if t - self._last_eval >= cfg.eval_every_timesteps:
                self._last_eval = t
                print(f"[评测] 开始（{cfg.eval_n_episodes} 局）…")
                stage_idx_before = scheduler.stage_index
                prev_name = scheduler.stage.name
                stage_steps_before = scheduler.stage_timesteps
                bot_snap = (
                    scheduler.bot.mode,
                    float(scheduler.bot.speed_scale),
                    float(scheduler.bot.mean_straight_frames),
                    int(scheduler.bot.turn_duration),
                )
                metrics = _eval_current(model, cfg, scheduler)
                rot_result = scheduler.maybe_advance_rotate(metrics)
                if rot_result == "advanced":
                    _sync_dummy_schedulers(vec, scheduler)
                    print(
                        f"[转向阶梯] hit_rate={metrics.hit_rate:.3f} → "
                        f"rotate_penalty={scheduler.rotate_penalty:g} "
                        f"（phase={scheduler.rotate_phase}）",
                        flush=True,
                    )
                elif rot_result == "stop":
                    print(
                        f"[转向阶梯] hit_rate={metrics.hit_rate:.3f} "
                        f"已在末档 rotate={scheduler.rotate_penalty:g} 再次达标 → 停训",
                        flush=True,
                    )
                    self._stop_requested = True

                promoted = scheduler.maybe_promote(metrics)
                promotion_path: Path | None = None
                if promoted:
                    promotion_path = _save_promotion_checkpoint(
                        self.model,
                        run_dir=run_dir,
                        timesteps=t,
                        frame_stack=cfg.frame_stack,
                        frame_stride=cfg.frame_stride,
                        stack_action_mean=cfg.stack_action_mean,
                        completed_stage_index=stage_idx_before,
                        completed_stage_name=prev_name,
                        stage_timesteps=stage_steps_before,
                        bot_mode=bot_snap[0],
                        speed_scale=bot_snap[1],
                        mean_straight_frames=bot_snap[2],
                        turn_duration=bot_snap[3],
                        metrics=metrics,
                        next_stage_index=scheduler.stage_index,
                        next_stage_name=scheduler.stage.name,
                    )
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
                    promotion_path=promotion_path,
                )
                payload = {
                    "timesteps": t,
                    "stage": scheduler.stage.name,
                    "stage_index": scheduler.stage_index,
                    "kill_rate": metrics.kill_rate,
                    "hit_rate": metrics.hit_rate,
                    "median_ttk": metrics.median_ttk,
                    "promoted": promoted,
                    "rotate_penalty": scheduler.rotate_penalty,
                    "rotate_phase": scheduler.rotate_phase,
                    "rotate_stop": rot_result == "stop",
                }
                if progress_callback:
                    progress_callback(payload)

                if self._stop_requested:
                    # 末档达标：立刻存盘再结束
                    stop_path = run_dir / "checkpoints" / (
                        f"model_t{t}_stage{scheduler.stage_index}_"
                        f"{scheduler.stage.name}_rotate_done.zip"
                    )
                    stop_path.parent.mkdir(parents=True, exist_ok=True)
                    save_sb3_model(self.model, stop_path)
                    _save_ckpt_meta(
                        stop_path,
                        scheduler,
                        timesteps=t,
                        frame_stack=cfg.frame_stack,
                        frame_stride=cfg.frame_stride,
                        stack_action_mean=cfg.stack_action_mean,
                    )
                    _update_latest_symlink(run_dir, stop_path)
                    print(f"[停训存盘] {stop_path}", flush=True)
                    return

            # 存盘/评测之后再重置，避免下一行 FPS 把开销算进去
            self._t0 = time.perf_counter()
            self._steps0 = t

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CurriculumCallback(),
        reset_num_timesteps=resume_ckpt is None,
    )
    final_t = int(model.num_timesteps)
    ckpt = run_dir / "final_model.zip"
    save_sb3_model(model, ckpt)
    _save_ckpt_meta(
        ckpt,
        scheduler,
        timesteps=final_t,
        frame_stack=cfg.frame_stack,
        frame_stride=cfg.frame_stride,
        stack_action_mean=cfg.stack_action_mean,
    )
    _update_latest_symlink(run_dir, ckpt)
    _print_finish(scheduler, ckpt, run_dir)
    return model


def _clamp_policy_log_std(model: Any, lo: float, hi: float) -> None:
    """将 ActorCriticPolicy.log_std 钳到 [lo, hi]（就地改 Parameter）。"""
    if hi < lo:
        lo, hi = hi, lo
    policy = getattr(model, "policy", None)
    log_std = getattr(policy, "log_std", None)
    if log_std is None:
        return
    log_std.data.clamp_(float(lo), float(hi))


def _policy_action_std(model: Any) -> float:
    """当前策略高斯 std 的均值（exp(log_std)）。"""
    policy = getattr(model, "policy", None)
    log_std = getattr(policy, "log_std", None)
    if log_std is None:
        return float("nan")
    vals = log_std.detach().float().exp().reshape(-1)
    if vals.numel() == 0:
        return float("nan")
    return float(vals.mean().item())


def _scheduled_ent_coef(
    t: int, start: float, end: float, t0: int, decay_steps: int
) -> float:
    """自 t0 起经 decay_steps 线性 start→end；decay_steps≤0 则恒为 end。"""
    if decay_steps <= 0:
        return float(end)
    u = (int(t) - int(t0)) / float(decay_steps)
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return float(start + (end - start) * u)


def _update_latest_symlink(run_dir: Path, checkpoint: Path) -> None:
    """更新 run 目录下 ``latest.zip`` 指向最近存盘（便于 ``--resume <run_dir>``）。"""
    latest = Path(run_dir) / "latest.zip"
    target = Path(checkpoint)
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        # 相对路径，移动整个 run 目录仍可用
        rel = os.path.relpath(target, start=latest.parent)
        latest.symlink_to(rel)
    except OSError:
        # 部分文件系统不支持 symlink：忽略即可，仍可用 --resume 指向具体 zip
        pass


def _seed_named_model_zero(run_dir: Path, source_ckpt: Path) -> Path:
    """把续训起点复制到 ``named/model_0``（及中文别名 ``0号模型``）。"""
    import shutil

    named = Path(run_dir) / "named"
    named.mkdir(parents=True, exist_ok=True)
    dest = named / "model_0.zip"
    shutil.copy2(source_ckpt, dest)
    meta_src = Path(source_ckpt).with_name(Path(source_ckpt).stem + ".curriculum.json")
    meta_dst = named / "model_0.curriculum.json"
    if meta_src.is_file():
        shutil.copy2(meta_src, meta_dst)
    try:
        alias_zip = named / "0号模型.zip"
        alias_meta = named / "0号模型.curriculum.json"
        for link, target in ((alias_zip, "model_0.zip"), (alias_meta, "model_0.curriculum.json")):
            if link.is_symlink() or link.exists():
                link.unlink()
            if target == "model_0.curriculum.json" and not meta_dst.is_file():
                continue
            link.symlink_to(target)
    except OSError:
        pass
    readme = named / "README.txt"
    if not readme.exists():
        readme.write_text(
            "model_0 / 0号模型\n"
            f"  来源: {source_ckpt}\n"
            "  说明: 本 run 续训起点权重\n",
            encoding="utf-8",
        )
    return dest


def _save_ckpt_meta(
    path,
    scheduler,
    *,
    timesteps: int,
    frame_stack: int,
    frame_stride: int,
    stack_action_mean: bool,
) -> None:
    from tank_rl.checkpoint_meta import build_meta_from_scheduler, save_curriculum_meta

    meta = build_meta_from_scheduler(
        scheduler,
        timesteps=timesteps,
        frame_stack=frame_stack,
        frame_stride=frame_stride,
        stack_action_mean=stack_action_mean,
    )
    meta_path = save_curriculum_meta(path, meta)
    print(
        f"[存盘] 课程元数据 {meta_path.name}  "
        f"stage={meta.stage_name}  "
        f"bot={meta.bot_mode} v={meta.speed_scale:.2f}"
    )


def _save_promotion_checkpoint(
    model,
    *,
    run_dir: Path,
    timesteps: int,
    frame_stack: int,
    frame_stride: int,
    stack_action_mean: bool,
    completed_stage_index: int,
    completed_stage_name: str,
    stage_timesteps: int,
    bot_mode: str,
    speed_scale: float,
    mean_straight_frames: float,
    turn_duration: int,
    metrics: EvalMetrics,
    next_stage_index: int,
    next_stage_name: str,
) -> Path:
    """晋级瞬间单独存盘：权重为刚通过评测的模型，meta 为刚完成的阶段。"""
    from tank_rl.checkpoint_meta import (
        build_meta_snapshot,
        save_curriculum_meta,
        save_promotion_sidecar,
    )

    prom_dir = run_dir / "promotions"
    prom_dir.mkdir(parents=True, exist_ok=True)
    path = prom_dir / (
        f"stage{completed_stage_index}_{completed_stage_name}"
        f"_to_{next_stage_name}_t{timesteps}.zip"
    )
    save_sb3_model(model, path)
    meta = build_meta_snapshot(
        stage_index=completed_stage_index,
        stage_name=completed_stage_name,
        stage_timesteps=stage_timesteps,
        timesteps=timesteps,
        frame_stack=frame_stack,
        bot_mode=bot_mode,
        speed_scale=speed_scale,
        mean_straight_frames=mean_straight_frames,
        turn_duration=turn_duration,
        frame_stride=frame_stride,
        stack_action_mean=stack_action_mean,
    )
    meta_path = save_curriculum_meta(path, meta)
    eval_path = save_promotion_sidecar(
        path,
        metrics=metrics,
        completed_stage_index=completed_stage_index,
        completed_stage_name=completed_stage_name,
        next_stage_index=next_stage_index,
        next_stage_name=next_stage_name,
        eval_timesteps=timesteps,
    )
    print(
        f"[晋级存盘] {path.name}  "
        f"（完成 {completed_stage_name} → {next_stage_name}，"
        f"hit_rate={metrics.hit_rate:.3f}）"
    )
    print(f"[晋级存盘] 元数据 {meta_path.name}  评测 {eval_path.name}")
    return path


def _make_timestamped_run_dir(base: Path) -> Path:
    """在 ``base`` 下创建 ``YYYYMMDD_HHMMSS``（同秒冲突则追加 ``_2``…）。"""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base / stamp
    if run_dir.exists():
        n = 2
        while True:
            candidate = base / f"{stamp}_{n}"
            if not candidate.exists():
                run_dir = candidate
                break
            n += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _print_banner(
    config_path: Path,
    run_dir: Path,
    device: str,
    cfg: CurriculumAimConfig,
    scheduler: CurriculumScheduler,
    obs_dim: int,
    *,
    resume_ckpt: Path | None = None,
    start_timesteps: int = 0,
) -> None:
    stage = scheduler.stage
    print("=" * 60)
    if resume_ckpt is not None:
        print("瞄准课程 PPO 训练（断点续训）")
    else:
        print("瞄准课程 PPO 训练")
    print("=" * 60)
    print(f"  配置文件     : {config_path}")
    print(f"  本次运行目录 : {run_dir}")
    if resume_ckpt is not None:
        print(f"  续训权重     : {resume_ckpt}")
        print(f"  已训步数     : {start_timesteps:,} → 目标 {cfg.total_timesteps:,}")
    print(f"  设备         : {device}")
    print(f"  总步数       : {cfg.total_timesteps:,}")
    print(f"  并行环境     : {cfg.n_envs}")
    from tank_rl.vec_env.strided_stack import stack_depth, stack_lags

    _depth = stack_depth(cfg.frame_stack, cfg.frame_stride)
    _lags = stack_lags(cfg.frame_stack, cfg.frame_stride)
    _in_dim = stacked_obs_dim(
        obs_dim, cfg.frame_stack, action_dim=3 if cfg.stack_action_mean else 0
    )
    _mean_note = " + 确定性均值×3" if cfg.stack_action_mean else ""
    print(
        f"  帧堆叠       : 选择性 {cfg.frame_stack} 步距={cfg.frame_stride} "
        f"depth={_depth} lags={_lags}（开火/子弹仅当前）{_mean_note} → 输入维 {_in_dim}"
    )
    print(f"  评测间隔     : 每 {cfg.eval_every_timesteps:,} 步 / {cfg.eval_n_episodes} 局"
          f"（单局≤{cfg.eval_max_episode_steps}步）")
    print(f"  存盘间隔     : 每 {cfg.checkpoint_every_timesteps:,} 步 → {run_dir}/checkpoints/")
    print(f"  晋级存盘     : 每次晋级 → {run_dir}/promotions/")
    print("-" * 60)
    print(f"  当前阶段     : {stage.name} ({scheduler.stage_index + 1}/{len(cfg.stages)})")
    print(f"  对手         : mode={scheduler.bot.mode}  speed={scheduler.bot.speed_scale:.2f}")
    print(
        f"  奖励         : aim={stage.reward.get('aim_align_scale')} "
        f"mode={stage.reward.get('aim_mode')}  kill={stage.reward.get('kill')}  "
        f"rotate={scheduler.rotate_penalty:g}"
    )
    if cfg.rotate_penalty_schedule is not None:
        sch = cfg.rotate_penalty_schedule
        op = ">" if sch.strict_gt else ">="
        print(
            f"  转向阶梯     : phase={scheduler.rotate_phase}/{len(sch.levels)-1}  "
            f"levels={sch.levels}  推进条件 hit_rate{op}{sch.threshold:g}  "
            f"末档再达标={'停训' if sch.stop_after_last else '否'}"
        )
    print("-" * 60)
    print(f"  TensorBoard  : tensorboard --logdir {run_dir.parent}")
    print(f"  浏览器打开   : http://localhost:6006")
    print("=" * 60)


def _print_eval_block(
    prev_name: str,
    scheduler: CurriculumScheduler,
    metrics: EvalMetrics,
    promoted: bool,
    *,
    promote_stage_index: int,
    promotion_path: Path | None = None,
) -> None:
    pr = scheduler.cfg.stages[promote_stage_index].promote
    value = metrics.kill_rate if pr.metric == "kill_rate" else metrics.hit_rate
    metric_ok = value >= pr.threshold
    ttk_s = f"{metrics.median_ttk:.0f}" if metrics.median_ttk is not None else "无"
    metric_zh = {
        "kill_rate": "命中/开火 kill_rate",
        "hit_rate": "直击/开火 hit_rate",
    }.get(pr.metric, pr.metric)
    print("-" * 60)
    print(f"[评测] 阶段={prev_name}  局数={metrics.n_episodes}")
    print(
        f"  kill_rate(命中/开火) = {metrics.kill_rate:.3f}  "
        f"hit_rate(直击/开火) = {metrics.hit_rate:.3f}  "
        f"中位TTK = {ttk_s}"
    )
    print(
        f"  被命中率(挨打/敌开火) = {metrics.hit_taken_rate:.3f}  "
        f"先发制人率(直击前未被打中的局/局数) = {metrics.preemptive_rate:.3f}"
    )
    print("  TTK口径     : 首直击步数；无直击=该局结束步数（评测步限封顶）")
    print(
        f"  晋级条件 {metric_zh}≥{pr.threshold:.2f}  "
        f"当前={value:.3f}  → {'指标达标' if metric_ok else '指标未达标'}"
    )
    # 额外门槛说明（速度/直行退火）
    gate_notes: list[str] = []
    stage_cfg = scheduler.cfg.stages[promote_stage_index]
    if pr.require_speed_scale is not None:
        gate_notes.append(
            f"需对手速度≥{pr.require_speed_scale:.2f}"
            f"（当前={scheduler.bot.speed_scale:.2f}）"
        )
    if pr.require_mean_straight_frames is not None:
        gate_notes.append(
            f"需直行间隔≤{pr.require_mean_straight_frames:.0f}"
            f"（当前={scheduler.bot.mean_straight_frames:.0f}）"
        )
    if pr.max_median_ttk is not None:
        gate_notes.append(f"需中位TTK≤{pr.max_median_ttk:.0f}")
    if gate_notes:
        print(f"  附加门槛 : {'；'.join(gate_notes)}")
    # 转向阶梯开启时禁止阶段晋级（与 maybe_promote 一致）
    rotate_blocks = scheduler.cfg.rotate_penalty_schedule is not None
    if rotate_blocks:
        print(
            f"  转向阶梯中 : phase={scheduler.rotate_phase}  "
            f"rotate={scheduler.rotate_penalty:g}  "
            f"（此期间不晋级下一阶段，只按 hit_rate 下调转向）"
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
        if promotion_path is not None:
            print(f"    晋级模型 : {promotion_path}")
    else:
        if rotate_blocks:
            print("  未晋级阶段：转向阶梯进行中（属预期）")
        elif metric_ok:
            # 仅当附加门槛确实未过时才报「未满足」
            gates_failed = False
            if pr.max_median_ttk is not None:
                if metrics.median_ttk is None or metrics.median_ttk > pr.max_median_ttk:
                    gates_failed = True
            if pr.require_speed_scale is not None:
                if scheduler.bot.speed_scale + 1e-6 < pr.require_speed_scale:
                    gates_failed = True
            if pr.require_mean_straight_frames is not None:
                if scheduler.bot.mean_straight_frames > pr.require_mean_straight_frames + 1e-6:
                    gates_failed = True
            if gates_failed:
                print("  未晋级：指标已达标，但附加门槛未满足")
            else:
                print("  未晋级，继续当前阶段")
        else:
            print("  未晋级，继续当前阶段")
    print("-" * 60)


def _print_finish(scheduler: CurriculumScheduler, ckpt: Path, run_dir: Path) -> None:
    print("=" * 60)
    print("训练结束")
    print(f"  最终阶段 : {scheduler.stage.name} (#{scheduler.stage_index})")
    print(f"  模型保存 : {ckpt}")
    print(f"  本次目录 : {run_dir}")
    print(f"  TensorBoard: tensorboard --logdir {run_dir.parent}")
    print("=" * 60)


def _log_curriculum_tb(logger, scheduler: CurriculumScheduler, aim: float, promoted: bool | None) -> None:
    logger.record("curriculum/stage_index", float(scheduler.stage_index))
    logger.record("curriculum/speed_scale", float(scheduler.bot.speed_scale))
    logger.record(
        "curriculum/mean_straight_frames",
        float(scheduler.bot.mean_straight_frames),
    )
    logger.record("curriculum/aim_align_scale", float(aim))
    logger.record("curriculum/rotate_penalty", float(scheduler.rotate_penalty))
    logger.record("curriculum/rotate_phase", float(scheduler.rotate_phase))
    if promoted is not None:
        logger.record("curriculum/promoted", 1.0 if promoted else 0.0)


def _log_reward_parts_tb(logger, sums: dict[str, float], count: int) -> None:
    """记录本 rollout 各奖励分项的逐步均值。"""
    n = max(1, int(count))
    total = 0.0
    for key in sorted(sums.keys()):
        mean = float(sums[key]) / n
        logger.record(f"reward/{key}", mean)
        total += mean
    logger.record("reward/total_mean", total)
    logger.record("reward/steps_logged", float(count))


def _log_eval_tb(logger, metrics: EvalMetrics, *, promoted: bool) -> None:
    logger.record("eval/kill_rate", float(metrics.kill_rate))
    logger.record("eval/hit_rate", float(metrics.hit_rate))
    logger.record("eval/hit_taken_rate", float(metrics.hit_taken_rate))
    logger.record("eval/preemptive_rate", float(metrics.preemptive_rate))
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
        sch.rotate_phase = master.rotate_phase
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
    eval_sched.rotate_phase = scheduler.rotate_phase
    eval_sched._apply_stage_bot(eval_sched.stage, eval_sched._anneal_progress(eval_sched.stage))
    eval_sched.bot.configure(
        mode=scheduler.bot.mode,
        speed_scale=scheduler.bot.speed_scale,
        mean_straight_frames=scheduler.bot.mean_straight_frames,
        turn_duration=scheduler.bot.turn_duration,
    )
    stacked = make_strided_duel_eval_env(cfg, eval_sched)
    predict_fn = make_eval_predict_fn(model, stacked)
    return evaluate_duel_policy(
        stacked,
        predict_fn,
        n_episodes=cfg.eval_n_episodes,
    )


