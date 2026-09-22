"""检查训练结果：加载权重实时对战课程靶，一局结束自动重开。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _resolve_stage(cfg, stage: str):
    """stage 可为名称或 0-based 下标。"""
    stages = cfg.stages
    if stage.isdigit():
        idx = int(stage)
        if idx < 0 or idx >= len(stages):
            raise SystemExit(f"阶段下标越界: {idx}（共 {len(stages)} 阶段）")
        return idx, stages[idx]
    for i, s in enumerate(stages):
        if s.name == stage:
            return i, s
    names = ", ".join(s.name for s in stages)
    raise SystemExit(f"未知阶段 {stage!r}，可选: auto / {names} 或 0..{len(stages)-1}")


def _configure_bot_for_watch(bot, stage_bot, *, use_anneal_end: bool) -> None:
    """观战默认用阶段退火终点难度（更难）；可用 --easy 用起点。"""
    speed = stage_bot.speed_scale
    if use_anneal_end and stage_bot.speed_scale_end is not None:
        speed = stage_bot.speed_scale_end
    straight = stage_bot.mean_straight_frames
    if use_anneal_end and stage_bot.mean_straight_frames_end is not None:
        straight = stage_bot.mean_straight_frames_end
    bot.configure(
        mode=stage_bot.mode,
        speed_scale=speed,
        mean_straight_frames=straight,
        turn_duration=stage_bot.turn_duration,
    )


def _sync_watch_scheduler(scheduler, stage_idx: int, bot) -> None:
    """观战对手与训练评测一致：写入 CurriculumScheduler 再建环境。"""
    scheduler.stage_index = stage_idx
    scheduler.bot.configure(
        mode=bot.mode,
        speed_scale=bot.speed_scale,
        mean_straight_frames=bot.mean_straight_frames,
        turn_duration=bot.turn_duration,
    )


def _episode_bullet_summary(info: dict) -> str:
    bf = int(info.get("bullets_fired", 0))
    ho = int(info.get("hits_on_enemy", 0))
    dh = int(info.get("direct_hits_on_enemy", 0))
    if bf <= 0:
        return "kill=0.000 hit=0.000 (0弹)"
    return (
        f"kill={ho / bf:.3f} hit={dh / bf:.3f} "
        f"(命中{ho}/开火{bf} 直击{dh})"
    )


def _configure_bot_from_meta(bot, meta) -> None:
    """用存盘时记录的退火后对手参数。"""
    bot.configure(
        mode=meta.bot_mode,  # type: ignore[arg-type]
        speed_scale=meta.speed_scale,
        mean_straight_frames=meta.mean_straight_frames,
        turn_duration=meta.turn_duration,
    )


def _checkpoint_timesteps(path: Path) -> int:
    """从 ``model_t114688_....zip`` 解析环境步数；解析失败返回 -1。"""
    import re

    m = re.search(r"_t(\d+)_", path.name)
    return int(m.group(1)) if m else -1


def _zip_obs_flat_and_steps(path: Path) -> tuple[int | None, int]:
    """轻量读取 zip：观测扁平维数 + 训练步数（失败则 (None, -1)）。"""
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
    except ImportError:
        return None, -1
    try:
        data, _params, _pytorch = load_from_zip_file(str(path), device="cpu")
    except Exception:
        return None, -1
    if not isinstance(data, dict):
        return None, -1
    space = data.get("observation_space")
    flat: int | None = None
    if space is not None and hasattr(space, "shape"):
        import numpy as np

        flat = int(np.prod(space.shape))
    steps = int(data.get("num_timesteps") or data.get("_total_timesteps") or -1)
    named = _checkpoint_timesteps(path)
    if named >= 0:
        steps = max(steps, named)
    return flat, steps


def _is_run_dir(path: Path) -> bool:
    """训练 run 目录：``YYYYMMDD_HHMMSS``（忽略 PPO_* / checkpoints 等杂项目录）。"""
    import re

    return bool(re.fullmatch(r"\d{8}_\d{6}", path.name))


def _auto_select_model(
    root: Path,
    *,
    expect_flat: int,
) -> Path | None:
    """
    自动选权重：优先**最新时间戳 run**，在该 run 内取维数匹配且步数最高的 zip。

    若最新 run 尚无兼容权重（例如旧 99 维），再往更早的 run 回退。
    """
    if not root.is_dir():
        return None
    runs = sorted(
        (p for p in root.iterdir() if p.is_dir() and _is_run_dir(p)),
        key=lambda p: p.name,
        reverse=True,
    )
    for run in runs:
        candidates: list[Path] = []
        final = run / "final_model.zip"
        if final.is_file():
            candidates.append(final)
        ckpt_dir = run / "checkpoints"
        if ckpt_dir.is_dir():
            candidates.extend(ckpt_dir.glob("model_t*.zip"))
        scored: list[tuple[int, Path]] = []
        for path in candidates:
            flat, steps = _zip_obs_flat_and_steps(path)
            if flat is None or flat != expect_flat:
                continue
            scored.append((steps, path))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]
    return None


# 兼容旧测试名
def _latest_model(root: Path = Path("runs/curriculum_aim")) -> Path | None:
    """无维数约束时：最新时间戳 run 内最大步数 checkpoint。"""
    if not root.is_dir():
        return None
    runs = sorted(
        (p for p in root.iterdir() if p.is_dir() and _is_run_dir(p)),
        key=lambda p: p.name,
        reverse=True,
    )
    for run in runs:
        candidates: list[Path] = []
        final = run / "final_model.zip"
        if final.is_file():
            candidates.append(final)
        ckpt_dir = run / "checkpoints"
        if ckpt_dir.is_dir():
            candidates.extend(ckpt_dir.glob("model_t*.zip"))
        if candidates:
            return max(
                candidates,
                key=lambda p: (
                    _checkpoint_timesteps(p)
                    if _checkpoint_timesteps(p) >= 0
                    else _zip_obs_flat_and_steps(p)[1]
                ),
            )
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="加载训练权重，对战瞄准课程靶，实时展示（结束自动重开）"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="SB3 PPO checkpoint（.zip）；默认自动选与当前观测维匹配、步数最高的权重",
    )
    parser.add_argument(
        "--curriculum",
        default="configs/train/curriculum_aim.yaml",
        help="课程 YAML（出生距离、frame_stack、阶段 Bot）",
    )
    parser.add_argument(
        "--stage",
        default="auto",
        help="auto=读 checkpoint 旁 .curriculum.json；或阶段名/下标 0/1/2",
    )
    parser.add_argument(
        "--easy",
        action="store_true",
        help="手动选阶段时用退火起点（默认终点）；auto+meta 时忽略",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="策略是否确定性动作（默认开）",
    )
    parser.add_argument(
        "--render-style",
        choices=("showcase", "lite"),
        default="showcase",
    )
    parser.add_argument(
        "--no-auto-restart",
        action="store_true",
        help="结束不自动重开（需按 R）",
    )
    args = parser.parse_args(argv)

    try:
        import pygame
    except ImportError:
        print("请安装: pip install -e '.[render]'")
        sys.exit(1)

    from stable_baselines3 import PPO

    from tank_rl.checkpoint_meta import (
        load_curriculum_meta,
        parse_stage_index_from_filename,
    )
    from tank_rl.curriculum.config import load_curriculum_aim_config
    from tank_rl.curriculum.scheduler import CurriculumScheduler
    from tank_rl.eval.duel_eval_env import make_eval_predict_fn, make_strided_duel_eval_env
    from tank_rl.vec_env.strided_stack import stacked_obs_dim
    from tank_sim.bots.curriculum_bot import CurriculumBot
    from tank_sim.config import load_env_config

    cur = load_curriculum_aim_config(args.curriculum)

    if args.model:
        model_path = Path(args.model)
    else:
        env_cfg = load_env_config(cur.env_config)
        expect_flat = stacked_obs_dim(
            int(env_cfg.obs.dim),
            int(cur.frame_stack),
            action_dim=3 if cur.stack_action_mean else 0,
        )
        found = _auto_select_model(
            Path("runs/curriculum_aim"), expect_flat=expect_flat
        )
        if found is None:
            raise SystemExit(
                "未找到与当前观测维匹配的模型：请传 --model，或先训练生成 "
                f"obs={env_cfg.obs.dim}×stack={cur.frame_stack}（输入 {expect_flat}）的 "
                "runs/curriculum_aim/<时间戳>/checkpoints/*.zip"
            )
        model_path = found
        steps = _checkpoint_timesteps(model_path)
        print(
            f"[加载] 自动选择 {model_path}"
            + (f"  (t={steps:,})" if steps >= 0 else "")
            + f"  匹配输入维 {expect_flat}"
        )

    meta = load_curriculum_meta(model_path)
    bot_from_meta = False

    if args.stage == "auto":
        if meta is not None:
            stage_idx = meta.stage_index
            if stage_idx < 0 or stage_idx >= len(cur.stages):
                raise SystemExit(
                    f"meta.stage_index={stage_idx} 越界（共 {len(cur.stages)} 阶段）"
                )
            stage = cur.stages[stage_idx]
            bot = CurriculumBot(mode=meta.bot_mode, seed=cur.seed)  # type: ignore[arg-type]
            _configure_bot_from_meta(bot, meta)
            bot_from_meta = True
            print(
                f"[加载] 课程元数据 → {meta.stage_name}  "
                f"mode={meta.bot_mode} v={meta.speed_scale:.2f} "
                f"直行={meta.mean_straight_frames:.0f}"
            )
        else:
            parsed = parse_stage_index_from_filename(model_path)
            stage_idx = parsed if parsed is not None else 0
            if stage_idx >= len(cur.stages):
                stage_idx = 0
            stage = cur.stages[stage_idx]
            bot = CurriculumBot(mode=stage.bot.mode, seed=cur.seed)
            _configure_bot_for_watch(bot, stage.bot, use_anneal_end=not args.easy)
            print(
                f"[加载] 无 .curriculum.json，按文件名/默认阶段 "
                f"→ {stage.name}（退火{'终点' if not args.easy else '起点'}）"
            )
    else:
        stage_idx, stage = _resolve_stage(cur, args.stage)
        bot = CurriculumBot(mode=stage.bot.mode, seed=cur.seed)
        _configure_bot_for_watch(bot, stage.bot, use_anneal_end=not args.easy)

    frame_stack = meta.frame_stack if meta is not None else cur.frame_stack
    frame_stride = (
        int(getattr(meta, "frame_stride", 1))
        if meta is not None
        else cur.frame_stride
    )
    stack_action_mean = (
        bool(getattr(meta, "stack_action_mean", False))
        if meta is not None
        else cur.stack_action_mean
    )

    scheduler = CurriculumScheduler(cur)
    _sync_watch_scheduler(scheduler, stage_idx, bot)
    stack_env = make_strided_duel_eval_env(
        cur,
        scheduler,
        render_mode="human",
        render_style=args.render_style,
        stack_action_mean=stack_action_mean,
    )
    env = stack_env.env

    device = args.device
    if device == "auto":
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    try:
        model = PPO.load(str(model_path), device=device)
    except Exception as e:
        stack_env.close()
        raise SystemExit(f"无法加载权重：{e}") from e
    flat_obs = stacked_obs_dim(
        int(env.cfg.obs.dim),
        int(frame_stack),
        action_dim=3 if stack_action_mean else 0,
    )
    if int(model.observation_space.shape[0]) != flat_obs:
        stack_env.close()
        raise SystemExit(
            f"观测维不匹配：模型 {model.observation_space.shape[0]} "
            f"≠ {flat_obs}（{env.cfg.obs.dim}×stack={frame_stack}）"
        )

    pygame.init()
    obs, _ = stack_env.reset(seed=cur.seed)
    clock = pygame.time.Clock()
    episode = 1
    auto_restart = not args.no_auto_restart

    ckpt_steps = _checkpoint_timesteps(model_path)
    print("=" * 56)
    print("检查结果 / 实时观战")
    print(f"  模型     : {model_path}")
    if ckpt_steps >= 0:
        print(f"  权重步数 : t={ckpt_steps:,}")
        if ckpt_steps < 200_000:
            print(
                "  警告     : 权重很早（<200k），行为会飘；"
                "等更新的 checkpoint 或传 --model 指定"
            )
    print(f"  课程阶段 : {stage.name} (#{stage_idx})")
    print(
        f"  对手     : mode={bot.mode} speed={bot.speed_scale:.2f} "
        f"直行间隔={bot.mean_straight_frames:.0f}"
        + ("  [存盘快照]" if bot_from_meta else "")
    )
    print(
        f"  环境     : obs={env.cfg.obs.dim}  "
        f"hits_to_die={env.cfg.sim.tank.hits_to_die}  "
        f"堆叠={frame_stack} stride={frame_stride}  "
        f"均值堆叠={'开' if stack_action_mean else '关'}  "
        f"确定性={args.deterministic}"
    )
    print("  Esc 退出 | R 重开 | 1/2/3 切换阶段0/1/2（手动难度）")
    print(f"  自动重开 : {'开' if auto_restart else '关'}")
    print("=" * 56)

    hud = [
        f"检查结果 ep={episode} stage={stage.name}",
        f"bot={bot.mode} v={bot.speed_scale:.2f}",
    ]
    env.set_play_hud(True, hud)

    predict = make_eval_predict_fn(model, stack_env)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = stack_env.reset()
                    episode += 1
                    hud[0] = f"检查结果 ep={episode} stage={stage.name}"
                    env.set_play_hud(True, hud)
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    new_idx = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2}[event.key]
                    if new_idx < len(cur.stages):
                        stage_idx = new_idx
                        stage = cur.stages[stage_idx]
                        bot_from_meta = False
                        _configure_bot_for_watch(
                            bot, stage.bot, use_anneal_end=not args.easy
                        )
                        _sync_watch_scheduler(scheduler, stage_idx, bot)
                        env.set_reward_overrides(stage.reward)
                        obs, _ = stack_env.reset()
                        episode += 1
                        hud = [
                            f"检查结果 ep={episode} stage={stage.name}",
                            f"bot={bot.mode} v={bot.speed_scale:.2f}",
                        ]
                        env.set_play_hud(True, hud)
                        print(
                            f"[切换] → {stage.name} mode={bot.mode} "
                            f"speed={bot.speed_scale:.2f}"
                        )

        action = predict(obs, deterministic=args.deterministic)
        obs, _reward, terminated, truncated, info = stack_env.step(action)
        env.render()

        if terminated or truncated:
            winner = info.get("winner", "?")
            print(
                f"[结束] ep={episode} winner={winner} steps={info.get('step')} "
                f"stage={stage.name} {_episode_bullet_summary(info)}"
            )
            if auto_restart:
                time.sleep(0.35)
                obs, _ = stack_env.reset()
                episode += 1
                hud[0] = f"检查结果 ep={episode} stage={stage.name}"
                env.set_play_hud(True, hud)
            else:
                # 停住画面等 R
                while True:
                    env.render()
                    paused = False
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            paused = True
                            break
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                                paused = True
                                break
                            if event.key == pygame.K_r:
                                obs, _ = stack_env.reset()
                                episode += 1
                                hud[0] = f"检查结果 ep={episode} stage={stage.name}"
                                env.set_play_hud(True, hud)
                                paused = True
                                break
                    if paused or not running:
                        break
                    clock.tick(args.fps)

        clock.tick(args.fps)

    stack_env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
