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
    raise SystemExit(f"未知阶段 {stage!r}，可选: {names} 或 0..{len(stages)-1}")


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="加载训练权重，对战瞄准课程靶，实时展示（结束自动重开）"
    )
    parser.add_argument(
        "--model",
        default="runs/curriculum_aim/final_model.zip",
        help="SB3 PPO checkpoint（.zip）",
    )
    parser.add_argument(
        "--curriculum",
        default="configs/train/curriculum_aim.yaml",
        help="课程 YAML（出生距离、frame_stack、阶段 Bot）",
    )
    parser.add_argument(
        "--stage",
        default="0",
        help="课程阶段名或下标，如 stage1_static / 0 / 1 / 2",
    )
    parser.add_argument(
        "--easy",
        action="store_true",
        help="用阶段退火起点难度（默认用终点：更快/更勤转弯）",
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

    from tank_rl.curriculum.config import load_curriculum_aim_config
    from tank_rl.inference.policy_bundle import PolicyBundle
    from tank_sim.bots.curriculum_bot import CurriculumBot
    from tank_sim.envs.duel_env import DuelEnv

    cur = load_curriculum_aim_config(args.curriculum)
    stage_idx, stage = _resolve_stage(cur, args.stage)
    bot = CurriculumBot(mode=stage.bot.mode, seed=cur.seed)
    _configure_bot_for_watch(bot, stage.bot, use_anneal_end=not args.easy)

    model_path = Path(args.model)
    policy = PolicyBundle(
        model_path,
        frame_stack=cur.frame_stack,
        device=args.device,
    )

    env = DuelEnv(
        config_path=cur.env_config,
        map_path=cur.map_path,
        agent_side=cur.agent_side,  # type: ignore[arg-type]
        opponent="curriculum",
        curriculum_bot=bot,
        render_mode="human",
        render_style=args.render_style,
        random_spawn=cur.random_spawn,
        min_spawn_dist=cur.min_spawn_dist,
        max_spawn_dist=cur.max_spawn_dist,
        reward_overrides=stage.reward,
    )

    pygame.init()
    obs, _ = env.reset(seed=cur.seed)
    policy.reset()
    clock = pygame.time.Clock()
    episode = 1
    auto_restart = not args.no_auto_restart

    print("=" * 56)
    print("检查结果 / 实时观战")
    print(f"  模型     : {model_path}")
    print(f"  课程阶段 : {stage.name} (#{stage_idx})")
    print(
        f"  对手     : mode={bot.mode} speed={bot.speed_scale:.2f} "
        f"直行间隔={bot.mean_straight_frames:.0f}"
    )
    print(f"  帧堆叠   : {cur.frame_stack}  确定性={args.deterministic}")
    print("  Esc 退出 | R 重开 | 1/2/3 切换阶段0/1/2")
    print(f"  自动重开 : {'开' if auto_restart else '关'}")
    print("=" * 56)

    hud = [
        f"检查结果 ep={episode} stage={stage.name}",
        f"bot={bot.mode} v={bot.speed_scale:.2f}",
    ]
    env.set_play_hud(True, hud)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    policy.reset()
                    episode += 1
                    hud[0] = f"检查结果 ep={episode} stage={stage.name}"
                    env.set_play_hud(True, hud)
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    new_idx = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2}[event.key]
                    if new_idx < len(cur.stages):
                        stage_idx = new_idx
                        stage = cur.stages[stage_idx]
                        _configure_bot_for_watch(
                            bot, stage.bot, use_anneal_end=not args.easy
                        )
                        env.set_reward_overrides(stage.reward)
                        obs, _ = env.reset()
                        policy.reset()
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

        action = policy.predict(obs, deterministic=args.deterministic)
        obs, _reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            winner = info.get("winner", "?")
            print(
                f"[结束] ep={episode} winner={winner} steps={info.get('step')} "
                f"stage={stage.name}"
            )
            if auto_restart:
                time.sleep(0.35)
                obs, _ = env.reset()
                policy.reset()
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
                                obs, _ = env.reset()
                                policy.reset()
                                episode += 1
                                hud[0] = f"检查结果 ep={episode} stage={stage.name}"
                                env.set_play_hud(True, hud)
                                paused = True
                                break
                    if paused or not running:
                        break
                    clock.tick(args.fps)

        clock.tick(args.fps)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
