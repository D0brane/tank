"""人玩：人机对战或本地双人对战，用于检查仿真手感 / 迷宫生成。"""

from __future__ import annotations

import argparse
import sys
import time


def _print_help(mode: str, layout: str, side: str, opponent: str, gen_maze: bool) -> None:
    print("--- 手感调试 ---")
    print("R 重开本图 | Esc 退出 | H 开关 HUD | 结束后需按 R 再开")
    if gen_maze:
        print("G 重新随机生成迷宫（新 seed）")
    if mode == "pvp":
        if layout == "tt2":
            print("红(玩家1): E/S/D/F 移动, Q 开火")
            print("蓝(玩家2): 方向键移动, M 开火")
        else:
            print("红: WASD + Space 开火")
            print("蓝: 方向键移动, M 开火（Space 仅红方）")
    else:
        who = "红" if side == "red" else "蓝"
        if layout == "tt2" and side == "red":
            print(f"你控制{who}: E/S/D/F, Q 开火")
        elif layout == "tt2" and side == "blue":
            print(f"你控制{who}: 方向键, M 开火")
        else:
            print(f"你控制{who}: WASD, Space 开火")
        if opponent == "none":
            print("对手: 静止靶")
        else:
            print("对手: 保守规则 Bot")


def _maze_options(args, seed: int | None = None) -> dict:
    return {
        "regen_maze": True,
        "maze_seed": args.seed if seed is None else seed,
        "maze_cols": args.maze_cols,
        "maze_rows": args.maze_rows,
        "maze_openness": args.maze_openness,
    }


def _build_env(args, duel: bool):
    from tank_sim.config import load_env_config
    from tank_sim.core.maze_gen import generate_maze, maze_to_ascii
    from tank_sim.envs.battle_env import BattleEnv
    from tank_sim.envs.duel_env import DuelEnv

    game_map = None
    map_path = args.map
    if args.gen_maze:
        cfg = load_env_config(args.config)
        game_map = generate_maze(
            cols=args.maze_cols,
            rows=args.maze_rows,
            cell_px=cfg.map.cell_px,
            wall_thickness=cfg.map.wall_thickness,
            seed=args.seed,
            openness=args.maze_openness,
        )
        map_path = None
        print(f"--- 生成迷宫 seed={args.seed} {args.maze_cols}x{args.maze_rows} ---")
        print(maze_to_ascii(game_map))
        print("---")

    if duel:
        return DuelEnv(
            config_path=args.config,
            map_path=map_path,
            game_map=game_map,
            agent_side=args.side,
            opponent=args.opponent,
            render_mode="human",
            render_style=args.render_style,
        )
    return BattleEnv(
        config_path=args.config,
        map_path=map_path,
        game_map=game_map,
        render_mode="human",
        render_style=args.render_style,
    )


def _reset_env(env, args, *, new_maze: bool = False) -> None:
    if args.gen_maze and new_maze:
        seed = int(time.time() * 1000) % 1_000_000
        args.seed = seed
        obs, _ = env.reset(options=_maze_options(args, seed=seed))
        from tank_sim.core.maze_gen import maze_to_ascii

        print(f"--- 新迷宫 seed={seed} ---")
        print(maze_to_ascii(env.state.game_map))
        return obs
    if args.gen_maze:
        return env.reset(options={"game_map": env.state.game_map})[0]
    return env.reset()[0]


def _run_duel(args) -> None:
    import pygame

    from tank_sim.control.human_keyboard import keys_to_action, scheme_for_side

    env = _build_env(args, duel=True)
    scheme = scheme_for_side(args.side, args.layout)
    hud_lines = [
        f"人机 opponent={args.opponent}"
        + (f" maze_seed={args.seed}" if args.gen_maze else ""),
        "R 重开 | G 新迷宫" if args.gen_maze else "R 重开",
    ]
    _print_help("duel", args.layout, args.side, args.opponent, args.gen_maze)

    obs = _reset_env(env, args, new_maze=False)
    clock = pygame.time.Clock()
    running = True
    show_hud = args.hud
    paused_after_end = False
    env.set_play_hud(show_hud, hud_lines)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key in (pygame.K_r, pygame.K_RETURN):
                    obs = _reset_env(env, args, new_maze=False)
                    paused_after_end = False
                    env.set_play_hud(show_hud, hud_lines)
                if event.key == pygame.K_g and args.gen_maze:
                    obs = _reset_env(env, args, new_maze=True)
                    paused_after_end = False
                    hud_lines[0] = f"人机 opponent={args.opponent} maze_seed={args.seed}"
                    env.set_play_hud(show_hud, hud_lines)
                if event.key == pygame.K_h:
                    show_hud = not show_hud
                    if not paused_after_end:
                        env.set_play_hud(show_hud, hud_lines)

        if not running:
            break

        if not paused_after_end:
            keys = pygame.key.get_pressed()
            action = keys_to_action(keys, scheme)
            obs, _reward, term, trunc, info = env.step(action)
            if term or trunc:
                print(
                    f"对局结束 winner={info.get('winner')} "
                    f"step={info.get('step')} —— 按 R 重开"
                    + (" / G 换图" if args.gen_maze else "")
                )
                paused_after_end = True

        if paused_after_end:
            tip = "按 R 重开" + (" / G 新迷宫" if args.gen_maze else "") + " / Esc 退出"
            env.set_play_hud(True, [f"结束 winner 已出", tip])
        else:
            env.set_play_hud(show_hud, hud_lines)
        env.render()
        clock.tick(args.fps)

    env.close()


def _run_pvp(args) -> None:
    import pygame

    from tank_sim.control.human_keyboard import keys_to_action

    env = _build_env(args, duel=False)
    red_scheme = "tt2_p1" if args.layout == "tt2" else "wasd"
    blue_scheme = "tt2_p2"
    hud_lines = [
        "双人对战" + (f" maze_seed={args.seed}" if args.gen_maze else ""),
        "R 重开 | G 新迷宫" if args.gen_maze else "R 重开",
    ]
    _print_help("pvp", args.layout, "red", "human", args.gen_maze)

    obs = _reset_env(env, args, new_maze=False)
    clock = pygame.time.Clock()
    running = True
    show_hud = args.hud
    paused_after_end = False
    env.set_play_hud(show_hud, hud_lines)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key in (pygame.K_r, pygame.K_RETURN):
                    obs = _reset_env(env, args, new_maze=False)
                    paused_after_end = False
                    env.set_play_hud(show_hud, hud_lines)
                if event.key == pygame.K_g and args.gen_maze:
                    obs = _reset_env(env, args, new_maze=True)
                    paused_after_end = False
                    hud_lines[0] = f"双人对战 maze_seed={args.seed}"
                    env.set_play_hud(show_hud, hud_lines)
                if event.key == pygame.K_h:
                    show_hud = not show_hud
                    if not paused_after_end:
                        env.set_play_hud(show_hud, hud_lines)

        if not running:
            break

        if not paused_after_end:
            keys = pygame.key.get_pressed()
            a_red = keys_to_action(keys, red_scheme)
            a_blue = keys_to_action(keys, blue_scheme)
            obs, _rewards, term, trunc, info = env.step({"red": a_red, "blue": a_blue})
            if term or trunc:
                print(
                    f"对局结束 winner={info.get('winner')} "
                    f"step={info.get('step')} —— 按 R 重开"
                )
                paused_after_end = True

        if paused_after_end:
            env.set_play_hud(True, ["双人对战已结束", "按 R 重开 / Esc 退出"])
        else:
            env.set_play_hud(show_hud, hud_lines)
        env.render()
        clock.tick(args.fps)

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tank 人玩 / 手感调试")
    parser.add_argument("--map", type=str, default="assets/maps/maze_small.txt")
    parser.add_argument("--config", type=str, default="configs/env/sim_p0_tt2_classic.yaml")
    parser.add_argument("--mode", choices=("duel", "pvp"), default="duel")
    parser.add_argument("--side", choices=("red", "blue"), default="red")
    parser.add_argument("--opponent", choices=("none", "rule"), default="none")
    parser.add_argument("--layout", choices=("tt2", "wasd"), default="wasd")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--hud", action="store_true")
    parser.add_argument(
        "--gen-maze",
        action="store_true",
        help="程序化生成迷宫（忽略 --map）",
    )
    parser.add_argument("--seed", type=int, default=42, help="迷宫随机种子")
    parser.add_argument("--maze-cols", type=int, default=11)
    parser.add_argument("--maze-rows", type=int, default=7)
    parser.add_argument(
        "--maze-openness",
        type=float,
        default=0.12,
        help="额外打通内墙比例，越大越空旷",
    )
    parser.add_argument(
        "--render-style",
        choices=("showcase", "lite"),
        default="showcase",
        help="showcase=正版贴图展示；lite=轻量纯色（训练用）",
    )
    args = parser.parse_args()

    try:
        import pygame
    except ImportError:
        print("请安装渲染依赖: pip install -e '.[render]'")
        sys.exit(1)

    pygame.init()
    if args.mode == "pvp":
        _run_pvp(args)
    else:
        _run_duel(args)
    pygame.quit()


if __name__ == "__main__":
    main()
