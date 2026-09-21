"""双模型/双 Bot 观战（占位：当前为 规则 vs 规则，后续接 SB3 checkpoint）。"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Tank 观战")
    parser.add_argument("--map", type=str, default="assets/maps/maze_small.txt")
    parser.add_argument("--config", type=str, default="configs/env/sim_p0_tt2_classic.yaml")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--render-style",
        choices=("showcase", "lite"),
        default="showcase",
        help="showcase=正版贴图；lite=轻量",
    )
    args = parser.parse_args()

    try:
        import pygame
    except ImportError:
        print("请安装: pip install -e '.[render]'")
        sys.exit(1)

    from tank_sim.bots.rule_bot_v1 import RuleBotV1
    from tank_sim.envs.battle_env import BattleEnv

    pygame.init()
    env = BattleEnv(
        config_path=args.config,
        map_path=args.map,
        render_mode="human",
        render_style=args.render_style,
    )
    bot = RuleBotV1()
    obs, _ = env.reset()
    clock = pygame.time.Clock()
    running = True

    print("观战：规则 Bot 红 vs 蓝。Esc 退出，R 重开。")

    while running:
        # TODO: 替换为 PolicyBundle 加载 checkpoint
        state = env.state
        a_red = bot.act(obs["red"], state, "red")
        a_blue = bot.act(obs["blue"], state, "blue")
        obs, rewards, term, trunc, info = env.step({"red": a_red, "blue": a_blue})
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, _ = env.reset()

        if term or trunc:
            print(f"结束 winner={info['winner']} 步数={info['step']}")
            obs, _ = env.reset()

        clock.tick(args.fps)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
