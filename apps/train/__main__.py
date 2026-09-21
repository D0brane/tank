"""训练入口：瞄准课程 PPO。"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Tank 瞄准课程 PPO 训练（中文状态输出 + TensorBoard）"
    )
    parser.add_argument(
        "--config",
        default="configs/train/curriculum_aim.yaml",
        help="课程 YAML 路径",
    )
    parser.add_argument(
        "--log-dir",
        default="runs/curriculum_aim",
        help="checkpoint 与 TensorBoard 日志目录",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="计算设备：auto / cpu / cuda / cuda:0",
    )
    args = parser.parse_args(argv)

    from tank_rl.train.ppo_curriculum import train_curriculum_aim

    train_curriculum_aim(
        config_path=Path(args.config),
        log_dir=Path(args.log_dir),
        device=args.device,
    )


if __name__ == "__main__":
    main()
