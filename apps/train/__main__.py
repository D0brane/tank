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
        help=(
            "实验根目录；新训或 --resume --new-run 时在其下新建 YYYYMMDD_HHMMSS/；"
            "默认续训写回原 run"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="计算设备：auto / cpu / cuda / cuda:0",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "断点续训：checkpoint .zip，或某次 run 目录"
            "（优先 latest.zip，否则 checkpoints 中步数最大的权重）"
        ),
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help=(
            "与 --resume 联用：新建时间戳 run，并将起点权重存为 named/model_0（0号模型）"
        ),
    )
    args = parser.parse_args(argv)

    from tank_rl.train.ppo_curriculum import train_curriculum_aim

    train_curriculum_aim(
        config_path=Path(args.config),
        log_dir=Path(args.log_dir),
        device=args.device,
        resume=Path(args.resume) if args.resume else None,
        new_run=bool(args.new_run),
    )


if __name__ == "__main__":
    main()
