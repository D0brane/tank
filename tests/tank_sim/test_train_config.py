"""训练配置与堆叠帧数约定。"""

from pathlib import Path

import yaml


def test_ppo_mlp_frame_stack():
    path = Path("configs/train/ppo_mlp_sb3.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["env"]["frame_stack"] == 16
    # 与 obs v3 组合后的 MLP 输入维
    assert 99 * raw["env"]["frame_stack"] == 1584
