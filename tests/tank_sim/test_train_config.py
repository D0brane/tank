"""训练配置与堆叠帧数约定。"""

from pathlib import Path

import yaml


def test_ppo_mlp_frame_stack():
    path = Path("configs/train/ppo_mlp_sb3.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["env"]["frame_stack"] == 16
    # 与 classic obs v3（58 维）组合后的 MLP 输入维
    assert 58 * raw["env"]["frame_stack"] == 928


def test_curriculum_aim_open_obs_stack():
    path = Path("configs/train/curriculum_aim.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["env"]["frame_stack"] == 3
    assert raw["env"]["frame_stride"] == 3
    assert raw["env_config"].endswith("sim_p0_tt2_aim_open.yaml")
    assert raw["arena"]["mode"] == "random_open"
    from tank_sim.observation.spec import selective_stacked_dim

    # 选择性：17×3 + 41 = 92；含动作均值 → 101
    assert selective_stacked_dim(3) == 92
    assert raw["env"].get("stack_action_mean", False) is True
    assert selective_stacked_dim(3) + 3 * 3 == 101
