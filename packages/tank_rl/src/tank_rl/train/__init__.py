"""tank_rl.train"""

from tank_rl.train.ppo_curriculum import make_curriculum_env, train_curriculum_aim

__all__ = ["make_curriculum_env", "train_curriculum_aim"]
