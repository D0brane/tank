"""tank_rl.train"""

from tank_rl.train.curriculum_env import make_curriculum_env

__all__ = ["make_curriculum_env", "train_curriculum_aim"]


def __getattr__(name: str):
    if name == "train_curriculum_aim":
        from tank_rl.train.ppo_curriculum import train_curriculum_aim

        return train_curriculum_aim
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
