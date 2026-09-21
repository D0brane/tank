"""坦克对战仿真包：物理、观测、Gymnasium 环境。"""

from tank_sim.envs.battle_env import BattleEnv
from tank_sim.envs.duel_env import DuelEnv
from tank_sim.envs.registration import register_envs

__all__ = ["BattleEnv", "DuelEnv", "register_envs"]
