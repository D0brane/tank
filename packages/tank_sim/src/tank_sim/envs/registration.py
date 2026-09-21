"""Gymnasium 注册。"""

from __future__ import annotations

import gymnasium as gym


def register_envs() -> None:
    """注册 TankDuel-v0 / TankBattle-v0。"""
    gym.register(
        id="TankDuel-v0",
        entry_point="tank_sim.envs.duel_env:DuelEnv",
    )
    gym.register(
        id="TankBattle-v0",
        entry_point="tank_sim.envs.battle_env:BattleEnv",
    )
