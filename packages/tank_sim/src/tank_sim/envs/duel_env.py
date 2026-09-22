"""单 agent 训练环境（对手可注入）。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal

import gymnasium as gym
import numpy as np

from tank_sim.bots.curriculum_bot import CurriculumBot
from tank_sim.config import EnvConfig, RewardConfig, default_config_path, load_env_config
from tank_sim.control.action_mapping import continuous_to_intent
from tank_sim.core.spawn import sample_dual_spawn
from tank_sim.core.world import create_initial_state, step_world
from tank_sim.observation.builder import ObservationBuilder
from tank_sim.reward.shaping import RewardState, compute_reward_breakdown


OpponentFn = Callable[[np.ndarray], np.ndarray]
OpponentSpec = Literal["none", "rule", "curriculum"] | OpponentFn | CurriculumBot


class DuelEnv(gym.Env):
    """
    训练侧环境：默认控制红方，蓝方由 rule / curriculum / callable / 零动作 控制。

    opponent:
      - "none": 蓝方不动
      - "rule": 使用规则 Bot
      - "curriculum": 使用 CurriculumBot（瞄准课程靶）
      - CurriculumBot 实例或带 act(obs,state,side) 的对象
      - callable: 接收 opponent_obs 返回 action
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config_path: str | None = None,
        map_path: str | None = None,
        game_map=None,
        agent_side: Literal["red", "blue"] = "red",
        opponent: OpponentSpec = "rule",
        curriculum_bot: CurriculumBot | None = None,
        render_mode: str | None = None,
        render_style: Literal["lite", "showcase"] | None = None,
        *,
        random_spawn: bool = False,
        min_spawn_dist: float = 120.0,
        max_spawn_dist: float = 360.0,
        reward_overrides: dict[str, Any] | None = None,
        open_arena: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg_path = config_path or str(default_config_path())
        self.cfg: EnvConfig = load_env_config(cfg_path)
        if reward_overrides:
            self.cfg = replace(
                self.cfg,
                reward=_merge_reward(self.cfg.reward, reward_overrides),
            )
        self._map_path = map_path
        self._game_map = game_map
        self._open_arena = dict(open_arena) if open_arena else None
        self.agent_side = agent_side
        self._opponent = opponent
        self._curriculum_bot = curriculum_bot
        if opponent == "curriculum" and self._curriculum_bot is None:
            self._curriculum_bot = CurriculumBot(mode="static")
        if isinstance(opponent, CurriculumBot):
            self._curriculum_bot = opponent
            self._opponent = "curriculum"
        self.render_mode = render_mode
        if render_style is None:
            self.render_style = "showcase" if render_mode == "human" else "lite"
        else:
            self.render_style = render_style
        self._rule_bot = None
        self.random_spawn = random_spawn
        self.min_spawn_dist = float(min_spawn_dist)
        self.max_spawn_dist = float(max_spawn_dist)
        self._np_random: np.random.Generator | None = None

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.obs.dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self._state = create_initial_state(
            self.cfg, self._map_path, game_map=self._game_map
        )
        self._obs_builder = ObservationBuilder(self.cfg)
        self._reward_state = RewardState()
        self._renderer = None
        self._hud_debug = False
        self._hud_lines: list[str] = []
        self._episode_fired = False
        self._episode_near_hit = False
        self._episode_hit_enemy = False
        self._episode_direct_hit_enemy = False

    @property
    def state(self):
        """当前仿真状态（调试 / 人玩 HUD）。"""
        return self._state

    @property
    def curriculum_bot(self) -> CurriculumBot | None:
        return self._curriculum_bot

    def set_reward_overrides(self, overrides: dict[str, Any]) -> None:
        """课程阶段切换时更新奖励权重。"""
        self.cfg = replace(
            self.cfg,
            reward=_merge_reward(self.cfg.reward, overrides),
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()

        opts = options or {}
        if "map_path" in opts:
            self._map_path = opts["map_path"]
            self._game_map = None
        if "game_map" in opts:
            self._game_map = opts["game_map"]
        if opts.get("regen_maze"):
            from tank_sim.core.maze_gen import generate_maze

            maze_seed = opts.get("maze_seed", seed)
            self._game_map = generate_maze(
                cols=int(opts.get("maze_cols", 11)),
                rows=int(opts.get("maze_rows", 7)),
                cell_px=self.cfg.map.cell_px,
                wall_thickness=self.cfg.map.wall_thickness,
                seed=maze_seed,
                openness=float(opts.get("maze_openness", 0.12)),
            )
            self._map_path = None
        elif self._open_arena and self._open_arena.get("mode") == "random_open":
            from tank_sim.core.map_loader import generate_open_arena

            assert self._np_random is not None
            cols_lo = int(self._open_arena.get("cols_min", 8))
            cols_hi = int(self._open_arena.get("cols_max", 12))
            rows_lo = int(self._open_arena.get("rows_min", 4))
            rows_hi = int(self._open_arena.get("rows_max", 6))
            if cols_lo > cols_hi:
                cols_lo, cols_hi = cols_hi, cols_lo
            if rows_lo > rows_hi:
                rows_lo, rows_hi = rows_hi, rows_lo
            cols = int(self._np_random.integers(cols_lo, cols_hi + 1))
            rows = int(self._np_random.integers(rows_lo, rows_hi + 1))
            self._game_map = generate_open_arena(
                cols=cols,
                rows=rows,
                cell_px=self.cfg.map.cell_px,
                wall_thickness=self.cfg.map.wall_thickness,
            )
            self._map_path = None

        random_spawn = bool(opts.get("random_spawn", self.random_spawn))
        min_dist = float(opts.get("min_spawn_dist", self.min_spawn_dist))
        max_dist = float(opts.get("max_spawn_dist", self.max_spawn_dist))

        spawn = None
        if random_spawn:
            # 先拿到地图再采样
            probe = create_initial_state(
                self.cfg, self._map_path, game_map=self._game_map
            )
            spawn = sample_dual_spawn(
                probe.game_map,
                self.cfg.sim.tank,
                self._np_random,
                min_dist=min_dist,
                max_dist=max_dist,
            )
            self._game_map = probe.game_map
            self._map_path = None

        self._state = create_initial_state(
            self.cfg, self._map_path, game_map=self._game_map, spawn=spawn
        )
        self._obs_builder.reset()
        self._reward_state = RewardState()
        self._episode_fired = False
        self._episode_near_hit = False
        self._episode_hit_enemy = False
        self._episode_direct_hit_enemy = False
        if self._curriculum_bot is not None:
            self._curriculum_bot.reset(seed=None if seed is None else int(seed) + 17)
        return self._agent_obs(), {"step": 0}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        prev = self._state
        agent_intent = continuous_to_intent(action)
        opp_action = self._opponent_action()
        opp_intent = continuous_to_intent(opp_action)

        if self.agent_side == "red":
            intent_red, intent_blue = agent_intent, opp_intent
        else:
            intent_red, intent_blue = opp_intent, agent_intent

        self._state = step_world(
            self._state, intent_red, intent_blue, self.cfg.sim
        )
        self._obs_builder.on_step_end(self._state)

        if self.agent_side == "red" and self._state.red_fired:
            self._episode_fired = True
        if self.agent_side == "blue" and self._state.blue_fired:
            self._episode_fired = True
        for ev in self._state.hit_events:
            if ev.attacker != self.agent_side or ev.victim == self.agent_side:
                continue
            self._episode_hit_enemy = True
            if ev.bounces == 0:
                self._episode_direct_hit_enemy = True
        self._episode_near_hit = self._episode_near_hit or _own_bullet_near_enemy(
            self._state, self.agent_side
        )

        breakdown = compute_reward_breakdown(
            prev,
            self._state,
            self.agent_side,
            self.cfg.reward,
            self._reward_state,
            bullet_speed=self.cfg.sim.bullet.speed,
            fire_intent=bool(agent_intent.fire),
            move_intent=agent_intent.move,
            rotate_intent=agent_intent.rotate,
        )
        reward = breakdown.total
        terminated = self._state.terminated
        truncated = self._state.step >= self.cfg.sim.max_episode_steps
        agent_won = self._state.winner == self.agent_side
        info = {
            "winner": self._state.winner,
            "step": self._state.step,
            "agent_won": agent_won,
            # 晋级 hit_rate：本局至少一发己弹未反弹命中敌方（不必致死）
            "direct_hit": self._episode_direct_hit_enemy,
            # 评测 kill_rate（总命中）：本局己弹至少打中敌方一次（含反弹）
            "hit_enemy": self._episode_hit_enemy,
            "near_hit": self._episode_near_hit,
            "kill_bullet_bounces": self._state.kill_bullet_bounces,
            "kill_bullet_owner": self._state.kill_bullet_owner,
            "fired": self._episode_fired,
            "ttk": self._state.step if agent_won else None,
            "reward_parts": breakdown.as_parts_dict(),
        }
        return self._agent_obs(), reward, terminated, truncated, info

    def _agent_obs(self) -> np.ndarray:
        return self._obs_builder.build(self._state, self.agent_side)

    def _opponent_action(self) -> np.ndarray:
        side = _other_side(self.agent_side)
        if self._opponent == "none":
            return np.zeros(3, dtype=np.float32)
        if self._opponent == "rule":
            if self._rule_bot is None:
                from tank_sim.bots.rule_bot_v1 import RuleBotV1

                self._rule_bot = RuleBotV1()
            obs = self._obs_builder.build(self._state, side)
            return self._rule_bot.act(obs, self._state, side)
        if self._opponent == "curriculum":
            assert self._curriculum_bot is not None
            obs = self._obs_builder.build(self._state, side)
            return self._curriculum_bot.act(obs, self._state, side)
        if hasattr(self._opponent, "act"):
            obs = self._obs_builder.build(self._state, side)
            return self._opponent.act(obs, self._state, side)  # type: ignore[union-attr]
        return self._opponent(self._obs_builder.build(self._state, side))  # type: ignore[operator]

    def set_play_hud(self, enabled: bool, lines: list[str] | None = None) -> None:
        """人玩时开关调试 HUD。"""
        self._hud_debug = enabled
        if lines is not None:
            self._hud_lines = lines
        if self._renderer is not None:
            self._renderer.show_debug_hud = enabled
            self._renderer.set_hud_extra(self._hud_lines)

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from tank_sim.rendering.factory import make_renderer

            self._renderer = make_renderer(self.cfg, style=self.render_style)
            self._renderer.show_debug_hud = self._hud_debug
            self._renderer.set_hud_extra(self._hud_lines)
        return self._renderer.render(self._state, mode=self.render_mode)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def _other_side(side: str) -> str:
    return "blue" if side == "red" else "red"


def _merge_reward(base: RewardConfig, overrides: dict[str, Any]) -> RewardConfig:
    data = {
        "kill": base.kill,
        "death": base.death,
        "survive_per_step": base.survive_per_step,
        "bullet_near_enemy": base.bullet_near_enemy,
        "bullet_threat_self": base.bullet_threat_self,
        "fire_penalty": base.fire_penalty,
        "path_delta_scale": base.path_delta_scale,
        "aim_align_scale": base.aim_align_scale,
        "aim_mode": base.aim_mode,
        "aim_align_power": base.aim_align_power,
        "fire_on_cd_penalty": base.fire_on_cd_penalty,
        "move_penalty": base.move_penalty,
        "rotate_penalty": base.rotate_penalty,
        "move_switch_penalty": base.move_switch_penalty,
        "kill_bounce": base.kill_bounce,
        "wall_proximity_scale": base.wall_proximity_scale,
        "wall_proximity_margin": base.wall_proximity_margin,
        "wall_proximity_power": base.wall_proximity_power,
        "enemy_proximity_scale": base.enemy_proximity_scale,
        "enemy_proximity_margin": base.enemy_proximity_margin,
        "enemy_proximity_power": base.enemy_proximity_power,
    }
    data.update({k: overrides[k] for k in data if k in overrides})
    return RewardConfig(**data)


def _own_bullet_near_enemy(state, side: str, radius: float = 24.0) -> bool:
    me = state.tanks[0] if side == "red" else state.tanks[1]
    enemy = state.tanks[1] if side == "red" else state.tanks[0]
    import math

    for b in state.bullets:
        if b.owner != me.owner:
            continue
        if math.hypot(b.x - enemy.x, b.y - enemy.y) <= radius:
            return True
    return False
