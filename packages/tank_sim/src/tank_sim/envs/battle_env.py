"""双外部控制的观战/评测环境。"""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np

from tank_sim.config import EnvConfig, default_config_path, load_env_config
from tank_sim.control.action_mapping import continuous_to_intent
from tank_sim.core.world import create_initial_state, step_world
from tank_sim.observation.builder import ObservationBuilder
from tank_sim.reward.shaping import RewardState, compute_reward_for_side


class BattleEnv(gym.Env):
    """
    红蓝双方均由外部传入动作的双人环境。

    观测：各 obs.dim 维（v3 为 99）；动作：Box(-1,1,(3,))。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config_path: str | None = None,
        map_path: str | None = None,
        game_map=None,
        render_mode: str | None = None,
        render_style: Literal["lite", "showcase"] | None = None,
    ) -> None:
        super().__init__()
        cfg_path = config_path or str(default_config_path())
        self.cfg: EnvConfig = load_env_config(cfg_path)
        self._map_path = map_path
        self._game_map = game_map
        self.render_mode = render_mode
        if render_style is None:
            self.render_style = "showcase" if render_mode == "human" else "lite"
        else:
            self.render_style = render_style

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

    @property
    def state(self):
        """当前仿真状态（调试 / 人玩 HUD）。"""
        return self._state

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            if "map_path" in options:
                self._map_path = options["map_path"]
                self._game_map = None
            if "game_map" in options:
                self._game_map = options["game_map"]
            if options.get("regen_maze"):
                from tank_sim.core.maze_gen import generate_maze

                maze_seed = options.get("maze_seed", seed)
                self._game_map = generate_maze(
                    cols=int(options.get("maze_cols", 11)),
                    rows=int(options.get("maze_rows", 7)),
                    cell_px=self.cfg.map.cell_px,
                    wall_thickness=self.cfg.map.wall_thickness,
                    seed=maze_seed,
                    openness=float(options.get("maze_openness", 0.12)),
                )
                self._map_path = None
        self._state = create_initial_state(
            self.cfg, self._map_path, game_map=self._game_map
        )
        self._obs_builder.reset()
        self._reward_state = RewardState()
        obs = self._pair_obs()
        return obs, {"step": 0}

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        bool,
        bool,
        dict[str, Any],
    ]:
        prev = self._state
        intent_red = continuous_to_intent(actions["red"])
        intent_blue = continuous_to_intent(actions["blue"])
        self._state = step_world(
            self._state,
            intent_red,
            intent_blue,
            self.cfg.sim,
        )
        self._obs_builder.on_step_end(self._state)

        rewards = {
            "red": compute_reward_for_side(
                prev,
                self._state,
                "red",
                self.cfg.reward,
                self._reward_state,
                bullet_speed=self.cfg.sim.bullet.speed,
                fire_intent=bool(intent_red.fire),
            ),
            "blue": compute_reward_for_side(
                prev,
                self._state,
                "blue",
                self.cfg.reward,
                self._reward_state,
                bullet_speed=self.cfg.sim.bullet.speed,
                fire_intent=bool(intent_blue.fire),
            ),
        }
        terminated = self._state.terminated
        truncated = self._state.step >= self.cfg.sim.max_episode_steps
        info = {
            "winner": self._state.winner,
            "step": self._state.step,
            "red_killed_by": self._state.red_killed_by,
            "blue_killed_by": self._state.blue_killed_by,
        }
        return self._pair_obs(), rewards, terminated, truncated, info

    def _pair_obs(self) -> dict[str, np.ndarray]:
        return {
            "red": self._obs_builder.build(self._state, "red"),
            "blue": self._obs_builder.build(self._state, "blue"),
        }

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
