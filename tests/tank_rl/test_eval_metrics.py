import numpy as np
import pytest

from tank_rl.eval.metrics import aggregate_bullet_rates, evaluate_duel_policy


def test_aggregate_bullet_rates():
    k, h = aggregate_bullet_rates(
        bullets_fired=100, hits_on_enemy=25, direct_hits_on_enemy=10
    )
    assert k == pytest.approx(0.25)
    assert h == pytest.approx(0.10)
    k0, h0 = aggregate_bullet_rates(
        bullets_fired=0, hits_on_enemy=0, direct_hits_on_enemy=0
    )
    assert k0 == 0.0 and h0 == 0.0


class _FakeEvalEnv:
    """可控直击步数的假环境：direct_at 步出现首直击；否则跑到 max_steps。"""

    def __init__(self, *, direct_at: int | None, max_steps: int = 100) -> None:
        self.direct_at = direct_at
        self.max_steps = max_steps
        self._step = 0
        self._bullets = 0

    def reset(self, **kwargs):
        self._step = 0
        self._bullets = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        self._bullets += 1
        direct = 0
        hits = 0
        if self.direct_at is not None and self._step >= self.direct_at:
            direct = 1
            hits = 1
        terminated = False
        truncated = self._step >= self.max_steps
        info = {
            "step": self._step,
            "bullets_fired": self._bullets,
            "hits_on_enemy": hits,
            "direct_hits_on_enemy": direct,
            "agent_won": False,
            "ttk": None,
        }
        return np.zeros(4, dtype=np.float32), 0.0, terminated, truncated, info


def test_eval_stops_on_first_direct_hit():
    env = _FakeEvalEnv(direct_at=42, max_steps=200)

    def predict(_obs, deterministic=True):
        return np.zeros(3, dtype=np.float32)

    metrics = evaluate_duel_policy(env, predict, n_episodes=1, progress_every=0)
    assert metrics.median_ttk == pytest.approx(42.0)
    # 每步记 1 发，第 42 步首直击 → 1/42
    assert metrics.hit_rate == pytest.approx(1.0 / 42.0)
    assert metrics.n_episodes == 1


def test_eval_ttk_caps_without_direct_hit():
    env = _FakeEvalEnv(direct_at=None, max_steps=60)

    def predict(_obs, deterministic=True):
        return np.zeros(3, dtype=np.float32)

    metrics = evaluate_duel_policy(env, predict, n_episodes=1, progress_every=0)
    assert metrics.median_ttk == pytest.approx(60.0)
    assert metrics.hit_rate == pytest.approx(0.0)
