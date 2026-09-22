import numpy as np

from tank_rl.vec_env.strided_stack import (
    StridedStackEnv,
    stacked_obs_dim,
)


def test_stacked_obs_dim_with_mean():
    assert stacked_obs_dim(58, 3) == 92
    assert stacked_obs_dim(58, 3, action_dim=3) == 101


def test_strided_stack_env_appends_mean():
    class _Fake:
        observation_space = type("S", (), {"shape": (2,)})()
        action_space = type("A", (), {"shape": (3,)})()

        def reset(self, **kwargs):
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action):
            return np.ones(2, dtype=np.float32), 0.0, False, False, {}

        def close(self):
            pass

    env = StridedStackEnv(_Fake(), 3, 3, action_dim=3)
    obs, _ = env.reset()
    assert obs.shape == (stacked_obs_dim(2, 3, action_dim=3),)
    env.record_action_mean(np.array([0.5, -0.2, 1.0], dtype=np.float32))
    obs2, *_ = env.step(np.zeros(3))
    assert obs2.shape == obs.shape
    # 非 58 维走整帧堆叠；均值段在末尾
    np.testing.assert_allclose(obs2[-3:], [0.5, -0.2, 1.0], atol=1e-6)
