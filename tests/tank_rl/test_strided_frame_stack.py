"""稀疏 / 选择性帧堆叠缓冲。"""

import numpy as np

from tank_rl.vec_env.strided_stack import (
    StridedFrameBuffer,
    stack_depth,
    stack_lags,
    stacked_obs_dim,
)
from tank_sim.observation.spec import (
    STACKABLE_SLICES,
    selective_stacked_dim,
)


def test_stack_lags_and_depth():
    assert stack_depth(3, 3) == 7
    assert stack_lags(3, 3) == [6, 3, 0]
    assert stack_depth(3, 1) == 3
    assert stack_lags(3, 1) == [2, 1, 0]


def test_strided_buffer_full_concat():
    buf = StridedFrameBuffer(3, 3, frame_dim=2, selective=False)
    for i in range(7):
        buf.append(np.array([float(i), float(i) * 10], dtype=np.float32))
    out = buf.stacked()
    assert out.shape == (6,)
    np.testing.assert_allclose(out[:2], [0.0, 0.0])
    np.testing.assert_allclose(out[2:4], [3.0, 30.0])
    np.testing.assert_allclose(out[4:6], [6.0, 60.0])


def test_strided_reset_zeros_past():
    buf = StridedFrameBuffer(3, 3, frame_dim=1, selective=False)
    buf.reset(np.array([1.0], dtype=np.float32))
    out = buf.stacked()
    np.testing.assert_allclose(out, [0.0, 0.0, 1.0])


def _fake_frame(i: int) -> np.ndarray:
    """58 维假帧：己开火=i，敌 dx=i+100，子弹槽全 i+200，墙/路径=i+300。"""
    f = np.zeros(58, dtype=np.float32)
    f[0] = float(i)
    f[1:7] = float(i + 100)
    f[7:47] = float(i + 200)
    f[47:58] = float(i + 300)
    return f


def test_selective_stacked_dim():
    assert selective_stacked_dim(3) == 92
    assert stacked_obs_dim(58, 3) == 92
    assert stacked_obs_dim(58, 3, action_dim=3) == 101


def test_selective_buffer_bullets_and_fire_current_only():
    buf = StridedFrameBuffer(3, 3, frame_dim=58, selective=True)
    for i in range(7):
        buf.append(_fake_frame(i))
    out = buf.stacked()
    assert out.shape == (92,)
    # stackable 17 × 3：lags 6,3,0 → frames 0,3,6 的 enemy+walls+planner
    sdim = sum(b - a for a, b in STACKABLE_SLICES)
    assert sdim == 17
    # lag6 = frame 0
    np.testing.assert_allclose(out[0:6], 100.0)  # enemy of frame 0
    np.testing.assert_allclose(out[6:17], 300.0)  # walls+planner of frame 0
    # lag3 = frame 3
    np.testing.assert_allclose(out[17:23], 103.0)
    np.testing.assert_allclose(out[23:34], 303.0)
    # lag0 = frame 6
    np.testing.assert_allclose(out[34:40], 106.0)
    np.testing.assert_allclose(out[40:51], 306.0)
    # current-only: self + bullets from frame 6
    np.testing.assert_allclose(out[51:52], 6.0)
    np.testing.assert_allclose(out[52:92], 206.0)
    # 确认不是旧帧的开火/子弹
    assert out[51] != 0.0
    assert out[52] != 200.0
