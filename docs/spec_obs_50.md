# 观测向量 50 维（obs_version=3，无墙雷达）

瞄准课程空场配置：[`configs/env/sim_p0_tt2_aim_open.yaml`](../configs/env/sim_p0_tt2_aim_open.yaml)。

| 索引 | 维数 | 含义 |
|------|------|------|
| 0 | 1 | 己方开火阶段 ∈ [-1,1] |
| 1-6 | 6 | 敌方：局部 dx, dy, dθ/π, 进退 m, 转向 r, 开火阶段 |
| 7-46 | 40 | **10** 子弹槽 × (局部 dx, dy, φ/π, valid) |
| 47-49 | 3 | A*：局部下一步角/π, 归一化路径长, reachable |

维度公式：`1 + 6 + 4×bullet_slots + wall_radar_rays + 3`，其中 `wall_radar_rays=0`。

相对 58 维：去掉 8 向墙雷达；有墙地图训练请用 classic（58）并恢复 `wall_radar_rays: 8`。

历史堆叠：`frame_stack × 50`，默认 16 帧 → **800** 维输入网络。
