# 观测向量 50 维（obs_version=3，无墙雷达）

历史变体：`wall_radar_rays=0` 时单帧 50 维。瞄准课程已恢复 8 向雷达，见 [`spec_obs_58.md`](spec_obs_58.md) 与 [`configs/env/sim_p0_tt2_aim_open.yaml`](../configs/env/sim_p0_tt2_aim_open.yaml)（当前 **58** 维）。

维度公式仍为：`1 + 6 + 4×bullet_slots + wall_radar_rays + 3`。
