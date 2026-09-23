# 瞄准课程（curriculum_aim）

配置：[`configs/train/curriculum_aim.yaml`](../configs/train/curriculum_aim.yaml)（参数均有中文注释）

环境：[`configs/env/sim_p0_tt2_aim_open.yaml`](../configs/env/sim_p0_tt2_aim_open.yaml) — **58 维**观测（含 8 向墙雷达）、每局 `arena.mode=random_open` 随机尺寸空场（**世界边界为外框墙**，无内墙）；选择性稀疏堆叠 `frame_stack=3`、`frame_stride=3`（敌方/墙/A* 取 **t-6,t-3,t**；己方开火与子弹仅当前帧）→ 输入 **92**；`stack_action_mean=true` → **101**。

贴敌惩罚：`enemy_proximity_scale × (1 - d/margin)^power`（高次，贴脸才大罚）。

## 三阶段

1. **stage1_static** — 空场固定靶；`aim_mode=current`；晋级 `hit_rate（直击次数/己方开火）≥ 0.10` 且中位 TTK ≤ 900
2. **stage2_linear** — 直线靶，速度 0.3→1.0 退火；`aim_mode=lead`；晋级 `hit_rate ≥ 0.06`（满速后）
3. **stage3_turn_cruise** — 直行+偶发转弯，间隔 240→60；晋级 `hit_rate ≥ 0.05`（直行间隔退火到位）

晋级看评测窗口，**不用**平均回报。

## 训练

```bash
pip install -e '.[rl]'
python -m apps.train --config configs/train/curriculum_aim.yaml --log-dir runs/curriculum_aim
```

每次启动会在 `--log-dir` 下新建时间戳目录，例如：

```text
runs/curriculum_aim/
  20260921_190215/
    final_model.zip
    checkpoints/
    promotions/     # 每次晋级单独存盘（通过评测的权重 + 刚完成阶段的 meta）
    PPO_1/          # TensorBoard
```

启动横幅里的「本次运行目录」即实际路径。

### 终端输出含义

| 前缀 | 含义 |
|------|------|
| 启动横幅 | 配置路径、本次运行目录、总步数、并行环境、当前阶段与 TB 命令 |
| `[进度]` | 每个 rollout：步数/进度/FPS、阶段、对手速度、直行间隔、aim 权重 |
| `[存盘]` | 每 `checkpoint.every_timesteps`（默认 100000）步写入 `checkpoints/model_t*_stage*.zip` |
| `[晋级存盘]` | 晋级成功时写入 `promotions/stage{N}_*_to_*_t*.zip`（及 `.curriculum.json` / `.promotion.json`） |

训练结束另存 `final_model.zip`，并更新 `latest.zip` 软链便于续训。

### 断点续训

```bash
# 续同一 run（优先 latest.zip，否则取 checkpoints 中步数最大者）
python -m apps.train --config configs/train/curriculum_aim.yaml \
  --resume runs/curriculum_aim/20260921_190215 --device cpu

# 或指定某个权重
python -m apps.train --config configs/train/curriculum_aim.yaml \
  --resume runs/curriculum_aim/20260921_190215/checkpoints/model_t100000_stage0_stage1_static.zip
```

续训会恢复 `.curriculum.json` 中的阶段 / 对手参数，在**原 run 目录**继续写 checkpoint 与 TensorBoard；`total_timesteps` 仍为配置中的绝对目标（须大于已训步数）。

### TensorBoard

对比多次训练可指向实验根目录；只看某次则指向该次时间戳目录：

```bash
tensorboard --logdir runs/curriculum_aim
# 或
tensorboard --logdir runs/curriculum_aim/20260921_190215
```

浏览器打开 http://localhost:6006 ，关注：

| 标量 | 含义 |
|------|------|
| `curriculum/stage_index` | 当前阶段序号 |
| `curriculum/speed_scale` | 对手速度比例 |
| `curriculum/mean_straight_frames` | 平均直行间隔 |
| `curriculum/aim_align_scale` | 瞄准塑形权重 |
| `curriculum/promoted` | 本次评测是否晋级（0/1） |
| `eval/kill_rate` | 评测汇总：己方命中敌方次数 / 己方开火数（含反弹，按 hit 事件计） |
| `eval/hit_rate` | 评测汇总：己方未反弹命中次数 / 己方开火数（晋级默认指标） |
| `eval/median_ttk` | 评测局 TTK 中位数：首直击步数；无直击则为该局结束步数（≤ `eval.max_episode_steps`） |
| `reward/aim_align` 等 | 各奖励分项在本 rollout 的逐步均值 |
| `reward/total_mean` | 分项之和（逐步均值） |
| `rollout/*` / `train/*` | SB3 默认训练曲线 |

### 检查结果（实时）

```bash
python -m apps.eval_watch --model runs/curriculum_aim/20260921_190215/final_model.zip
```

默认 `--stage auto`：读取同目录 `*.curriculum.json`（存盘时写入的阶段与退火后对手参数）。无 meta 时回退到文件名里的 `stageN` 或阶段 0。手动 `--stage 1` 仍可用。

## 关键实现

| 模块 | 路径 |
|------|------|
| CurriculumBot | `tank_sim.bots.curriculum_bot` |
| 随机出生 | `tank_sim.core.spawn` |
| aim 塑形 | `tank_sim.reward.shaping`（`aim_align_scale` / `aim_mode`） |
| 调度 | `tank_rl.curriculum.scheduler` |
| PPO 入口 | `tank_rl.train.ppo_curriculum` |
