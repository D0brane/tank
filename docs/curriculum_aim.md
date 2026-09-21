# 瞄准课程（curriculum_aim）

配置：[`configs/train/curriculum_aim.yaml`](../configs/train/curriculum_aim.yaml)（参数均有中文注释）

## 三阶段

1. **stage1_static** — 空场固定靶；`aim_mode=current`；晋级 `kill_rate ≥ 0.7`
2. **stage2_linear** — 直线靶，速度 0.3→1.0 退火；`aim_mode=lead`；晋级 `hit_rate ≥ 0.5`（满速后）
3. **stage3_turn_cruise** — 直行+偶发转弯，间隔 240→60；aim 权重降到 0.04；晋级 `kill_rate ≥ 0.55`

晋级看评测窗口，**不用**平均回报。

## 训练

```bash
pip install -e '.[rl]'
python -m apps.train --config configs/train/curriculum_aim.yaml --log-dir runs/curriculum_aim
```

### 终端输出含义

| 前缀 | 含义 |
|------|------|
| 启动横幅 | 配置路径、总步数、并行环境、当前阶段与 TB 命令 |
| `[进度]` | 每个 rollout：步数/进度/FPS、阶段、对手速度、直行间隔、aim 权重 |
| `[存盘]` | 每 `checkpoint.every_timesteps`（默认 100000）步写入 `checkpoints/model_t*_stage*.zip` |

训练结束另存 `final_model.zip`。

### TensorBoard

日志写在 `--log-dir` 下（SB3 子目录 `PPO_*`）。另开终端：

```bash
tensorboard --logdir runs/curriculum_aim
```

浏览器打开 http://localhost:6006 ，关注：

| 标量 | 含义 |
|------|------|
| `curriculum/stage_index` | 当前阶段序号 |
| `curriculum/speed_scale` | 对手速度比例 |
| `curriculum/mean_straight_frames` | 平均直行间隔 |
| `curriculum/aim_align_scale` | 瞄准塑形权重 |
| `curriculum/promoted` | 本次评测是否晋级（0/1） |
| `eval/kill_rate` | 评测击杀率 |
| `eval/hit_rate` | 评测命中率 |
| `eval/median_ttk` | 中位击杀耗时（无击杀为 -1） |
| `rollout/*` / `train/*` | SB3 默认训练曲线 |

### 检查结果（实时）

```bash
python -m apps.eval_watch --model runs/curriculum_aim/final_model.zip --stage 0
```

对手为对应课程阶段的 `CurriculumBot`；结束自动重开。按 `1`/`2`/`3` 切换阶段。

## 关键实现

| 模块 | 路径 |
|------|------|
| CurriculumBot | `tank_sim.bots.curriculum_bot` |
| 随机出生 | `tank_sim.core.spawn` |
| aim 塑形 | `tank_sim.reward.shaping`（`aim_align_scale` / `aim_mode`） |
| 调度 | `tank_rl.curriculum.scheduler` |
| PPO 入口 | `tank_rl.train.ppo_curriculum` |
