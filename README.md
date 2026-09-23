# tank-rl

[Tank Trouble 2](https://tanktrouble.com/) 风格的 **1v1 坦克对战强化学习**：自研 god-mode 物理仿真 + Stable-Baselines3 PPO 瞄准课程。当前主线是在随机尺寸空场里，从静止靶 → 直线移动靶 → 直行偶发转弯，学会瞄准、开火与命中。

仓库：[github.com/D0brane/tank-rl](https://github.com/D0brane/tank-rl)

## 项目做什么

| 能力 | 说明 |
|------|------|
| 仿真 `tank_sim` | Gymnasium 环境：车体/炮塔、定速子弹、墙 AABB 反弹、开火冷却；无 RL 依赖 |
| 训练 `tank_rl` | 瞄准课程调度 + SB3 PPO（DummyVecEnv、选择性帧堆叠） |
| 人玩 / 观战 | `apps.play` 手感调试；`apps.eval_watch` 加载 checkpoint 实时观战 |
| 展示渲染 | `apps.showcase` 录展示画面（见 `docs/showcase_render.md`） |

动作是连续 3 维意图（进退 / 转向 / 开火），再映射到离散控制；观测默认 **58 维**（obs v3），训练时选择性稀疏堆叠到约 **92–101** 维。

## 观测与环境（瞄准课程）

环境配置：[`configs/env/sim_p0_tt2_aim_open.yaml`](configs/env/sim_p0_tt2_aim_open.yaml)

- **地图**：每局 `random_open` 随机尺寸空场；世界边界为外框墙，**无内墙**
- **单帧 58 维**：己方开火阶段、敌方相对位姿与意图、10 个子弹身份槽、8 向墙雷达、A* 提示（详见 [`docs/spec_obs_58.md`](docs/spec_obs_58.md)）
- **堆叠**：`frame_stack=3`、`frame_stride=3`（敌方/墙/A* 取 t−6,t−3,t；开火与子弹仅当前帧）→ 92；再拼动作均值 → 101

## 瞄准课程三阶段

配置：[`configs/train/curriculum_aim.yaml`](configs/train/curriculum_aim.yaml)（参数带中文注释）  
说明：[`docs/curriculum_aim.md`](docs/curriculum_aim.md)

1. **stage1_static** — 空场固定靶；`aim_mode=current`；晋级看 `hit_rate`（直击/开火）与中位 TTK  
2. **stage2_linear** — 直线靶，速度退火；`aim_mode=lead`  
3. **stage3_turn_cruise** — 直行 + 偶发转弯，转弯间隔退火  

晋级依据评测窗口的 **hit_rate / TTK**，不用平均回报。另有炮塔阶段配置 [`configs/train/curriculum_turret.yaml`](configs/train/curriculum_turret.yaml)。

## 安装（推荐 Conda 环境 `tank`）

```bash
cd /path/to/tank   # 或克隆后的目录
conda env create -f environment.yml   # 仅首次
conda activate tank
pip install -e ".[render,dev]"
# 训练（CUDA 示例）：
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[rl]'
```

已有环境时：

```bash
conda activate tank
pip install -e ".[render,dev,rl]"
```

请勿使用项目下的 `.venv`（已废弃）。

## 训练

```bash
conda activate tank

python -m apps.train \
  --config configs/train/curriculum_aim.yaml \
  --log-dir runs/curriculum_aim \
  --device cpu   # 有 GPU 可改为 cuda

# 断点续训（指向 run 目录或某个 .zip）
python -m apps.train \
  --config configs/train/curriculum_aim.yaml \
  --resume runs/curriculum_aim/<YYYYMMDD_HHMMSS> \
  --device cpu
```

每次启动在 `--log-dir` 下新建时间戳目录，例如：

```text
runs/curriculum_aim/20260921_190215/
  final_model.zip
  latest.zip          # 软链，便于续训
  checkpoints/        # 默认每 100k 环境步
  promotions/         # 晋级时单独存盘
  PPO_1/              # TensorBoard
```

短冒烟可用 [`configs/train/curriculum_aim_smoke.yaml`](configs/train/curriculum_aim_smoke.yaml)。

### TensorBoard

```bash
tensorboard --logdir runs/curriculum_aim
# 浏览器 http://localhost:6006
```

重点看：`eval/hit_rate`、`eval/kill_rate`、`eval/median_ttk`、`curriculum/*`、`reward/*`（aim_align、fire、kill、wall/enemy proximity 等）。

## 观战与人玩

```bash
# 观战：默认加载最新 run 的 final_model / checkpoint；Esc 退出，R 重开，1/2/3 切手动阶段
python -m apps.eval_watch --stage auto

# 人玩：红方 WASD，蓝方静止靶；局末不自动重开，按 R 再来
python -m apps.play --mode duel --layout wasd --opponent none --map assets/maps/empty.txt --hud

# 程序化迷宫（局内按 G 换图）
python -m apps.play --mode duel --layout wasd --opponent none --gen-maze --seed 42 --hud
```

手感与保真说明见 [`docs/fidelity_playtest.md`](docs/fidelity_playtest.md)。

## 测试

```bash
pytest
```

## 目录结构

```text
packages/tank_sim/   # 物理、观测、环境、课程靶 Bot（无 RL 依赖）
packages/tank_rl/    # 课程调度、SB3 PPO 训练入口
apps/                # play / train / eval_watch / showcase / preview_maze
configs/env/         # 仿真与观测 YAML
configs/train/       # 课程与 PPO YAML
assets/maps/         # 地图
docs/                # 观测规格、课程、渲染、手感文档
```

## 文档索引

| 文档 | 内容 |
|------|------|
| [`docs/curriculum_aim.md`](docs/curriculum_aim.md) | 瞄准课程阶段、存盘、TB、观战 |
| [`docs/spec_obs_58.md`](docs/spec_obs_58.md) | 58 维观测与堆叠 |
| [`docs/spec_physics_calib.md`](docs/spec_physics_calib.md) | 物理标定 |
| [`docs/fidelity_playtest.md`](docs/fidelity_playtest.md) | 人玩手感 |
| [`docs/showcase_render.md`](docs/showcase_render.md) | 展示渲染 |
