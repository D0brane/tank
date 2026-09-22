# Tank — 坦克动荡 2 风格自博弈 RL

仿真包 `tank_sim` 提供 Gymnasium 环境；瞄准课程默认 **50 维**单帧观测（无墙雷达，见 `configs/env/sim_p0_tt2_aim_open.yaml` / `docs/spec_obs_50.md`），有墙地图仍可用 classic **58 维**（8 向雷达）。算法包 `tank_rl` 提供瞄准课程 PPO（见 `configs/train/curriculum_aim.yaml` / `docs/curriculum_aim.md`）。

## 安装（推荐 Conda 环境 `tank`）

```bash
cd /home/d0brane/tank
conda env create -f environment.yml   # 仅首次
conda activate tank
pip install -e ".[render,dev]"
# 训练用（CUDA）：
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[rl]'
```

已有环境时更新依赖：

```bash
conda activate tank
pip install -e ".[render,dev,rl]"
```

请勿再使用项目目录下的 `.venv`（已废弃清理）。

## 常用命令

```bash
conda activate tank
cd /home/d0brane/tank

# 瞄准课程训练（结果写入 runs/curriculum_aim/<YYYYMMDD_HHMMSS>/）
python -m apps.train \
  --config configs/train/curriculum_aim.yaml \
  --log-dir runs/curriculum_aim \
  --device cpu

# 断点续训（run 目录或具体 .zip；写回原目录）
python -m apps.train \
  --config configs/train/curriculum_aim.yaml \
  --resume runs/curriculum_aim/20260921_190215 \
  --device cpu

# TensorBoard（对比多次 run；浏览器 http://localhost:6006）
tensorboard --logdir runs/curriculum_aim
# 只看某次：
# tensorboard --logdir runs/curriculum_aim/20260921_190215

# 观战：默认加载最新一次训练的 final_model / checkpoint
python -m apps.eval_watch --stage auto

# 人玩：红方 WASD，蓝方静止靶
python -m apps.play --mode duel --layout wasd --opponent none --map assets/maps/empty.txt --hud
```

TensorBoard 关注：`eval/*`、`curriculum/*`、`reward/*`（各奖励分项逐步均值：aim_align、fire、kill、move_switch、wall_proximity、enemy_proximity 等）。

## 人玩 / 手感调试

```bash
conda activate tank
cd /home/d0brane/tank

# 推荐：你控红，蓝方静止（测移动/反弹/CD）
python -m apps.play --mode duel --layout wasd --opponent none --hud

# 空地图
python -m apps.play --mode duel --layout wasd --opponent none --map assets/maps/empty.txt --hud

# 程序化迷宫（终端会打印 ASCII；游戏内按 G 换图）
python -m apps.play --mode duel --layout wasd --opponent none --gen-maze --seed 42 --hud
python -m apps.preview_maze --seed 42 --count 3
```

对局结束后**不会自动重开**，按 **R** 再来一局。详见 [docs/fidelity_playtest.md](docs/fidelity_playtest.md)。

## 瞄准课程训练

```bash
pip install -e '.[rl]'   # SB3 / torch
python -m apps.train \
  --config configs/train/curriculum_aim.yaml \
  --log-dir runs/curriculum_aim \
  --device cpu

# 断点续训
python -m apps.train \
  --config configs/train/curriculum_aim.yaml \
  --resume runs/curriculum_aim/<YYYYMMDD_HHMMSS> \
  --device cpu
```

每次启动在 `--log-dir` 下新建时间戳目录，例如 `runs/curriculum_aim/20260921_190215/`（含 `final_model.zip`、`checkpoints/`、`latest.zip`、TB 日志）。`--resume` 指向该目录或其中某个 `.zip` 时，恢复课程阶段并继续写同一目录。

详见 [docs/curriculum_aim.md](docs/curriculum_aim.md)。

## 检查训练结果（实时观战）

加载 checkpoint，对手使用课程靶逻辑；一局结束默认自动重开。

```bash
pip install -e '.[rl,render]'
python -m apps.eval_watch --stage 0
# 或指定某次 run：
python -m apps.eval_watch \
  --model runs/curriculum_aim/20260921_190215/final_model.zip \
  --curriculum configs/train/curriculum_aim.yaml \
  --stage 0
```

（省略 `--model` 时默认加载最新 run；`--stage auto`（默认）读同名 `.curriculum.json`，用存盘时的对手参数。）
- `--stage`：`auto` / 阶段名 / 下标；手动指定时仍可用 `--easy` 选退火起点
- `Esc` 退出 · `R` 重开 · `1`/`2`/`3` 切换阶段（手动难度）
- `--no-auto-restart`：结束后停住，需按 R

详见 [docs/curriculum_aim.md](docs/curriculum_aim.md)。

## 测试

```bash
pytest
```

## 目录

- `packages/tank_sim` — 物理、观测、环境（无 RL 依赖）
- `packages/tank_rl` — 瞄准课程调度与 SB3 PPO 训练
- `apps/` — 启动入口（play / train / showcase / eval_watch）
- `configs/` — YAML 配置
- `assets/maps/` — 地图
