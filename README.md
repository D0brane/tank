# Tank — 坦克动荡 2 风格自博弈 RL

仿真包 `tank_sim` 提供 Gymnasium 环境与上帝模式 **99 维**单帧观测（obs v3，10 子弹固定槽）；算法包 `tank_rl` 提供瞄准课程 PPO（见 `configs/train/curriculum_aim.yaml` / `docs/curriculum_aim.md`）。

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
python -m apps.train --config configs/train/curriculum_aim.yaml
```

详见 [docs/curriculum_aim.md](docs/curriculum_aim.md)。

## 检查训练结果（实时观战）

加载 checkpoint，对手使用课程靶逻辑；一局结束默认自动重开。

```bash
pip install -e '.[rl,render]'
python -m apps.eval_watch \
  --model runs/curriculum_aim/final_model.zip \
  --curriculum configs/train/curriculum_aim.yaml \
  --stage 0
```

- `--stage`：阶段名或下标（`0`/`1`/`2`，或 `stage1_static` …）
- 默认用该阶段**退火终点**难度；加 `--easy` 用起点
- `Esc` 退出 · `R` 重开 · `1`/`2`/`3` 切换阶段
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
