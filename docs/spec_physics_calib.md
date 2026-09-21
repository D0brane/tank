# TT2 物理比例标定（P0）

目标：在改地图/物理前，先锁定**格子∶车∶墙∶子弹∶帧率**的原版比例。  
结论：墙是**格子间的薄分割线**（edge wall），不是实心占格；走廊净宽约 **2 车宽**，必须能原地转弯。

---

## 1. 结论表（本仓库 P0 采用）

以 **cell = 60 px** 为基准（与经典克隆一致；OliverBryan 用 62，比例等价）。

| 量 | 绝对 (cell=60) | 相对 cell | 相对车宽(20) | 依据 |
|---|---|---|---|---|
| 格距 / 走廊中心距 | **60** | 1 | 3.0 | JustDoIt `GRID_SIZE`；OliverBryan `62` |
| 墙厚 | **4** | **1/15 ≈ 0.067** | 0.20 | JustDoIt `BLOCK_WIDTH=4`（首选） |
| 走廊净宽 | **≈56** (=60−4) | 0.933 | **≈2.8** | 几何推导；Wiki「约 2 车宽」同量级 |
| 车宽×车长 | **20×28** | 0.333×0.467 | 1×1.4 | JustDoIt / 多克隆共识 |
| 车对角线 | ≈34.4 | 0.573 | — | `hypot(20,28)`；**34.4 < 56 → 能转弯** |
| 子弹半径 | **2.5** | ≈0.042 | 0.125 | JustDoIt `Shell::RADIUS` |
| 仿真步频 | **60 TPS** | — | — | OliverBryan 固定 60；训练/游玩统一 |
| 车前进 (px/tick@60) | **2.24** (≈134 px/s) | **2.24 cell/s** | — | OliverBryan `1.386` Box2D×100 → 138.6 px/s @cell62，按 60/62 缩放到本仓库 cell |
| 子弹 (px/tick@60) | **3.77** (≈226 px/s) | **3.77 cell/s** | — | OliverBryan `234` px/s；与 mglyn 350/94≈3.72 cell/s 一致；**≈1.69×** 车速 |
| 转速 | **≈3.7°/tick** (≈0.065 rad) | — | — | OliverBryan `3.7°`；当前配置已接近 |
| 子弹寿命 | **≈10 s**（600 帧 @60） | — | — | OliverBryan `bulletTime=10`；simple-tank `10*TPS`；超时移除 |
| 同时弹数 | **每车最多 5** | — | — | JustDoIt `remainBullets=5`；经典 TT |
| 地图模型 | **边墙** `hasLeft`/`hasTop` | — | — | 全体正统实现；禁止实心格墙 |

### 为何「太窄转不过弯」

当前 `tile_px=32` + **实心占格墙** → 单格走廊净宽 ≈32，而车对角线 ≈34.4，**净宽 < 对角线**，物理上无法在走廊内旋转。  
原版走廊净宽 ≈56（或按 Wiki 约 2×20=40），都 ≥ 对角线。

---

## 2. 多源对照（原始常数）

### 2.1 JustDoIt0910/TankTrouble（经典 TT 结构最清晰）

| 常数 | 值 |
|---|---|
| 网格 | 11×7 |
| `GRID_SIZE` | 60 |
| `TANK_WIDTH/HEIGHT` | 20 / 28 |
| `BLOCK_WIDTH`（墙厚） | 4 |
| `Shell::RADIUS` | 2.5 |
| 墙几何 | 格子**边**上的线段，长度 ≈ `GRID_SIZE`，厚度 `BLOCK_WIDTH`，端点各外扩半墙厚 |
| 仿真 | `runEvery(0.01, moveAll)` → **100 Hz**；默认步进 1 px/tick → 车/弹名义 100 px/s（偏慢，作结构参考非速度金标） |
| 旋转 | `ROTATING_STEP=3` °/tick @100Hz |

### 2.2 OliverBryan/TankTroubleRecreation（自称 TT2 复刻）

| 常数 | 值 |
|---|---|
| 固定仿真 | **60 TPS** |
| 格距 | **62** px；9×9 |
| 墙 | 竖墙 `2×64`，横墙 `62×2` → **厚度 2**（更细） |
| 地图格式 | 每行 `gridX gridY hasLeftWall hasTopWall`（0=有墙） |
| 车线速度 | Box2D `1.386` × scale100 ≈ **138.6 px/s** ≈ 2.31 px/@60 |
| 角速度 | ≈ **3.7°/tick** |
| 子弹初速 | `234 * dir`（像素空间）≈ **234 px/s** ≈ 3.9/@60 |
| 子弹寿命 | **10 s** |

**速度金标**：本仓库车/弹速取 OliverBryan 的 **格/秒**，再乘本仓库 `cell_px=60`：
`tank = 138.6/62*60 ≈ 134 → 2.24 px/tick`，`bullet = 234/62*60 ≈ 226 → 3.77 px/tick`。

### 2.3 Lander-Hatsune/simple-tank（Flash TT2 重写）

| 常数 | 值 | / BLOCK_SIZE |
|---|---|---|
| `Display.FPS` / `TPS` | 60 / **200** | — |
| `BLOCK_SIZE` | 10 | 1 |
| `WALL_WIDTH` | 1 | **0.10** |
| `TANK_WIDTH×HEIGHT` | 3.25×5.5 | **0.325×0.55** |
| `BULLET_RADIUS` | 0.4 | **0.04** |
| 车前进 | 20 world-u/s | **2.0 cell/s** |
| 子弹 | 22.5 world-u/s | **2.25 cell/s**（≈1.13× 车） |
| 墙模型 | 格子间 vert/horiz 边墙 | — |

车∶格、弹∶格与 JustDoIt 几乎一致。

### 2.4 mglyn/TANKTROUBLE-pythonedition

| 常数 | 值 | / BLOCK_SIZE(94) |
|---|---|---|
| `FRAME_RATE` | 144（速度已按 60 归一） | — |
| `BLOCK_SIZE` / `WALL_SIZE` | 94 / 8 | wall **0.085** |
| 车 `PLAYER_PX×PY` | 41×27 | 0.436×0.287 |
| 弹 `ROUND` 9×9 | r≈4.5 | ≈0.048 |
| `BASE_MOVE_V` @60eq | **255 px/s** | ≈2.71 cell/s |
| `BASE_ROUND_V` @60eq | **350 px/s** | ≈3.72 cell/s（≈1.37× 车） |

弹的 **格/秒** 与 OliverBryan（3.77）几乎一致，交叉验证通过；车速 mglyn 偏快，不采用。

### 2.5 GOOFR-Group/tank-trouble（网页 clone，手感参考）

| 常数 | 值 |
|---|---|
| `Tank.speed` | **150** px/s（与 OliverBryan 138.6 接近） |
| `Bullet.speed` | **400** px/s（偏快，不作金标） |

### 2.6 官方侧旁证（Subterranean / Fandom）

- Laika AI 文：渲染/更新上限约 **30 Hz**；墙为**细边**，射线检测相对墙内缩 **5 px**（弹宽），墙缝重叠 **5 px**。
- Fandom Mazes：走廊「almost **2 tanks** wide」；墙为深灰分割，地面可通行。

Flash 原版精确帧率未从 SWF 反编译确认；游玩/训练取 **60 TPS**（与 OliverBryan 一致，且为 30 的倍数）。

---

## 3. 与当前 `sim_p0_tt2_classic.yaml` 的差距

| 项 | 当前 | 标定目标 | 影响 |
|---|---|---|---|
| 地图 | 实心 tile，`tile_px=32` | 边墙，cell=60，wall=4 | 走廊过窄、墙像砖块 |
| 车尺寸 | 20×28 | 20×28 | 已正确 |
| 车速 | 曾用 4.0（偏快） | **2.24** px/@60 | 按 OliverBryan 格速标定 |
| 弹速 | 曾用 5.8（偏快） | **3.77** px/@60 | 与 OliverBryan / mglyn 格速一致 |
| 弹半径 | 3.0 | 2.5 | 略大 |
| 转速 | 0.065 rad ≈3.7° | ≈3.7° | 已接近 |
| 帧率 | 60 | 60 | 已正确 |

---

## 4. 下一步实现约束（改代码时遵守）

1. **地图表示**：`h_walls[rows+1][cols]` + `v_walls[rows][cols+1]`（或 `hasLeft`/`hasTop`），碰撞用薄 AABB 列表，**不再**把墙当成占用整格的 solid tile。
2. **生成器**：递归分割 / Prim 只打通**边**；可选再随机拆掉若干内墙（OliverBryan 拆 10 面）以增加开阔度。
3. **A\***：在 **cell 中心图**上搜（边权=是否有墙），或把边墙栅格化到细网格；局部 7×7 观测按「中心是否可站 / 邻边是否有墙」编码。
4. **配置建议写入值**（待改 yaml，本文仅标定）：

```yaml
sim:
  fps: 60
  tank:
    width: 20.0
    height: 28.0
    speed_forward: 2.24     # px/tick @60 ≈ 134 px/s
    angular_speed: 0.065    # ≈ 3.7°/tick
  bullet:
    speed: 3.77             # px/tick @60 ≈ 226 px/s
    radius: 2.5
map:
  cell_px: 60
  wall_thickness: 4
  # 弃用 solid tile_px 作为墙模型
```

5. **验收**：单格宽走廊内，车以任意朝角应能完成 ≥90° 旋转且不卡死；目视墙为格子间细线。

---

## 6. 实现状态

已落地（边墙 `GameMap` + Prim 生成 + 薄墙碰撞/反射 + yaml 标定 + 渲染细线）：

- `packages/tank_sim/.../types.py` / `map_loader.py` / `maze_gen.py` / `collision.py`
- `configs/env/sim_p0_tt2_classic.yaml`
- 手感：`python -m apps.play --gen-maze --hud`

---

## 7. 来源链接

- [JustDoIt0910/TankTrouble](https://github.com/JustDoIt0910/TankTrouble) — `defs.h`, `Tank.h`, `Block.h`, `Shell.cc`, `LocalController.cc`
- [OliverBryan/TankTroubleRecreation](https://github.com/OliverBryan/TankTroubleRecreation) — `CLAUDE.md`, `Maze.cpp`, `Tank.cpp`
- [Lander-Hatsune/simple-tank](https://github.com/Lander-Hatsune/simple-tank) — `scripts/commons.js`
- [mglyn/TANKTROUBLE-pythonedition](https://github.com/mglyn/TANKTROUBLE-pythonedition) — `source/constants.py`
- [Subterranean: TankTrouble Single Player AI](https://www.subterraneansoftware.com/tanktrouble-single-player-ai/)
- [Fandom: Mazes](https://tanktrouble-tank-game.fandom.com/wiki/Mazes)
