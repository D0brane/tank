# 仿真手感检查手册

通过 **人玩** 对照 Tank Trouble 2（或 [GOOFR 网页 clone](https://goofr-group.github.io/tank-trouble/)）判断物理是否接近原版。训练前建议至少完成下面 **一轮**。

**比例标定见 [spec_physics_calib.md](spec_physics_calib.md)**（格∶车∶墙∶弹∶帧率；边墙非实心格）。

## 启动方式

```bash
cd /home/d0brane/tank
conda activate tank

# 你 vs 静止靶（手感调试推荐，蓝方不动）
python -m apps.play --mode duel --layout wasd --side red --opponent none --hud

# 空地图练反射与 CD（蓝方仍是静止靶）
python -m apps.play --mode duel --layout wasd --opponent none --map assets/maps/empty.txt --hud

# 需要陪练时再开保守规则 Bot（约 1.5s 后才开火，不会开局冲锋秒杀）
python -m apps.play --mode duel --layout wasd --opponent rule --hud

# 本地双人
python -m apps.play --mode pvp --layout wasd --hud
```

HUD（`H` 切换）显示：步数、子弹数、双方开火 CD、**车体 OBB 尺寸**、移速/弹速/CD。

## 检查清单

### 1. 移动与转向 / 矩形碰撞箱

| 检查项 | 期望 | 不对时改什么 |
|--------|------|----------------|
| 按住前进/后退 | **匀速**，无加减速 | `sim.tank.speed_forward` |
| 按住左转/右转 | **匀速**旋转，可同时走；贴墙转不动 | `sim.tank.angular_speed` |
| 车体外形 | 画面为 **旋转矩形**，与 hitbox 一致 | `width` / `height` |
| 顶墙 | 不能穿墙；可沿墙滑动（轴分解） | `width`/`height`、碰撞逻辑 |
| 窄道 | 单格走廊净宽 ≥ 车对角线（标定 ≈56 vs ≈34），能原地转弯；墙是细线不是实心砖 | `cell_px`/`wall_thickness`；地图须为边墙 |
| 顶牛 | 两车 **不能重叠**，对冲会停住 | 车-车阻挡 |

### 2. 开火与 CD

| 检查项 | 期望 |
|--------|------|
| 连按开火 | 有 **明显间隔**，不能每帧一发 |
| HUD 中 CD | 开火后从 `fire_cooldown_frames` 倒数到 0 |
| 自杀 | 贴墙朝墙开火会 **弹回来杀自己** |
| 命中 | 擦过矩形边角也能击杀（圆弹 vs OBB） |

参数：`sim.fire_cooldown_frames`（默认 30 @ 60fps ≈ 0.5s）。

### 3. 子弹与反弹

| 检查项 | 期望 |
|--------|------|
| 弹速 | 明显快于坦克 |
| 直线飞行 | 稳定、不抖动 |
| 撞墙 | **法线镜面反射**；角点取最先命中边 |
| 寿命 | 未命中则持续反弹 |

参数：`sim.bullet.speed`、`sim.bullet.substeps`（穿墙则增大 substeps）。

### 4. 与原版粗对比（可选）

1. 浏览器打开 TT2 / GOOFR，选 **相似迷宫**。  
2. 记：**横穿地图约几帧**、**转一圈约几帧**、**两发间隔约几帧**。  
3. 在本仿真空地图重复，微调 `configs/env/sim_p0_tt2_classic.yaml` 直到量级一致。

### 5. 训练相关（手感 OK 后再看）

- 规则 Bot 能否正常对射 → `rule_bot_v1`  
- `pytest` 含 OBB / 反射 / 顶牛 / 重叠不变量  

## 键位对照（`--layout tt2`）

| 阵营 | 前进 | 后退 | 左转 | 右转 | 开火 |
|------|------|------|------|------|------|
| 红 P1 | E | D | S | F | Q |
| 蓝 P2 | ↑ | ↓ | ← | → | M |

`--layout wasd`：红方 WASD + Space；双人时蓝方仍用方向键 + M。
