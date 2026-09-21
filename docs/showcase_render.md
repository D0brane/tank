# 展示渲染说明

## 两种渲染

| style | 用途 | 入口 |
|-------|------|------|
| `showcase` | 人玩 / 观战，正版配色贴图 | `play` / `showcase` 默认 |
| `lite` | 训练 `rgb_array`、轻量调试 | env 非 human 默认 |

```bash
# 展示（默认 showcase）
python -m apps.play --gen-maze --hud

# 强制轻量
python -m apps.play --render-style lite --hud
```

## 视觉与碰撞对齐

- **坦克**：贴图内容区 = `height×width`（车头沿局部 +x），旋转中心 = 物理中心；`--hud` 时画绿色 OBB 轮廓核对
- **墙**：直接画 `game_map.wall_rects`（与碰撞 AABB 相同）
- **子弹**：圆半径 = `bullet.radius`

## 配色（经典 TT）

- 地面：浅灰棋盘 `#E7E7E7` / `#D2D2D2`
- 墙：`#4D4D4D`
- 1v1：红 vs 蓝
- 坦克：按物理 OBB 直接填色（无贴图旋转晕边）
- 弹：近黑
