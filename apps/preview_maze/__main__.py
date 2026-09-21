"""迷宫生成预览：打印 ASCII 并做连通性检查。"""

from __future__ import annotations

import argparse

from tank_sim.core.maze_gen import _bfs_reachable, _free_cells, generate_maze, maze_to_ascii


def main() -> None:
    parser = argparse.ArgumentParser(description="预览程序化迷宫")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cols", type=int, default=11)
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--openness", type=float, default=0.12)
    parser.add_argument("--count", type=int, default=3, help="连续生成几张")
    parser.add_argument("--cell", type=float, default=60.0)
    parser.add_argument("--wall", type=float, default=4.0)
    args = parser.parse_args()

    for i in range(args.count):
        seed = args.seed + i
        m = generate_maze(
            cols=args.cols,
            rows=args.rows,
            cell_px=args.cell,
            wall_thickness=args.wall,
            seed=seed,
            openness=args.openness,
        )
        free = _free_cells(m)
        rx = int(m.spawn_red[0] // m.cell_px)
        ry = int(m.spawn_red[1] // m.cell_px)
        bx = int(m.spawn_blue[0] // m.cell_px)
        by = int(m.spawn_blue[1] // m.cell_px)
        reach = _bfs_reachable(m, (rx, ry))
        ok = (bx, by) in reach
        n_h = sum(sum(1 for w in row if w) for row in m.h_walls)
        n_v = sum(sum(1 for w in row if w) for row in m.v_walls)
        print(
            f"===== seed={seed} size={m.cols}x{m.rows} "
            f"free={len(free)} edges={n_h + n_v} reachable={ok} ====="
        )
        print(maze_to_ascii(m))
        print()


if __name__ == "__main__":
    main()
