"""地图连通性。"""

import pytest

from tank_sim.core.map_loader import load_map, world_to_tile
from tank_sim.core.maze_gen import _bfs_reachable, generate_maze


def test_maze_small_spawns_connected():
    gm = load_map("assets/maps/maze_small.txt", cell_px=60, wall_thickness=4)
    rx, ry = world_to_tile(gm.spawn_red[0], gm.spawn_red[1], gm.cell_px)
    bx, by = world_to_tile(gm.spawn_blue[0], gm.spawn_blue[1], gm.cell_px)
    assert (bx, by) in _bfs_reachable(gm, (rx, ry))


def test_empty_spawns_connected():
    gm = load_map("assets/maps/empty.txt", cell_px=60, wall_thickness=4)
    rx, ry = world_to_tile(gm.spawn_red[0], gm.spawn_red[1], gm.cell_px)
    bx, by = world_to_tile(gm.spawn_blue[0], gm.spawn_blue[1], gm.cell_px)
    assert (bx, by) in _bfs_reachable(gm, (rx, ry))


def test_disconnected_ascii_raises(tmp_path):
    bad = tmp_path / "split.txt"
    bad.write_text(
        "#####\n#R.#B#\n#####\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="不连通"):
        load_map(bad, cell_px=60, wall_thickness=4)


def test_generated_mazes_connected():
    for seed in range(20):
        m = generate_maze(cols=11, rows=7, seed=seed)
        rx = int(m.spawn_red[0] // m.cell_px)
        ry = int(m.spawn_red[1] // m.cell_px)
        bx = int(m.spawn_blue[0] // m.cell_px)
        by = int(m.spawn_blue[1] // m.cell_px)
        assert (bx, by) in _bfs_reachable(m, (rx, ry)), seed
