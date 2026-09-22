"""对比 frame_stack=1/4/8，各训练 total_timesteps 步（默认 1M），汇总 TensorBoard eval。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _write_config(base: Path, frame_stack: int, total_timesteps: int, out: Path) -> None:
    raw = yaml.safe_load(base.read_text(encoding="utf-8"))
    raw["total_timesteps"] = int(total_timesteps)
    raw["seed"] = 42
    env = raw.setdefault("env", {})
    env["frame_stack"] = int(frame_stack)
    if "frame_stride" not in env:
        env["frame_stride"] = 1
    # 消融：少写 checkpoint，保留评测节奏与主配置一致
    ck = raw.setdefault("checkpoint", {})
    ck["every_timesteps"] = max(int(total_timesteps), 1_000_000)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _latest_run_dir(parent: Path) -> Path | None:
    if not parent.is_dir():
        return None
    runs = sorted(
        (p for p in parent.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    return runs[0] if runs else None


def _read_eval_series(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ppo = run_dir / "PPO_1"
    if not ppo.is_dir():
        return {}
    acc = EventAccumulator(str(ppo), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in ("eval/hit_rate", "eval/kill_rate", "eval/median_ttk", "curriculum/stage_index"):
        if tag not in tags:
            continue
        out[tag] = [(e.step, e.value) for e in acc.Scalars(tag)]
    return out


def _summarize(series: dict[str, list[tuple[int, float]]], target_steps: int) -> dict:
    def last_at(tag: str) -> tuple[int, float] | None:
        pts = series.get(tag, [])
        if not pts:
            return None
        # 最接近 target_steps 的评测点
        best = min(pts, key=lambda x: abs(x[0] - target_steps))
        return best

    hit = last_at("eval/hit_rate")
    kill = last_at("eval/kill_rate")
    ttk = last_at("eval/median_ttk")
    stage = last_at("curriculum/stage_index")
    hit_pts = series.get("eval/hit_rate", [])
    best_hit = max(hit_pts, key=lambda x: x[1]) if hit_pts else None
    return {
        "eval_points": len(hit_pts),
        "last_hit": hit[1] if hit else None,
        "last_kill": kill[1] if kill else None,
        "last_ttk": ttk[1] if ttk else None,
        "last_eval_step": hit[0] if hit else None,
        "best_hit": best_hit[1] if best_hit else None,
        "best_hit_step": best_hit[0] if best_hit else None,
        "stage_index": stage[1] if stage else None,
        "promote_ready_hit": bool(hit and hit[1] >= 0.70),
        "promote_ready_ttk": bool(ttk and 0 < ttk[1] <= 900),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="frame_stack 1/4/8 消融训练")
    parser.add_argument(
        "--base-config",
        default=str(ROOT / "configs/train/curriculum_aim.yaml"),
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--stacks", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--log-root",
        default=str(ROOT / "runs/ablation_frame_stack"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只写配置，不训练",
    )
    args = parser.parse_args()

    base = Path(args.base_config)
    if not base.is_file():
        raise SystemExit(f"找不到配置: {base}")

    log_root = Path(args.log_root)
    cfg_dir = log_root / "_configs"
    results: list[dict] = []

    sys.path.insert(0, str(ROOT / "packages/tank_sim/src"))
    sys.path.insert(0, str(ROOT / "packages/tank_rl/src"))

    for fs in args.stacks:
        tag = f"fs{fs}"
        cfg_path = cfg_dir / f"curriculum_aim_{tag}_1m.yaml"
        _write_config(base, fs, args.total_timesteps, cfg_path)
        parent = log_root / tag
        row: dict = {"frame_stack": fs, "config": str(cfg_path), "log_parent": str(parent)}

        if args.dry_run:
            results.append(row)
            continue

        from tank_rl.train.ppo_curriculum import train_curriculum_aim

        t0 = time.perf_counter()
        print("\n" + "=" * 60)
        print(f"开始训练 frame_stack={fs}  →  {args.total_timesteps:,} 步")
        print("=" * 60)
        train_curriculum_aim(
            config_path=cfg_path,
            log_dir=parent,
            device=args.device,
        )
        wall = time.perf_counter() - t0
        run_dir = _latest_run_dir(parent)
        row["run_dir"] = str(run_dir) if run_dir else None
        row["wall_seconds"] = wall
        if run_dir:
            series = _read_eval_series(run_dir)
            row.update(_summarize(series, args.total_timesteps))
        results.append(row)

    out_path = log_root / "summary.json"
    log_root.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("消融汇总（写入", out_path, ")")
    print("=" * 60)
    hdr = (
        f"{'stack':>5}  {'last_hit':>8}  {'best_hit':>8}  "
        f"{'last_ttk':>8}  {'stage':>5}  {'wall_min':>8}"
    )
    print(hdr)
    for r in results:
        if args.dry_run:
            print(f"{r['frame_stack']:>5}  (dry-run)")
            continue
        lh = r.get("last_hit")
        bh = r.get("best_hit")
        lt = r.get("last_ttk")
        st = r.get("stage_index")
        wm = (r.get("wall_seconds") or 0) / 60.0
        print(
            f"{r['frame_stack']:>5}  "
            f"{lh if lh is None else f'{lh:.3f}':>8}  "
            f"{bh if bh is None else f'{bh:.3f}':>8}  "
            f"{lt if lt is None else f'{lt:.0f}':>8}  "
            f"{st if st is None else f'{st:.0f}':>5}  "
            f"{wm:>8.1f}"
        )


if __name__ == "__main__":
    main()
