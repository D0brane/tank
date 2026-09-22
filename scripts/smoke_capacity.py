"""冒烟训练 + n_envs / device 容量探测（conda tank）。"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np


def _vram_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None


def _reset_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    gc.collect()


def run_one(
    *,
    n_envs: int,
    device: str,
    total_timesteps: int,
    n_steps: int,
    config_path: str,
    log_dir: Path,
) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    from tank_rl.curriculum.config import load_curriculum_aim_config
    from tank_rl.curriculum.scheduler import CurriculumScheduler
    from tank_rl.train.curriculum_env import make_curriculum_env

    cfg = load_curriculum_aim_config(config_path)
    cfg.n_envs = n_envs
    # dataclass frozen? CurriculumAimConfig is not frozen - it's a normal dataclass
    scheduler = CurriculumScheduler(cfg)

    def _thunk(rank: int):
        def _init():
            local = CurriculumScheduler(cfg)
            env = make_curriculum_env(cfg, local)
            env.reset(seed=cfg.seed + rank)
            return env

        return _init

    _reset_vram()
    vec = DummyVecEnv([_thunk(i) for i in range(n_envs)])
    vec = VecFrameStack(vec, n_stack=cfg.frame_stack)

    model = PPO(
        "MlpPolicy",
        vec,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=min(256, n_steps * n_envs),
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [64, 64]},
        seed=0,
        verbose=0,
        device=device,
        tensorboard_log=None,
    )

    t0 = time.perf_counter()
    ok = True
    err = ""
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
    except Exception as e:  # noqa: BLE001
        ok = False
        err = f"{type(e).__name__}: {e}"
    elapsed = max(1e-6, time.perf_counter() - t0)
    fps = total_timesteps / elapsed if ok else 0.0
    vram = _vram_mb()

    vec.close()
    del model, vec
    _reset_vram()

    return {
        "n_envs": n_envs,
        "device": device,
        "ok": ok,
        "err": err,
        "elapsed_s": round(elapsed, 2),
        "fps": round(fps, 1),
        "vram_mb": None if vram is None else round(vram, 1),
        "n_steps": n_steps,
        "timesteps": total_timesteps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train/curriculum_aim_smoke.yaml",
    )
    parser.add_argument("--timesteps", type=int, default=16384)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument(
        "--n-envs-list",
        default="1,2,4,8,16,24,32",
        help="逗号分隔的 n_envs 探测列表",
    )
    parser.add_argument("--devices", default="cuda,cpu")
    args = parser.parse_args()

    import torch

    print("=" * 64)
    print("冒烟 / 容量探测")
    print(f"  torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu={torch.cuda.get_device_name(0)}")
    print(f"  timesteps/试次={args.timesteps}  n_steps={args.n_steps}")
    print("=" * 64)

    n_list = [int(x) for x in args.n_envs_list.split(",") if x.strip()]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if "cuda" in devices and not torch.cuda.is_available():
        devices = [d for d in devices if d != "cuda"]
        print("CUDA 不可用，跳过 cuda")

    results = []
    log_dir = Path("runs/smoke_capacity")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1) 冒烟：cuda + 默认 8
    smoke_n = 8 if 8 in n_list else n_list[0]
    print(f"\n[冒烟] device=cuda n_envs={smoke_n} …")
    r0 = run_one(
        n_envs=smoke_n,
        device="cuda" if torch.cuda.is_available() else "cpu",
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        config_path=args.config,
        log_dir=log_dir,
    )
    results.append(r0)
    status = "OK" if r0["ok"] else f"FAIL {r0['err']}"
    print(
        f"  → {status}  FPS={r0['fps']}  "
        f"耗时={r0['elapsed_s']}s  VRAM峰值={r0['vram_mb']}MB"
    )
    if not r0["ok"]:
        print("冒烟失败，停止容量扫描")
        return

    # 2) 容量扫描
    for device in devices:
        print(f"\n[扫描] device={device}")
        for n in n_list:
            print(f"  n_envs={n} …", flush=True)
            r = run_one(
                n_envs=n,
                device=device,
                total_timesteps=args.timesteps,
                n_steps=args.n_steps,
                config_path=args.config,
                log_dir=log_dir,
            )
            results.append(r)
            if r["ok"]:
                print(
                    f"    OK  FPS={r['fps']:8.1f}  "
                    f"耗时={r['elapsed_s']:6.2f}s  VRAM={r['vram_mb']}MB"
                )
            else:
                print(f"    FAIL {r['err']}")
                # 同设备更大 n_envs 大概率也挂，可继续试以找上限

    # 汇总
    print("\n" + "=" * 64)
    print(f"{'device':6} {'n_envs':>6} {'FPS':>10} {'VRAM_MB':>10} {'ok':>4}")
    print("-" * 64)
    best = None
    for r in results:
        v = "-" if r["vram_mb"] is None else f"{r['vram_mb']:.0f}"
        print(
            f"{r['device']:6} {r['n_envs']:6d} {r['fps']:10.1f} {v:>10} "
            f"{'Y' if r['ok'] else 'N':>4}"
        )
        if r["ok"] and (best is None or r["fps"] > best["fps"]):
            best = r
    print("=" * 64)
    if best:
        print(
            f"推荐（本机实测最高吞吐）: device={best['device']} "
            f"n_envs={best['n_envs']}  FPS≈{best['fps']}"
        )
        # 相对正式 yaml 的建议
        print(
            "正式训练可在 curriculum_aim.yaml 设 env.n_envs 为上述值附近；"
            "当前训练路径为 DummyVecEnv（同进程），过大 n_envs 未必线性加速。"
        )


if __name__ == "__main__":
    main()
