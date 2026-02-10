"""
批量评估脚本 — 自动扫描 logs/experiments/ 下所有实验，逐个评估

用法:
    python -m experiments.evaluate_all
    python -m experiments.evaluate_all --num_episodes 20   # 快速评估
    python -m experiments.evaluate_all --no_upload
"""
import os
import sys
import json
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.config import METHODS, GENERATORS, SEEDS


def parse_experiment_dir(dirname):
    """从目录名解析 method, generator, seed
    格式: method-generator-seedN_timestamp
    例如: atpc-G1-seed42_0210_120000
    """
    parts = dirname.split("-")
    if len(parts) < 3:
        return None
    method = parts[0]
    gen = parts[1]
    seed_part = parts[2].split("_")[0]
    if not seed_part.startswith("seed"):
        return None
    try:
        seed = int(seed_part.replace("seed", ""))
    except ValueError:
        return None
    if method not in METHODS or gen not in GENERATORS:
        return None
    return method, gen, seed


def main():
    parser = argparse.ArgumentParser(description="批量评估所有实验")
    parser.add_argument("--results_dir", type=str, default="logs/experiments")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_upload", action="store_true")
    parser.add_argument("--skip_evaluated", action="store_true", default=True,
                        help="跳过已有 eval_results.json 的实验")
    parser.add_argument("--force", action="store_true",
                        help="强制重新评估所有实验")
    args = parser.parse_args()

    if args.force:
        args.skip_evaluated = False

    # 扫描所有实验目录
    experiments = []
    for dirname in sorted(os.listdir(args.results_dir)):
        dirpath = os.path.join(args.results_dir, dirname)
        if not os.path.isdir(dirpath):
            continue
        parsed = parse_experiment_dir(dirname)
        if parsed is None:
            continue
        method, gen, seed = parsed

        # 检查是否有 checkpoint
        has_ckpt = os.path.exists(os.path.join(dirpath, "model_final.pt"))
        if not has_ckpt:
            pts = [f for f in os.listdir(dirpath)
                   if f.startswith("model_") and f.endswith(".pt")]
            has_ckpt = len(pts) > 0

        if not has_ckpt:
            print(f"  SKIP (no checkpoint): {dirname}")
            continue

        # 检查是否已评估
        has_eval = os.path.exists(os.path.join(dirpath, "eval_results.json"))
        if has_eval and args.skip_evaluated:
            print(f"  SKIP (already evaluated): {dirname}")
            continue

        experiments.append((method, gen, seed, dirpath, dirname))

    total = len(experiments)
    if total == 0:
        print("No experiments to evaluate.")
        if args.skip_evaluated:
            print("  (Use --force to re-evaluate all)")
        return

    print("=" * 70)
    print(f"  Batch Evaluation: {total} experiments")
    print("=" * 70)

    # 逐个评估 — 每个实验需要创建环境，所以用 subprocess
    import subprocess
    results = []
    global_start = time.time()

    for i, (method, gen, seed, dirpath, dirname) in enumerate(experiments):
        exp_id = f"{method}-{gen}-seed{seed}"
        print(f"\n{'#'*70}")
        print(f"  [{i+1}/{total}] Evaluating: {exp_id}")
        print(f"  Log dir: {dirpath}")
        print(f"{'#'*70}\n")

        cmd = [
            sys.executable, "-m", "experiments.evaluate_experiment",
            "--log_dir", dirpath,
            "--generator", gen,
            "--method", method,
            "--num_episodes", str(args.num_episodes),
            "--num_envs", str(args.num_envs),
            "--device", args.device,
        ]
        if args.no_upload:
            cmd.append("--no_upload")

        t0 = time.time()
        try:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
            status = "OK"
        except subprocess.CalledProcessError as e:
            status = f"FAILED (code {e.returncode})"
        except KeyboardInterrupt:
            print("\n[Interrupted]")
            break

        elapsed = time.time() - t0
        results.append({"experiment": exp_id, "status": status,
                         "time_min": elapsed / 60})
        print(f"  [{exp_id}] {status} — {elapsed/60:.1f} min")

    total_time = (time.time() - global_start) / 60
    print(f"\n{'='*70}")
    print(f"  Evaluation complete — {total_time:.1f} min total")
    print(f"{'='*70}")
    for r in results:
        print(f"  {r['experiment']}: {r['status']} ({r['time_min']:.1f} min)")

    print(f"\nNext step: python -m experiments.analyze_results")


if __name__ == "__main__":
    main()
