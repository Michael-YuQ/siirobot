"""
批量实验启动器 — 顺序或选择性运行 36 个实验

用法:
    # 运行全部 36 个实验
    python -m experiments.run_all --headless

    # 只跑某个生成器
    python -m experiments.run_all --generators G1 G2 --headless

    # 只跑某个方法
    python -m experiments.run_all --methods atpc paired --headless

    # 指定种子
    python -m experiments.run_all --seeds 42 123 --headless

    # 带上传
    python -m experiments.run_all --upload_url http://server:8080/upload --headless
"""
import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.config import SEEDS, METHODS, GENERATORS, TRAIN_CFG


def build_command(generator, method, seed, args):
    """构建单个实验的命令行"""
    cmd = [
        sys.executable, "-m", "experiments.run_experiment",
        "--generator", generator,
        "--method", method,
        "--seed", str(seed),
        "--max_iterations", str(args.max_iterations),
        "--device", args.device,
    ]
    if args.headless:
        cmd.append("--headless")
    if args.no_upload:
        cmd.append("--no_upload")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="批量运行 AT-PC 跨生成器实验")
    parser.add_argument("--generators", nargs="+", default=list(GENERATORS.keys()),
                        choices=list(GENERATORS.keys()))
    parser.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                        choices=list(METHODS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--max_iterations", type=int, default=TRAIN_CFG["max_iterations"])
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_upload", action="store_true",
                        help="禁用上传")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印命令，不执行")
    args = parser.parse_args()

    # 生成实验列表
    experiments = []
    for gen in args.generators:
        for method in args.methods:
            for seed in args.seeds:
                experiments.append((gen, method, seed))

    total = len(experiments)
    print("=" * 70)
    print(f"  AT-PC Cross-Generator Experiments")
    print(f"  Total: {total} runs")
    print(f"  Generators: {args.generators}")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Iterations: {args.max_iterations}")
    print("=" * 70)

    if args.dry_run:
        for i, (g, m, s) in enumerate(experiments):
            cmd = build_command(g, m, s, args)
            print(f"[{i+1}/{total}] {' '.join(cmd)}")
        return

    # 顺序执行
    results = []
    global_start = time.time()

    for i, (gen, method, seed) in enumerate(experiments):
        exp_id = f"{method}-{gen}-seed{seed}"
        print(f"\n{'#'*70}")
        print(f"  [{i+1}/{total}] Starting: {exp_id}")
        print(f"  Time elapsed: {(time.time()-global_start)/60:.1f} min")
        print(f"{'#'*70}\n")

        cmd = build_command(gen, method, seed, args)
        t0 = time.time()

        try:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
            status = "OK"
        except subprocess.CalledProcessError as e:
            status = f"FAILED (code {e.returncode})"
        except KeyboardInterrupt:
            print("\n[Interrupted] Stopping batch run.")
            break

        elapsed = time.time() - t0
        results.append({"experiment": exp_id, "status": status,
                         "time_min": elapsed / 60})
        print(f"\n  [{exp_id}] {status} — {elapsed/60:.1f} min")

    # 汇总
    total_time = (time.time() - global_start) / 60
    print(f"\n{'='*70}")
    print(f"  All experiments finished — {total_time:.1f} min total")
    print(f"{'='*70}")
    for r in results:
        print(f"  {r['experiment']}: {r['status']} ({r['time_min']:.1f} min)")

    # 保存运行记录
    log_path = os.path.join("logs", "experiments",
                            f"batch_log_{datetime.now().strftime('%m%d_%H%M%S')}.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        for r in results:
            f.write(f"{r['experiment']}\t{r['status']}\t{r['time_min']:.1f}min\n")
    print(f"  Batch log: {log_path}")


if __name__ == "__main__":
    main()
