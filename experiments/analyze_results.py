"""
结果分析与可视化 — 聚合 36 个实验的评估结果

用法:
    python -m experiments.analyze_results --results_dir logs/experiments

输出:
    - 对比表格 (终端 + CSV)
    - 训练曲线图 (PNG)
    - 假设检验结果
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.config import METHODS, GENERATORS, SEEDS


def load_all_results(results_dir):
    """扫描 results_dir 下所有实验目录，加载 eval_results.json 和 training_stats.json"""
    data = {}  # key: (method, generator, seed)
    if not os.path.isdir(results_dir):
        print(f"Directory not found: {results_dir}")
        return data

    for dirname in os.listdir(results_dir):
        dirpath = os.path.join(results_dir, dirname)
        if not os.path.isdir(dirpath):
            continue

        # 解析实验 ID: method-generator-seedN_timestamp
        parts = dirname.split("-")
        if len(parts) < 3:
            continue
        method = parts[0]
        gen = parts[1]
        seed_part = parts[2].split("_")[0]  # seed42
        if not seed_part.startswith("seed"):
            continue
        seed = int(seed_part.replace("seed", ""))

        entry = {"dir": dirpath, "method": method, "generator": gen, "seed": seed}

        # 加载评估结果
        eval_path = os.path.join(dirpath, "eval_results.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                entry["eval"] = json.load(f)

        # 加载训练统计
        stats_path = os.path.join(dirpath, "training_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                entry["stats"] = json.load(f)

        data[(method, gen, seed)] = entry

    return data


def print_comparison_table(data):
    """打印方法 × 生成器对比表"""
    print("\n" + "=" * 90)
    print("  METHOD × GENERATOR COMPARISON (Mean Reward ± Std across seeds)")
    print("=" * 90)

    header = f"{'Method':<10}"
    for g in GENERATORS:
        header += f"  {g:>16}"
    header += f"  {'Average':>16}"
    print(header)
    print("-" * 90)

    for m in METHODS:
        row = f"{m:<10}"
        method_vals = []
        for g in GENERATORS:
            vals = []
            for s in SEEDS:
                entry = data.get((m, g, s))
                if entry and "eval" in entry:
                    vals.append(entry["eval"]["mean_reward_across_terrains"])
            if vals:
                mean, std = np.mean(vals), np.std(vals)
                row += f"  {mean:>7.2f}±{std:<6.2f}"
                method_vals.extend(vals)
            else:
                row += f"  {'N/A':>16}"
        if method_vals:
            row += f"  {np.mean(method_vals):>7.2f}±{np.std(method_vals):<6.2f}"
        else:
            row += f"  {'N/A':>16}"
        print(row)

    print()

    # 成功率表
    print("  METHOD × GENERATOR COMPARISON (Success Rate %)")
    print("-" * 90)
    header = f"{'Method':<10}"
    for g in GENERATORS:
        header += f"  {g:>16}"
    print(header)
    print("-" * 90)

    for m in METHODS:
        row = f"{m:<10}"
        for g in GENERATORS:
            vals = []
            for s in SEEDS:
                entry = data.get((m, g, s))
                if entry and "eval" in entry:
                    vals.append(entry["eval"]["mean_success_across_terrains"] * 100)
            if vals:
                mean, std = np.mean(vals), np.std(vals)
                row += f"  {mean:>6.1f}%±{std:<5.1f}%"
            else:
                row += f"  {'N/A':>16}"
        print(row)

    print("=" * 90)


def compute_hypothesis_tests(data):
    """假设检验: AT-PC vs PAIRED, AT-PC vs DR"""
    print("\n  HYPOTHESIS TESTS (Welch's t-test, per generator)")
    print("-" * 70)

    for g in GENERATORS:
        atpc_vals = []
        paired_vals = []
        dr_vals = []
        for s in SEEDS:
            for m, lst in [("atpc", atpc_vals), ("paired", paired_vals), ("dr", dr_vals)]:
                entry = data.get((m, g, s))
                if entry and "eval" in entry:
                    lst.append(entry["eval"]["mean_reward_across_terrains"])

        print(f"\n  Generator {g}:")
        if len(atpc_vals) >= 2 and len(paired_vals) >= 2:
            from scipy import stats as sp_stats
            t, p = sp_stats.ttest_ind(atpc_vals, paired_vals, equal_var=False)
            win = "AT-PC" if np.mean(atpc_vals) > np.mean(paired_vals) else "PAIRED"
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "ns"))
            print(f"    AT-PC vs PAIRED: t={t:.3f}, p={p:.4f} {sig} (winner: {win})")

        if len(atpc_vals) >= 2 and len(dr_vals) >= 2:
            from scipy import stats as sp_stats
            t, p = sp_stats.ttest_ind(atpc_vals, dr_vals, equal_var=False)
            win = "AT-PC" if np.mean(atpc_vals) > np.mean(dr_vals) else "DR"
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "ns"))
            print(f"    AT-PC vs DR:     t={t:.3f}, p={p:.4f} {sig} (winner: {win})")


def save_csv(data, output_dir):
    """保存 CSV 格式的结果"""
    csv_path = os.path.join(output_dir, "comparison_results.csv")
    with open(csv_path, "w") as f:
        f.write("method,generator,seed,mean_reward,success_rate\n")
        for (m, g, s), entry in sorted(data.items()):
            if "eval" in entry:
                rew = entry["eval"]["mean_reward_across_terrains"]
                suc = entry["eval"]["mean_success_across_terrains"]
                f.write(f"{m},{g},{s},{rew:.4f},{suc:.4f}\n")
    print(f"  CSV saved: {csv_path}")


def plot_training_curves(data, output_dir):
    """绘制训练曲线"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    for g in GENERATORS:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for m in METHODS:
            all_curves = []
            for s in SEEDS:
                entry = data.get((m, g, s))
                if entry and "stats" in entry:
                    rewards = [e.get("reward", 0) for e in entry["stats"]]
                    all_curves.append(rewards)
            if all_curves:
                min_len = min(len(c) for c in all_curves)
                arr = np.array([c[:min_len] for c in all_curves])
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                # 平滑
                window = min(50, len(mean) // 5) if len(mean) > 10 else 1
                if window > 1:
                    kernel = np.ones(window) / window
                    mean = np.convolve(mean, kernel, mode="valid")
                    std = np.convolve(std, kernel, mode="valid")
                x = np.arange(len(mean))
                ax.plot(x, mean, label=METHODS[m]["name"])
                ax.fill_between(x, mean - std, mean + std, alpha=0.2)

        ax.set_title(f"Training Curves — Generator {g}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig_path = os.path.join(output_dir, f"training_curve_{g}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="AT-PC 实验结果分析")
    parser.add_argument("--results_dir", type=str,
                        default="logs/experiments")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading experiment results...")
    data = load_all_results(args.results_dir)
    print(f"  Found {len(data)} experiment runs")

    if not data:
        print("  No results found. Run experiments first.")
        return

    print_comparison_table(data)

    try:
        compute_hypothesis_tests(data)
    except ImportError:
        print("  scipy not available, skipping hypothesis tests")

    save_csv(data, args.output_dir)
    plot_training_curves(data, args.output_dir)

    print(f"\nAnalysis complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
