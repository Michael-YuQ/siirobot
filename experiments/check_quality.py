"""
检查第二轮训练质量：accept rate、reward 趋势、方法差异

用法:
    python -m experiments.check_quality
"""
import json
import requests
from experiments.config import SEEDS, METHODS, GENERATORS

BASE_URL = "http://111.170.6.103:10005"
REMOTE_ROOT = "atpc_experiments"
ROUND2_CUTOFF = "20260210_180000"


def list_remote(path):
    try:
        resp = requests.get(f"{BASE_URL}/list", params={"path": path}, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return data["items"] if isinstance(data, dict) and "items" in data else []
        return []
    except:
        return []


def download_json(path):
    """通过服务器下载 JSON 文件 (GET /download?path=xxx)"""
    try:
        resp = requests.get(f"{BASE_URL}/download", params={"path": path}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None


def find_latest_stats(exp_id):
    """找第二轮最新的 training_stats 文件"""
    files = list_remote(f"{REMOTE_ROOT}/{exp_id}")
    if not files:
        return None

    # 找编号最大的 stats 文件 (第二轮编号 >= 40)
    stats_files = []
    for f in files:
        name = f["name"] if isinstance(f, dict) else str(f)
        if name.startswith("training_stats_") and name.endswith(".json"):
            try:
                num = int(name.replace("training_stats_", "").replace(".json", ""))
                if num >= 40:  # 第二轮
                    stats_files.append((num, name))
            except:
                pass

    if not stats_files:
        return None

    stats_files.sort(reverse=True)
    latest = stats_files[0][1]
    return f"{REMOTE_ROOT}/{exp_id}/{latest}"


def analyze_stats(data):
    """分析单个实验的 training_stats"""
    if not data or not isinstance(data, list):
        return None

    # 取有 solver_reward 的记录 (每 10 iter 一条)
    full_records = [r for r in data if "solver_reward" in r]
    if not full_records:
        return None

    # 最后 20% 的记录
    n = len(full_records)
    tail = full_records[int(n * 0.8):]

    # accept rate
    ar_values = [r.get("accept_rate", 0) for r in tail if "accept_rate" in r]
    avg_ar = sum(ar_values) / len(ar_values) if ar_values else 0

    # reward (新字段 step_reward 或 fallback 到 reward)
    rew_key = "step_reward" if "step_reward" in full_records[-1] else "reward"
    rew_values = [r.get(rew_key, 0) for r in tail]
    avg_rew = sum(rew_values) / len(rew_values) if rew_values else 0

    # solver_reward
    sr_values = [r.get("solver_reward", 0) for r in tail]
    avg_sr = sum(sr_values) / len(sr_values) if sr_values else 0

    # 早期 solver_reward (前 20%)
    head = full_records[:max(int(n * 0.2), 1)]
    sr_early = sum(r.get("solver_reward", 0) for r in head) / len(head)

    # regret
    regret_values = [r.get("regret", 0) for r in tail if "regret" in r]
    avg_regret = sum(regret_values) / len(regret_values) if regret_values else 0

    return {
        "total_records": n,
        "avg_accept_rate": avg_ar,
        "avg_reward": avg_rew,
        "avg_solver_reward": avg_sr,
        "early_solver_reward": sr_early,
        "sr_improvement": avg_sr - sr_early,
        "avg_regret": avg_regret,
        "rew_key": rew_key,
    }


def main():
    print(f"服务器: {BASE_URL}")
    print(f"分析第二轮训练质量...")
    print("=" * 90)

    results = {}
    for method in METHODS:
        for gen in GENERATORS:
            for seed in SEEDS:
                exp_id = f"{method}-{gen}-seed{seed}"
                stats_path = find_latest_stats(exp_id)
                if not stats_path:
                    print(f"  {exp_id}: 未找到第二轮 stats")
                    continue

                data = download_json(stats_path)
                if data is None:
                    print(f"  {exp_id}: 下载失败")
                    continue

                info = analyze_stats(data)
                if info is None:
                    print(f"  {exp_id}: 解析失败")
                    continue

                results[exp_id] = info

    if not results:
        print("没有获取到任何数据！")
        print("尝试检查服务器 API...")
        # debug: 看看文件下载路径
        test_files = list_remote(f"{REMOTE_ROOT}/dr-G1-seed42")
        stats = [f for f in test_files if isinstance(f, dict) and "training_stats_7" in f.get("name", "")]
        if stats:
            print(f"  示例文件: {stats[0]}")
            for prefix in ["/download/", "/file/", "/files/"]:
                url = f"{BASE_URL}{prefix}{REMOTE_ROOT}/dr-G1-seed42/{stats[0]['name']}"
                try:
                    r = requests.get(url, timeout=10)
                    print(f"  {url} -> {r.status_code}")
                except Exception as e:
                    print(f"  {url} -> {e}")
        return

    # ============================================================
    # 1. Accept Rate 汇总
    # ============================================================
    print("\n📊 Accept Rate (最后 20% 迭代均值)")
    print("-" * 90)
    print(f"{'实验':<28} {'AR%':>8} {'solver_rew':>12} {'early_sr':>12} {'改善':>10} {'regret':>10}")
    print("-" * 90)

    low_ar = []
    for exp_id in sorted(results.keys()):
        r = results[exp_id]
        ar_pct = r["avg_accept_rate"] * 100 if r["avg_accept_rate"] <= 1 else r["avg_accept_rate"]
        flag = " ⚠️" if ar_pct < 10 and "dr" not in exp_id else ""
        print(f"{exp_id:<28} {ar_pct:>7.1f}% {r['avg_solver_reward']:>12.4f} "
              f"{r['early_solver_reward']:>12.4f} {r['sr_improvement']:>+10.4f} "
              f"{r['avg_regret']:>10.4f}{flag}")
        if ar_pct < 10 and "dr" not in exp_id:
            low_ar.append(exp_id)

    # ============================================================
    # 2. 按方法聚合
    # ============================================================
    print("\n\n📊 按方法 × 生成器聚合 (solver_reward 均值)")
    print("-" * 70)
    print(f"{'方法':<10} {'G1':>12} {'G2':>12} {'G3':>12} {'G4':>12}")
    print("-" * 70)

    for method in METHODS:
        row = f"{method:<10}"
        for gen in GENERATORS:
            vals = []
            for seed in SEEDS:
                exp_id = f"{method}-{gen}-seed{seed}"
                if exp_id in results:
                    vals.append(results[exp_id]["avg_solver_reward"])
            if vals:
                avg = sum(vals) / len(vals)
                row += f" {avg:>12.4f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # ============================================================
    # 3. 问题汇总
    # ============================================================
    print("\n\n📋 问题检查")
    print("-" * 70)

    if low_ar:
        print(f"⚠️  Accept Rate < 10% 的实验 ({len(low_ar)} 个):")
        for e in low_ar:
            r = results[e]
            ar_pct = r["avg_accept_rate"] * 100 if r["avg_accept_rate"] <= 1 else r["avg_accept_rate"]
            print(f"   {e}: AR={ar_pct:.1f}%")
    else:
        print("✅ 所有 PAIRED/AT-PC 实验 Accept Rate >= 10%")

    # 检查 reward 是否全为 0
    zero_rew = [e for e, r in results.items() if r["avg_reward"] == 0 and r["rew_key"] == "reward"]
    if zero_rew:
        print(f"⚠️  reward 字段仍为 0 的实验 ({len(zero_rew)} 个) — 检查是否用了 step_reward")
    else:
        print("✅ reward 记录正常")

    # 检查 solver_reward 是否有改善
    no_improve = [e for e, r in results.items() if r["sr_improvement"] <= 0]
    if no_improve:
        print(f"⚠️  solver_reward 无改善的实验 ({len(no_improve)} 个):")
        for e in no_improve[:5]:
            r = results[e]
            print(f"   {e}: {r['early_solver_reward']:.4f} -> {r['avg_solver_reward']:.4f}")
        if len(no_improve) > 5:
            print(f"   ... 还有 {len(no_improve)-5} 个")
    else:
        print("✅ 所有实验 solver_reward 有改善")


if __name__ == "__main__":
    main()
