"""
检查网盘服务器上的实验完成情况（区分第一轮/第二轮）

第一轮: FINAL 文件时间戳 <= 20260210_12xxxx
第二轮: FINAL 文件时间戳 >= 20260210_19xxxx

用法:
    python -m experiments.check_progress
"""
import re
import requests
from experiments.config import SEEDS, METHODS, GENERATORS

BASE_URL = "http://111.170.6.103:10005"
REMOTE_ROOT = "atpc_experiments"

# 第二轮开始时间 (大约 2026-02-10 19:00)
ROUND2_CUTOFF = "20260210_180000"


def list_remote(path):
    try:
        resp = requests.get(f"{BASE_URL}/list", params={"path": path}, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "items" in data:
                return data["items"]
            return data if isinstance(data, list) else []
        return []
    except Exception as e:
        print(f"[Error] 无法连接服务器: {e}")
        return None


def parse_final_timestamp(filename):
    """从 FINAL 文件名提取时间戳, 如 xxx_FINAL_20260210_193233.tar.gz -> 20260210_193233"""
    m = re.search(r'_FINAL_(\d{8}_\d{6})\.tar\.gz', filename)
    return m.group(1) if m else None


def get_latest_stats_num(names):
    """找最大的 training_stats_N.json 编号"""
    nums = []
    for n in names:
        m = re.match(r'training_stats_(\d+)\.json', n)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) if nums else -1


def get_max_checkpoint_iter(names):
    """找最大的 checkpoint iter"""
    iters = []
    for n in names:
        m = re.search(r'_iter(\d+)_', n)
        if m:
            iters.append(int(m.group(1)))
    return max(iters) if iters else 0


def main():
    all_experiments = []
    for method in METHODS:
        for gen in GENERATORS:
            for seed in SEEDS:
                all_experiments.append(f"{method}-{gen}-seed{seed}")

    print(f"服务器: {BASE_URL}")
    print(f"远程目录: {REMOTE_ROOT}")
    print(f"预期实验数: {len(all_experiments)}")
    print(f"第二轮判定: FINAL 时间戳 >= {ROUND2_CUTOFF}")
    print("=" * 70)

    r1_complete = []  # 第一轮完成
    r2_complete = []  # 第二轮完成
    r2_progress = []  # 第二轮进行中
    r2_not_started = []  # 第二轮未开始

    for exp_id in all_experiments:
        files = list_remote(f"{REMOTE_ROOT}/{exp_id}")
        if files is None:
            print("服务器连接失败，退出")
            return

        names = [f["name"] if isinstance(f, dict) else str(f) for f in files]

        # 找所有 FINAL 文件及其时间戳
        finals = {}
        for n in names:
            ts = parse_final_timestamp(n)
            if ts:
                finals[ts] = n

        r1_finals = {ts: n for ts, n in finals.items() if ts < ROUND2_CUTOFF}
        r2_finals = {ts: n for ts, n in finals.items() if ts >= ROUND2_CUTOFF}

        # 找第二轮的 checkpoint (时间戳 >= cutoff)
        r2_checkpoints = []
        for n in names:
            m = re.search(r'_iter(\d+)_(\d{8}_\d{6})\.tar\.gz', n)
            if m and m.group(2) >= ROUND2_CUTOFF:
                r2_checkpoints.append((int(m.group(1)), n))

        # 找第二轮的 training_stats (编号 >= 40, 因为第一轮是 0-39)
        r2_stats = []
        for n in names:
            m = re.match(r'training_stats_(\d+)\.json', n)
            if m and int(m.group(1)) >= 40:
                r2_stats.append(int(m.group(1)))

        has_r1 = len(r1_finals) > 0

        if r2_finals:
            r2_complete.append(exp_id)
        elif r2_checkpoints or r2_stats:
            max_iter = max([cp[0] for cp in r2_checkpoints]) if r2_checkpoints else 0
            max_stat = max(r2_stats) if r2_stats else 0
            # 每个 stats 文件约覆盖 50 iter, 所以 stat_num * ~25 ≈ iter
            est_iter = max(max_iter, max_stat * 25) if max_stat > 40 else max_iter
            r2_progress.append((exp_id, max_iter, max_stat, has_r1))
        elif has_r1:
            r2_not_started.append(exp_id)
        else:
            r2_not_started.append(exp_id)

    # 输出
    print(f"\n✅ 第二轮已完成 ({len(r2_complete)}/{len(all_experiments)}):")
    for e in sorted(r2_complete):
        print(f"   {e}")

    if r2_progress:
        print(f"\n⏳ 第二轮进行中 ({len(r2_progress)}):")
        for e, max_cp, max_stat, has_r1 in sorted(r2_progress):
            r1_tag = " [有第一轮数据]" if has_r1 else ""
            print(f"   {e}  最新checkpoint=iter{max_cp}  stats#{max_stat}{r1_tag}")

    if r2_not_started:
        print(f"\n❌ 第二轮未开始 ({len(r2_not_started)}):")
        for e in sorted(r2_not_started):
            print(f"   {e}")

    total_done = len(r2_complete)
    total_wip = len(r2_progress)
    total_todo = len(r2_not_started)
    print(f"\n{'='*70}")
    print(f"第二轮总计: {total_done} 完成 / {total_wip} 进行中 / {total_todo} 未开始")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
