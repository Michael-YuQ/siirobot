"""
从网盘服务器下载所有实验结果到本地

用法:
    python download_results.py
    python download_results.py --output_dir ./downloaded_results
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

SERVER_BASE = "http://111.170.6.103:10005"


def list_remote_dir(path):
    """列出远程目录内容"""
    import urllib.request
    url = f"{SERVER_BASE}/list?path={path}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  Error listing {path}: {e}")
        return None


def download_file(remote_path, local_path):
    """下载单个文件"""
    url = f"{SERVER_BASE}/download?path={remote_path}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    cmd = ["wget", "-q", url, "-O", local_path]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="下载实验结果")
    parser.add_argument("--output_dir", type=str, default="./downloaded_results")
    parser.add_argument("--experiments_only", action="store_true",
                        help="只下载 FINAL 包和最新的 training_stats")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 列出所有实验
    print("Listing experiments on server...")
    root = list_remote_dir("atpc_experiments")
    if not root or "items" not in root:
        print("Failed to list experiments")
        return

    experiments = [item for item in root["items"] if item["is_dir"]]
    print(f"Found {len(experiments)} experiments")

    for exp in experiments:
        exp_name = exp["name"]
        exp_path = exp["path"]
        print(f"\n{exp_name}:")

        # 列出实验内容
        contents = list_remote_dir(exp_path)
        if not contents or "items" not in contents:
            print(f"  Failed to list contents")
            continue

        files = contents["items"]
        
        # 找到要下载的文件
        to_download = []
        
        # 1. FINAL 包（最新的）
        finals = [f for f in files if "FINAL" in f["name"] and f["name"].endswith(".tar.gz")]
        if finals:
            finals.sort(key=lambda x: x["modified"], reverse=True)
            to_download.append(finals[0])
        
        # 2. 最新的 training_stats
        stats = [f for f in files if f["name"].startswith("training_stats") and f["name"].endswith(".json")]
        if stats:
            # 找最大编号的
            def get_num(name):
                if name == "training_stats.json":
                    return 0
                try:
                    return int(name.replace("training_stats_", "").replace(".json", ""))
                except:
                    return -1
            stats.sort(key=lambda x: get_num(x["name"]), reverse=True)
            to_download.append(stats[0])
        
        if not args.experiments_only:
            # 下载所有 checkpoint
            checkpoints = [f for f in files if f["name"].startswith("atpc-") or 
                          f["name"].startswith("dr-") or f["name"].startswith("paired-")]
            to_download.extend(checkpoints)

        # 下载
        for f in to_download:
            local_path = output_dir / exp_name / f["name"]
            if local_path.exists():
                print(f"  SKIP (exists): {f['name']}")
                continue
            print(f"  Downloading: {f['name']} ({f['size']/1024/1024:.1f}MB)...", end=" ", flush=True)
            if download_file(f["path"], str(local_path)):
                print("OK")
            else:
                print("FAILED")

    print(f"\nDownload complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()
