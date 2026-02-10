"""
Colab 一键设置脚本 — 在 Colab 中运行此脚本完成环境配置

用法 (在 Colab cell 中):
    !git clone https://github.com/YOUR_USERNAME/dognew.git
    %cd dognew
    !python experiments/colab_setup.py
    # 然后运行实验:
    !python -m experiments.run_experiment --generator G1 --method atpc --seed 42
"""
import os
import sys
import subprocess


def run(cmd, check=True):
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=check)


def main():
    print("=" * 60)
    print("  AT-PC Experiment — Colab Setup")
    print("=" * 60)

    # 1. 安装 IsaacGym
    print("\n[1/4] Checking IsaacGym...")
    try:
        import isaacgym
        print("  IsaacGym already installed.")
    except ImportError:
        print("  IsaacGym not found. Please upload and install manually:")
        print("  1. Upload isaacgym.tar.gz to Colab")
        print("  2. Run: !tar xf isaacgym.tar.gz")
        print("  3. Run: !pip install -e isaacgym/python")
        print("  Then re-run this script.")
        sys.exit(1)

    # 2. 安装 rsl_rl
    print("\n[2/4] Installing rsl_rl...")
    try:
        import rsl_rl
        print("  rsl_rl already installed.")
    except ImportError:
        run("pip install git+https://github.com/leggedrobotics/rsl_rl.git")

    # 3. 安装本项目
    print("\n[3/4] Installing dognew...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run(f"pip install -e {project_root}")

    # 4. 安装额外依赖
    print("\n[4/4] Installing extra dependencies...")
    run("pip install requests matplotlib scipy", check=False)

    # 验证
    print("\n" + "=" * 60)
    print("  Verification")
    print("=" * 60)
    try:
        import isaacgym
        import rsl_rl
        import torch
        print(f"  isaacgym: OK")
        print(f"  rsl_rl: OK")
        print(f"  torch: {torch.__version__}")
        print(f"  CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print("\n  Setup complete! You can now run experiments.")
    except Exception as e:
        print(f"  Verification failed: {e}")

    # 打印使用说明
    print("\n" + "=" * 60)
    print("  Quick Start")
    print("=" * 60)
    print("""
  # 运行单个实验:
  !python -m experiments.run_experiment \\
      --generator G1 --method atpc --seed 42 \\
      --upload_url http://YOUR_SERVER:8080/upload

  # 运行全部 36 个实验:
  !python -m experiments.run_all \\
      --upload_url http://YOUR_SERVER:8080/upload

  # 只跑 AT-PC vs DR 对比:
  !python -m experiments.run_all \\
      --methods atpc dr --generators G1 G2 --seeds 42

  # 评估:
  !python -m experiments.evaluate_experiment \\
      --log_dir logs/experiments/atpc-G1-seed42_XXXX \\
      --generator G1 --method atpc

  # 分析结果:
  !python -m experiments.analyze_results
""")


if __name__ == "__main__":
    main()
