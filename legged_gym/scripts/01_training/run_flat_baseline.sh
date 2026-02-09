#!/bin/bash
# Standard PPO (Flat) Baseline Training Script

# 设置环境
source /home/wsl/dev/toolchains/miniconda3/etc/profile.d/conda.sh
conda activate unitree-rl

# 设置库路径
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/wsl/dev/toolchains/miniconda3/envs/unitree-rl/lib
export LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so.1

# 进入项目目录
cd /home/wsl/dev/opensource/unitree_rl_gym/legged_gym

# 运行训练
echo "=========================================="
echo "Standard PPO (Flat) Baseline Training"
echo "=========================================="
echo "Terrain: FLAT (plane only)"
echo "Domain Randomization: DISABLED"
echo "External Push: DISABLED"
echo "=========================================="

python scripts/train_flat_baseline.py \
    --task go2_flat \
    --headless \
    --num_envs 1024 \
    --max_iterations 2000

echo "Training complete!"
