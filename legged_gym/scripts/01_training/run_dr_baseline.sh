#!/bin/bash
# DR Baseline Training Script for Go2
# 用于 ReMiDi 研究的基线实验

cd /home/wsl/dev/opensource/unitree_rl_gym/legged_gym
source /home/wsl/dev/toolchains/miniconda3/etc/profile.d/conda.sh
conda activate unitree-rl

# 设置库路径 - WSL2 CUDA 支持
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/wsl/dev/toolchains/miniconda3/envs/unitree-rl/lib:$LD_LIBRARY_PATH

# 预加载 libcuda
export LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so.1

echo "=========================================="
echo "Starting DR Baseline Training for Go2"
echo "=========================================="

# 默认使用 512 envs (RTX 4060 Laptop 8GB 显存安全值)
NUM_ENVS=${1:-512}

echo "Number of environments: $NUM_ENVS"
echo ""

# 检查 CUDA 是否可用
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo ""

python scripts/train_dr_baseline.py \
    --task=go2_dr_baseline \
    --headless \
    --num_envs=$NUM_ENVS \
    --sim_device=cuda:0 \
    --rl_device=cuda:0 \
    --physx
