#!/bin/bash
# PAIRED Training Script for Go2
# 用于 ReMiDi 研究的 PAIRED 基线实验

cd /home/wsl/dev/opensource/unitree_rl_gym/legged_gym
source /home/wsl/dev/toolchains/miniconda3/etc/profile.d/conda.sh
conda activate unitree-rl

# 设置库路径
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/wsl/dev/toolchains/miniconda3/envs/unitree-rl/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so.1

echo "=========================================="
echo "Starting PAIRED Training for Go2"
echo "=========================================="

NUM_ENVS=${1:-512}

echo "Number of environments: $NUM_ENVS"
echo ""

python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo ""

python scripts/train_paired.py \
    --task=go2_paired \
    --headless \
    --num_envs=$NUM_ENVS \
    --sim_device=cuda:0 \
    --rl_device=cuda:0 \
    --physx
