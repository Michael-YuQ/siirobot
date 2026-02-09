#!/bin/bash
# ReMiDi Training Script for Go2
# 改进的 PAIRED 算法，解决遗憾值停滞问题

cd /home/wsl/dev/opensource/unitree_rl_gym/legged_gym
source /home/wsl/dev/toolchains/miniconda3/etc/profile.d/conda.sh
conda activate unitree-rl

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/wsl/dev/toolchains/miniconda3/envs/unitree-rl/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so.1

echo "=========================================="
echo "Starting ReMiDi Training for Go2"
echo "=========================================="

NUM_ENVS=${1:-512}

echo "Number of environments: $NUM_ENVS"
echo ""

python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo ""

python scripts/train_remidi.py \
    --task=go2_remidi \
    --headless \
    --num_envs=$NUM_ENVS \
    --sim_device=cuda:0 \
    --rl_device=cuda:0 \
    --physx
