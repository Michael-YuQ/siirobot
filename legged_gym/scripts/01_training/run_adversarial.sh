#!/bin/bash
# Adversarial Curriculum Training
# 真正的对抗训练框架

# 激活环境
source /home/wsl/dev/toolchains/miniconda3/etc/profile.d/conda.sh
conda activate unitree-rl

# 设置 CUDA 和 Python 库路径
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/wsl/dev/toolchains/miniconda3/envs/unitree-rl/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so.1

cd "$(dirname "$0")/.."

echo "=============================================="
echo "Adversarial Curriculum Training"
echo "=============================================="

# 选择实验类型
EXPERIMENT=${1:-"remidi"}  # remidi 或 paired

if [ "$EXPERIMENT" == "remidi" ]; then
    echo "Running ReMiDi (with novelty filter)"
    python scripts/train_adversarial.py \
        --task=go2_remidi \
        --headless \
        --num_envs=512 \
        --max_iterations=2000 \
        --use_novelty=true
elif [ "$EXPERIMENT" == "paired" ]; then
    echo "Running PAIRED (without novelty filter)"
    python scripts/train_adversarial.py \
        --task=go2_paired \
        --headless \
        --num_envs=512 \
        --max_iterations=2000 \
        --use_novelty=false
else
    echo "Unknown experiment: $EXPERIMENT"
    echo "Usage: ./run_adversarial.sh [remidi|paired]"
    exit 1
fi

echo "=============================================="
echo "Training Complete!"
echo "=============================================="
