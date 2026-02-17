#!/bin/bash
# AT-PC 第三轮重跑脚本 — 在训练服务器上执行
# 确保用最新代码，清掉旧 logs，全部重跑 36 个实验

set -e

echo "=========================================="
echo "  AT-PC 第三轮完整重跑"
echo "=========================================="

# 1. 环境配置
eval "$(/root/miniconda/bin/conda shell.bash hook)"
conda activate rl
export LD_LIBRARY_PATH=/root/miniconda/envs/rl/lib/python3.8/site-packages/torch/lib:/root/miniconda/envs/rl/lib:
unset PYTHONPATH
cd /root/siirobot

# 2. 拉最新代码
echo ""
echo "[1/4] 拉取最新代码..."
git pull
echo ""
echo "最新 commit:"
git log --oneline -1
echo ""

# 3. 验证关键修复
echo "[2/4] 验证代码修复..."
if grep -q "step_reward" experiments/run_experiment.py; then
    echo "  ✅ step_reward 字段存在"
else
    echo "  ❌ step_reward 缺失，代码未更新！"
    exit 1
fi

if grep -q "curriculum_update_freq.*50" experiments/config.py; then
    echo "  ✅ curriculum_freq=50"
else
    echo "  ⚠️  curriculum_freq 可能未更新"
fi

if grep -q "warmup_iterations.*500" experiments/config.py; then
    echo "  ✅ warmup=500"
else
    echo "  ⚠️  warmup 可能未更新"
fi

if grep -q "novelty_threshold.*0.95" experiments/config.py; then
    echo "  ✅ novelty_threshold=0.95"
else
    echo "  ⚠️  novelty_threshold 可能未更新"
fi

if grep -q "min_easy_prob.*0.25" experiments/config.py; then
    echo "  ✅ min_easy_prob=0.25"
else
    echo "  ⚠️  min_easy_prob 可能未更新"
fi
echo ""

# 4. 清旧 logs
echo "[3/4] 清除旧训练日志..."
rm -rf logs/experiments/
echo "  ✅ logs/experiments/ 已清除"
echo ""

# 5. 开跑
echo "[4/4] 启动全部 36 个实验..."
echo "预计耗时: ~12 小时"
echo "=========================================="
echo ""

python -m experiments.run_all --headless
