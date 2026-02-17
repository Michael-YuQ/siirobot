#!/bin/bash
# AT-PC 第五轮重跑脚本 — 在训练服务器上执行
# 前四轮 run_experiment.py 从未更新成功，这次确保用最新代码
# 
# 用法: 在训练服务器 terminal 里直接粘贴运行:
#   bash experiments/restart_all.sh

set -e

echo "=========================================="
echo "  AT-PC 第五轮完整重跑 (代码修复版)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 1. 环境配置
echo ""
echo "[1/5] 配置环境..."
eval "$(/root/miniconda/bin/conda shell.bash hook)"
conda activate rl
export LD_LIBRARY_PATH=/root/miniconda/envs/rl/lib/python3.8/site-packages/torch/lib:/root/miniconda/envs/rl/lib:
unset PYTHONPATH
cd /root/siirobot
echo "  ✅ 环境就绪"

# 2. 拉最新代码
echo ""
echo "[2/5] 拉取最新代码..."
git fetch origin
git reset --hard origin/main
echo ""
echo "当前 commit:"
git log --oneline -1
echo ""

# 3. 验证关键修复 (这些是之前四轮都没生效的修复)
echo "[3/5] 验证代码修复..."
FAIL=0

# run_experiment.py 的修复
if grep -q "step_reward" experiments/run_experiment.py; then
    echo "  ✅ step_reward 字段存在"
else
    echo "  ❌ step_reward 缺失！"
    FAIL=1
fi

if grep -q "env\.rew_buf\.mean" experiments/run_experiment.py; then
    echo "  ✅ env.rew_buf 直接读取"
else
    echo "  ❌ env.rew_buf 读取缺失！"
    FAIL=1
fi

if grep -q 'it % 10 == 0' experiments/run_experiment.py; then
    echo "  ✅ DR 每10iter评估"
else
    echo "  ⚠️  DR 评估频率可能不对"
fi

# config.py 的修复
if grep -q "warmup_iterations.*500" experiments/config.py; then
    echo "  ✅ warmup=500"
else
    echo "  ❌ warmup 未更新！"
    FAIL=1
fi

if grep -q "curriculum_update_freq.*50" experiments/config.py; then
    echo "  ✅ curriculum_freq=50"
else
    echo "  ❌ curriculum_freq 未更新！"
    FAIL=1
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

if [ $FAIL -ne 0 ]; then
    echo ""
    echo "  ❌❌❌ 关键修复缺失，中止！请检查 git pull 是否成功 ❌❌❌"
    exit 1
fi
echo ""

# 4. 清旧 logs
echo "[4/5] 清除旧训练日志..."
rm -rf logs/experiments/
mkdir -p logs/experiments
echo "  ✅ logs/experiments/ 已清除"
echo ""

# 5. 开跑
echo "[5/5] 启动全部 36 个实验..."
echo "预计耗时: ~12-15 小时 (36 runs × ~20 min)"
echo "=========================================="
echo ""

python -m experiments.run_all --headless
