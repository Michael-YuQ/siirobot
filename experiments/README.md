# AT-PC 跨地形生成器验证实验

3 方法 (DR / PAIRED / AT-PC) × 4 生成器 (G1-7D / G2-15D / G3-21D / G4-18D) × 3 种子 = 36 runs

## Quick Start (Colab)

```bash
git clone https://github.com/YOUR_USERNAME/dognew.git
cd dognew
python experiments/colab_setup.py

# 单个实验
python -m experiments.run_experiment \
    --generator G1 --method atpc --seed 42 \
    --upload_url http://YOUR_SERVER:8080/upload

# 全部 36 个
python -m experiments.run_all \
    --upload_url http://YOUR_SERVER:8080/upload

# 评估
python -m experiments.evaluate_experiment \
    --log_dir logs/experiments/atpc-G1-seed42_XXXX \
    --generator G1 --method atpc

# 分析
python -m experiments.analyze_results
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `config.py` | 实验常量、超参数、测试地形 |
| `generators.py` | G1-G4 四种地形生成器 + 统一 GAN 网络 |
| `uploader.py` | HTTP/SCP 结果上传 |
| `run_experiment.py` | 单个实验入口 (method × generator × seed) |
| `run_all.py` | 批量启动 36 个实验 |
| `evaluate_experiment.py` | 在 10 种测试地形上评估 |
| `analyze_results.py` | 聚合结果、对比表、训练曲线、假设检验 |
| `colab_setup.py` | Colab 环境一键配置 |

## 上传方式

支持 HTTP POST 和 SCP 两种:

```bash
# HTTP
--upload_url http://your-server:8080/upload

# SCP
--scp_target user@server:/path/to/results/
```

每 500 iterations 上传一次 checkpoint (~5MB)，训练结束上传完整结果 (~80MB/run)。
