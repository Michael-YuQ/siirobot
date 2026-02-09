# dognew - Go2 四足机器人 AT-PC 训练框架

从 unitree_rl_gym 提取的精简版，专注于 Go2 机器狗在普通地面/梅花桩地形上的训练。

## 支持的训练方法

| 方法 | Task 名称 | 训练脚本 | 说明 |
|------|-----------|----------|------|
| Standard PPO (Flat) | `go2_flat` | `train_flat_baseline.py` | 平地基线 |
| Domain Randomization | `go2_dr_baseline` | `train_dr_baseline.py` | DR 基线 |
| PAIRED | `go2_paired` | `train_paired.py` | 对抗课程（无新颖性过滤） |
| ReMiDi | `go2_remidi` | `train_remidi.py` | 带轨迹新颖性检测 |
| AT-PC | `go2_remidi` | `train_adversarial.py` | 完整对抗框架 |
| Standard PPO (Rough) | `go2` | `train.py` | 粗糙地形标准训练 |

## 安装

```bash
conda activate unitree-rl
cd dognew
pip install -e .
```

## 训练

```bash
# 进入 legged_gym 目录
cd legged_gym

# 1. 标准 PPO 训练（粗糙地形）
python scripts/01_training/train.py --task go2 --headless --num_envs=512

# 2. DR 基线
python scripts/01_training/train_dr_baseline.py --task go2_dr_baseline --headless --num_envs=512

# 3. PAIRED
python scripts/01_training/train_paired.py --task go2_paired --headless --num_envs=512

# 4. AT-PC（完整对抗框架）
python scripts/01_training/train_adversarial.py --task go2_remidi --headless --num_envs=512

# 5. 平地基线
python scripts/01_training/train_flat_baseline.py --headless --num_envs=1024
```

## 播放/录屏

```bash
# 带 Viewer 播放（需要 WSL Vulkan 配置）
export DISPLAY=:0
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/dzn_icd.json
python scripts/03_recording/play.py --task go2 --num_envs 16

# 录制视频
python scripts/03_recording/record_demo.py --model=remidi_v2
python scripts/03_recording/record_go2_terrain.py
```

## 评估

```bash
python scripts/02_evaluation/evaluate_all.py --dr_run=<run> --paired_run=<run> --remidi_run=<run>
```

## 目录结构

```
dognew/
├── setup.py
├── play_go2_viewer.py
├── legged_gym/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py          # 注册 go2 任务
│   │   ├── base/                # 基础环境类
│   │   └── go2/                 # Go2 配置（flat/dr/paired/remidi）
│   ├── curriculum/
│   │   ├── generator.py         # 地形生成器网络
│   │   ├── regret_buffer.py     # 遗憾值缓冲区
│   │   ├── paired_trainer.py    # PAIRED 训练器
│   │   ├── remidi.py            # ReMiDi 轨迹新颖性
│   │   └── adversarial_trainer.py  # AT-PC 核心
│   ├── utils/                   # 工具函数
│   └── scripts/
│       ├── 01_training/         # 训练脚本
│       ├── 02_evaluation/       # 评估脚本
│       └── 03_recording/        # 录屏脚本
├── resources/robots/go2/        # Go2 URDF 和 mesh
└── logs/                        # 训练日志输出
```
