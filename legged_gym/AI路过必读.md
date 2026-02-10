# AI 路过必读 - dognew 环境配置

本项目是从 unitree_rl_gym 提取的精简版，专注于 Go2 四足机器狗在普通地面/梅花桩地形上的 AT-PC 对抗课程训练。

运行环境为 WSL2 Ubuntu + IsaacGym + Vulkan。

## 项目路径

```
/home/wsl/dev/opensource/unitree_rl_gym/dognew
```

## 一键环境设置（每次开终端复制粘贴）

```bash
cd /home/wsl/dev/opensource/unitree_rl_gym/dognew/legged_gym
source /home/wsl/dev/toolchains/miniconda3/etc/profile.d/conda.sh
conda activate unitree-rl
export DISPLAY=:0
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/dzn_icd.json
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/wsl/dev/toolchains/miniconda3/envs/unitree-rl/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so.1
ulimit -c unlimited
```

## 环境变量说明

| 变量 | 值 | 作用 |
|------|-----|------|
| `DISPLAY` | `:0` | WSLg 显示，否则 GLFW 窗口创建失败 |
| `VK_ICD_FILENAMES` | `/usr/share/vulkan/icd.d/dzn_icd.json` | WSL2 Vulkan 驱动（kisak-mesa dzn），注意不是 x86_64 版本 |
| `LD_LIBRARY_PATH` | `/usr/lib/wsl/lib:...` | CUDA 和 conda 库路径 |
| `LD_PRELOAD` | `/usr/lib/wsl/lib/libcuda.so.1` | 预加载 libcuda |

## Conda 环境

```
名称: unitree-rl
Python: 3.8
PyTorch: 2.3.1
CUDA: 12.1
```

## 首次安装

如果是第一次使用，需要先安装本项目为 Python 包：

```bash
cd /home/wsl/dev/opensource/unitree_rl_gym/dognew
pip install -e .
```

前置依赖（应该已经装好了）：
- Isaac Gym Preview 4: `cd isaacgym/python && pip install -e .`
- rsl_rl v1.0.2: `git clone https://github.com/leggedrobotics/rsl_rl.git && cd rsl_rl && git checkout v1.0.2 && pip install -e .`

## 已有的 libstdc++ 修复（不要重复操作）

conda 环境里的 libstdc++ 已经软链接到系统版本，解决 Vulkan/GLFW 问题：
```
/home/wsl/dev/toolchains/miniconda3/envs/unitree-rl/lib/libstdc++.so.6 -> /usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

## 支持的训练方法和 Task

| 方法 | Task 名称 | 训练脚本 | 说明 |
|------|-----------|----------|------|
| Standard PPO (Rough) | `go2` | `train.py` | 粗糙地形（trimesh），含课程学习 |
| Standard PPO (Flat) | `go2_flat` | `train_flat_baseline.py` | 纯平地，无随机化，性能下界 |
| Domain Randomization | `go2_dr_baseline` | `train_dr_baseline.py` | DR 基线，随机摩擦/质量/扰动 |
| PAIRED | `go2_paired` | `train_paired.py` | 对抗课程，无新颖性过滤 |
| AT-PC (ReMiDi) | `go2_remidi` | `train_adversarial.py` | 完整对抗框架，带轨迹新颖性检测 |
| ReMiDi | `go2_remidi` | `train_remidi.py` | ReMiDi 多级缓冲版本 |

## 训练命令

所有训练命令都在 `legged_gym/` 目录下执行：

```bash
# 标准 PPO（粗糙地形，含梅花桩/台阶/斜坡）
python scripts/01_training/train.py --task go2 --headless --num_envs 512

# DR 基线
python scripts/01_training/train_dr_baseline.py --task go2_dr_baseline --headless --num_envs 512

# PAIRED（对比方法）
python scripts/01_training/train_paired.py --task go2_paired --headless --num_envs 512

# AT-PC 完整对抗框架（核心方法）
python scripts/01_training/train_adversarial.py --task go2_remidi --headless --num_envs 512

# AT-PC 无新颖性过滤（消融对比）
python scripts/01_training/train_adversarial.py --task go2_remidi --headless --num_envs 512 --use_novelty=False

# 平地基线
python scripts/01_training/train_flat_baseline.py --headless --num_envs 1024
```

显存不够时把 `--num_envs` 降到 256。

## 播放和录屏

```bash
# 带 Viewer 播放（需要设置好 DISPLAY 和 Vulkan）
python scripts/03_recording/play.py --task go2 --num_envs 16

# 录制演示视频（不同物理条件）
python scripts/03_recording/record_demo.py --model=remidi_v2

# 录制地形上的视频
python scripts/03_recording/record_go2_terrain.py
```

## 评估

```bash
python scripts/02_evaluation/evaluate_all.py --dr_run=<run_name> --paired_run=<run_name> --remidi_run=<run_name>
```

## 地形类型说明

`go2` 和 `go2_dr_baseline` 等使用 trimesh 地形，包含：
- 平滑斜坡、粗糙斜坡
- 上楼梯、下楼梯
- 离散障碍（梅花桩）
- 踏脚石、缝隙

地形比例由 `terrain_proportions` 控制，在各 config 文件中定义。

## 训练日志输出

模型和 TensorBoard 日志保存在 `logs/` 目录下，按 task 名称分子目录。

## 硬件信息

- GPU: NVIDIA RTX 4060 Laptop (8GB VRAM)
- 推荐 num_envs: 512 (headless), 16-64 (with viewer)
- 训练 2000 迭代约 28-40 分钟

## 重要提醒

1. Isaac Gym 地形几何在环境创建时固定，运行时只能改物理参数（摩擦、扰动、负载）
2. 训练脚本需要在 `legged_gym/` 目录下运行，因为路径解析依赖工作目录
3. 带 Viewer 运行前必须设置 `DISPLAY=:0`，否则 Segfault
4. `VK_ICD_FILENAMES` 用 `dzn_icd.json` 不是 `dzn_icd.x86_64.json`
