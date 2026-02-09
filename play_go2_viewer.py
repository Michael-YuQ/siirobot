"""
在 WSL 上运行带 Viewer 的 Go2 IsaacGym 环境

使用方法:
    python play_go2_viewer.py --task go2

WSL Vulkan 配置要求:
1. 确保 WSL2 已更新到最新版本
2. 安装 Windows 的 GPU 驱动 (支持 WSLg)
3. 设置环境变量 (如果需要):
   export DISPLAY=:0
   export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
"""

import isaacgym  # 必须在其他 gym 导入之前
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import torch


def play_with_viewer(args):
    # 获取配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 减少环境数量以便更好地可视化
    env_cfg.env.num_envs = 16
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.curriculum = False
    
    # 关闭噪声和随机化以便更清晰地观察
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # 创建环境 (headless=False 会启用 viewer)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # 加载训练好的模型 (如果有)
    try:
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg
        )
        policy = ppo_runner.get_inference_policy(device=env.device)
        print("成功加载训练好的模型!")
        use_policy = True
    except Exception as e:
        print(f"未找到训练好的模型，使用随机动作: {e}")
        use_policy = False
    
    # 主循环
    print("\n=== Viewer 控制 ===")
    print("V: 切换相机视角")
    print("W/A/S/D: 移动相机")
    print("鼠标拖拽: 旋转视角")
    print("ESC: 退出")
    print("==================\n")
    
    step = 0
    while True:
        step += 1
        
        if use_policy:
            actions = policy(obs.detach())
        else:
            # 随机动作
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        # 每 100 步打印一次信息
        if step % 100 == 0:
            print(f"Step: {step}, Mean Reward: {rews.mean().item():.3f}")


if __name__ == '__main__':
    args = get_args()
    play_with_viewer(args)
