#!/usr/bin/env python3
"""
Standard PPO (Flat) Baseline Training Script
仅在平坦地面训练，作为性能下界
"""

import os
import sys
import argparse

# 确保 Isaac Gym 在 torch 之前导入
from isaacgym import gymapi
from isaacgym import gymutil

import torch
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legged_gym.envs import *
from legged_gym.utils import task_registry, Logger
from legged_gym import LEGGED_GYM_ROOT_DIR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='go2_flat')
    parser.add_argument('--headless', action='store_true', default=True)
    parser.add_argument('--num_envs', type=int, default=1024)
    parser.add_argument('--max_iterations', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--load_run', type=str, default=None)
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    
    # Isaac Gym 参数
    parser.add_argument('--sim_device', type=str, default='cuda:0')
    parser.add_argument('--rl_device', type=str, default='cuda:0')
    parser.add_argument('--graphics_device_id', type=int, default=0)
    parser.add_argument('--compute_device_id', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--subscenes', type=int, default=0)
    
    args = parser.parse_args()
    
    # 设置 physics_engine 为 gymapi 枚举值
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = True
    args.use_gpu_pipeline = True
    args.device = 'cuda:0'
    
    return args


def train_flat_baseline():
    """训练 Standard PPO (Flat) 基线"""
    
    args = get_args()
    
    # 获取环境和配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 覆盖配置
    env_cfg.env.num_envs = args.num_envs
    train_cfg.runner.max_iterations = args.max_iterations
    
    # 设置运行名称
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    train_cfg.runner.run_name = f"flat_baseline_{timestamp}"
    
    # 创建环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 创建训练器
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, 
        name=args.task, 
        args=args, 
        train_cfg=train_cfg
    )
    
    print("=" * 60)
    print("Standard PPO (Flat) Baseline Training")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Num Envs: {env_cfg.env.num_envs}")
    print(f"Max Iterations: {train_cfg.runner.max_iterations}")
    print(f"Terrain: FLAT (plane)")
    print(f"Domain Randomization: DISABLED")
    print(f"External Push: DISABLED")
    print("=" * 60)
    
    # 开始训练
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: logs/{train_cfg.runner.experiment_name}/")
    print("=" * 60)


if __name__ == "__main__":
    train_flat_baseline()
