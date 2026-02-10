"""
Domain Randomization Baseline Training Script
用于 ReMiDi 研究的基线实验

使用方法:
    conda activate unitree-rl
    python scripts/train_dr_baseline.py --headless --num_envs=1024

显存不足时可降低 num_envs:
    python scripts/train_dr_baseline.py --headless --num_envs=512
"""
import os
import sys
import argparse
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


def train_dr_baseline(args):
    """训练 Domain Randomization 基线"""
    
    print("=" * 60)
    print("Domain Randomization Baseline Training")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Num Envs: {args.num_envs if hasattr(args, 'num_envs') else 'default'}")
    print(f"Device: {args.rl_device}")
    print(f"Headless: {args.headless}")
    print("=" * 60)
    
    # 创建环境
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # 打印关键配置确认
    print("\n[Config Verification]")
    print(f"  - randomize_friction: {env_cfg.domain_rand.randomize_friction}")
    print(f"  - friction_range: {env_cfg.domain_rand.friction_range}")
    print(f"  - randomize_base_mass: {env_cfg.domain_rand.randomize_base_mass}")
    print(f"  - added_mass_range: {env_cfg.domain_rand.added_mass_range}")
    print(f"  - push_robots: {env_cfg.domain_rand.push_robots}")
    print(f"  - terrain.curriculum: {env_cfg.terrain.curriculum}")
    print(f"  - terrain.mesh_type: {env_cfg.terrain.mesh_type}")
    print()
    
    # 创建训练器
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, 
        name=args.task, 
        args=args
    )
    
    print(f"[Training] Starting for {train_cfg.runner.max_iterations} iterations...")
    print(f"[Logging] Experiment: {train_cfg.runner.experiment_name}")
    print(f"[Logging] Run: {train_cfg.runner.run_name}")
    
    # 开始训练
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations, 
        init_at_random_ep_len=True
    )
    
    print("\n[Training Complete]")


if __name__ == '__main__':
    # 解析参数，默认使用 go2_dr_baseline 任务
    args = get_args()
    
    # 如果没有指定 task，默认使用 dr_baseline
    if args.task == 'go2':
        print("[Info] Switching to go2_dr_baseline task for baseline experiment")
        args.task = 'go2_dr_baseline'
    
    train_dr_baseline(args)
