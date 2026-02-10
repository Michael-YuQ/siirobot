"""
PAIRED Training Script for Go2
训练 PAIRED 算法 - 简化版本，直接使用 OnPolicyRunner

使用方法:
    python scripts/train_paired.py --headless --num_envs=512
"""
import os
import sys
import time
import numpy as np
from datetime import datetime

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

from curriculum.paired_trainer import PAIREDTrainer


class PAIREDOnPolicyRunner:
    """
    集成 PAIRED 的 OnPolicyRunner
    在标准 PPO 训练基础上添加 PAIRED 课程生成
    """
    
    def __init__(self, env, train_cfg, log_dir, device='cuda:0'):
        from rsl_rl.runners import OnPolicyRunner
        from legged_gym.utils.helpers import class_to_dict
        
        self.env = env
        self.device = device
        self.log_dir = log_dir
        
        # 创建标准 PPO runner
        train_cfg_dict = class_to_dict(train_cfg)
        self.ppo_runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=device)
        
        # 创建 PAIRED trainer
        self.paired_trainer = PAIREDTrainer(
            env=env,
            solver_runner=self.ppo_runner,
            device=device,
            generator_lr=1e-4,
            generator_update_freq=10,
            antagonist_ema_decay=0.995,
            use_multi_level_buffer=False,
            log_dir=log_dir,
        )
        
        self.current_learning_iteration = 0
        self.paired_update_freq = 20  # 每 20 次 PPO 迭代更新一次 PAIRED
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """
        训练循环
        """
        # 初始化
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, 
                high=int(self.env.max_episode_length)
            )
        
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        
        self.ppo_runner.alg.actor_critic.train()
        
        ep_infos = []
        rewbuffer = []
        lenbuffer = []
        
        start_time = time.time()
        
        for iteration in range(num_learning_iterations):
            self.current_learning_iteration = iteration
            
            # PAIRED 更新
            paired_stats = {}
            if iteration % self.paired_update_freq == 0 and iteration > 0:
                try:
                    paired_stats = self.paired_trainer.train_step()
                except Exception as e:
                    print(f"[PAIRED] Warning: {e}")
            
            # 收集 rollout
            with torch.inference_mode():
                for step in range(self.ppo_runner.num_steps_per_env):
                    actions = self.ppo_runner.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
                    self.ppo_runner.alg.process_env_step(rewards, dones, infos)
                    
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                
                # 计算 returns
                self.ppo_runner.alg.compute_returns(critic_obs)
            
            # PPO 更新
            mean_value_loss, mean_surrogate_loss = self.ppo_runner.alg.update()
            
            # 统计
            ep_infos = []
            
            # 记录 PAIRED 统计 (跳过 tensorboard)
            # if paired_stats:
            #     self.ppo_runner.writer.add_scalar(...)
            
            # 打印进度
            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                eta = (num_learning_iterations - iteration) * (elapsed / max(iteration, 1))
                
                mean_reward = self.env.extras.get('episode', {}).get('rew_tracking_lin_vel', 0)
                mean_ep_len = self.env.extras.get('episode', {}).get('mean_episode_length', 0)
                
                print(f"{'#' * 80}")
                print(f" Learning iteration {iteration}/{num_learning_iterations}")
                print(f"  Value loss: {mean_value_loss:.4f}, Surrogate loss: {mean_surrogate_loss:.4f}")
                print(f"  Mean reward: {mean_reward:.4f}")
                
                if paired_stats:
                    print(f"  PAIRED regret: {paired_stats.get('regret', 0):.4f}, "
                          f"terrain: {paired_stats.get('terrain_type', 0)}, "
                          f"difficulty: {paired_stats.get('difficulty', 0):.2f}")
                
                print(f"  Time: {elapsed:.1f}s, ETA: {eta/60:.1f}min")
                print(f"{'-' * 80}")
            
            # 保存
            if iteration % 100 == 0:
                self.ppo_runner.save(os.path.join(self.log_dir, f'model_{iteration}.pt'))
                self.paired_trainer.save()
        
        # 最终保存
        self.ppo_runner.save(os.path.join(self.log_dir, 'model_final.pt'))
        self.paired_trainer.save()
        
        print(f"\n[Training Complete] Total time: {(time.time() - start_time)/60:.1f} min")


def train_paired(args):
    """训练 PAIRED"""
    
    print("=" * 60)
    print("PAIRED Training for Go2")
    print("=" * 60)
    
    # 创建环境
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # 获取训练配置
    _, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 创建日志目录
    log_dir = os.path.join(
        'logs', train_cfg.runner.experiment_name,
        datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name
    )
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n[Config]")
    print(f"  Task: {args.task}")
    print(f"  Num Envs: {env.num_envs}")
    print(f"  Max Iterations: {train_cfg.runner.max_iterations}")
    print(f"  Log Dir: {log_dir}")
    print()
    
    # 创建 PAIRED runner
    runner = PAIREDOnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=args.rl_device
    )
    
    # 开始训练
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )


if __name__ == '__main__':
    args = get_args()
    
    # 默认使用 PAIRED 任务
    if args.task == 'go2':
        args.task = 'go2_paired'
    
    train_paired(args)
