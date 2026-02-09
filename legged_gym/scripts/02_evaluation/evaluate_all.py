"""
综合评估脚本
对比 DR Baseline, PAIRED, ReMiDi 三种方法的性能

使用方法:
    python scripts/evaluate_all.py --dr_run=<run> --paired_run=<run> --remidi_run=<run>
"""
import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


class PolicyEvaluator:
    """策略评估器"""
    
    def __init__(self, env, device='cuda:0'):
        self.env = env
        self.device = device
        self.num_envs = env.num_envs
    
    def evaluate(self, policy, num_episodes=100, max_steps=1000):
        """评估策略"""
        policy.eval()
        
        total_rewards = []
        episode_lengths = []
        success_count = 0
        fall_count = 0
        distances = []
        
        obs = self.env.get_observations()
        
        episode_reward = torch.zeros(self.num_envs, device=self.device)
        episode_length = torch.zeros(self.num_envs, device=self.device)
        start_pos = self.env.root_states[:, :2].clone()
        
        completed = 0
        step = 0
        
        while completed < num_episodes and step < max_steps * 10:
            with torch.no_grad():
                actions = policy.act_inference(obs)
            
            obs, _, rewards, dones, infos = self.env.step(actions)
            
            episode_reward += rewards
            episode_length += 1
            step += 1
            
            done_indices = dones.nonzero(as_tuple=False).flatten()
            
            for idx in done_indices:
                idx = idx.item()
                
                total_rewards.append(episode_reward[idx].item())
                episode_lengths.append(episode_length[idx].item())
                
                end_pos = self.env.root_states[idx, :2]
                distance = torch.norm(end_pos - start_pos[idx]).item()
                distances.append(distance)
                
                if 'time_outs' in infos and infos['time_outs'][idx]:
                    success_count += 1
                else:
                    fall_count += 1
                
                episode_reward[idx] = 0
                episode_length[idx] = 0
                start_pos[idx] = self.env.root_states[idx, :2].clone()
                
                completed += 1
                if completed >= num_episodes:
                    break
        
        return {
            'num_episodes': completed,
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'success_rate': success_count / completed if completed > 0 else 0,
            'fall_rate': fall_count / completed if completed > 0 else 0,
            'mean_distance': np.mean(distances),
            'max_distance': np.max(distances) if distances else 0,
        }


def load_policy(task_name, run_name, checkpoint=-1, device='cuda:0'):
    """加载训练好的策略"""
    from legged_gym.utils.helpers import get_load_path
    
    log_root = os.path.join('logs', task_name.replace('go2_', 'go2_'))
    
    # 查找实验目录
    if task_name == 'go2_dr_baseline':
        log_root = 'logs/go2_dr_baseline'
    elif task_name == 'go2_paired':
        log_root = 'logs/go2_paired'
    elif task_name == 'go2_remidi':
        log_root = 'logs/go2_remidi'
    
    resume_path = get_load_path(log_root, load_run=run_name, checkpoint=checkpoint)
    
    print(f"Loading from: {resume_path}")
    
    # 创建环境和 runner
    args = get_args()
    args.task = task_name
    args.headless = True
    args.num_envs = 256
    
    env, env_cfg = task_registry.make_env(name=task_name, args=args)
    
    ppo_runner, _ = task_registry.make_alg_runner(
        env=env,
        name=task_name,
        args=args,
        log_root=None
    )
    
    # 加载权重
    ppo_runner.load(resume_path)
    
    return env, ppo_runner.alg.actor_critic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dr_run', type=str, default=None, help='DR baseline run name')
    parser.add_argument('--paired_run', type=str, default=None, help='PAIRED run name')
    parser.add_argument('--remidi_run', type=str, default=None, help='ReMiDi run name')
    parser.add_argument('--num_episodes', type=int, default=100)
    args = parser.parse_args()
    
    results = {}
    
    # 评估 DR Baseline
    if args.dr_run:
        print("\n" + "=" * 60)
        print("Evaluating DR Baseline")
        print("=" * 60)
        
        env, policy = load_policy('go2_dr_baseline', args.dr_run)
        evaluator = PolicyEvaluator(env)
        results['DR Baseline'] = evaluator.evaluate(policy, args.num_episodes)
    
    # 评估 PAIRED
    if args.paired_run:
        print("\n" + "=" * 60)
        print("Evaluating PAIRED")
        print("=" * 60)
        
        env, policy = load_policy('go2_paired', args.paired_run)
        evaluator = PolicyEvaluator(env)
        results['PAIRED'] = evaluator.evaluate(policy, args.num_episodes)
    
    # 评估 ReMiDi
    if args.remidi_run:
        print("\n" + "=" * 60)
        print("Evaluating ReMiDi")
        print("=" * 60)
        
        env, policy = load_policy('go2_remidi', args.remidi_run)
        evaluator = PolicyEvaluator(env)
        results['ReMiDi'] = evaluator.evaluate(policy, args.num_episodes)
    
    # 打印对比结果
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"\n{'Method':<15} {'Mean Reward':>12} {'Success Rate':>14} {'Mean Distance':>14} {'Episode Len':>12}")
    print("-" * 70)
    
    for method, res in results.items():
        print(f"{method:<15} {res['mean_reward']:>12.2f} {res['success_rate']*100:>13.1f}% "
              f"{res['mean_distance']:>14.2f} {res['mean_episode_length']:>12.1f}")
    
    # 保存结果
    save_path = f"results/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(save_path, 'w') as f:
        f.write("Method Comparison Results\n")
        f.write("=" * 60 + "\n\n")
        for method, res in results.items():
            f.write(f"{method}:\n")
            for key, value in res.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"\nResults saved to: {save_path}")


if __name__ == '__main__':
    main()
