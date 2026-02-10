"""
Adversarial Curriculum Training Script
真正的对抗训练：生成器 vs Solver 的博弈

使用方法:
    python scripts/train_adversarial.py --task=go2_remidi --headless --num_envs=512

对比实验:
    --use_novelty=True   # ReMiDi (带轨迹新颖性过滤)
    --use_novelty=False  # PAIRED (无过滤)
"""
import os
import sys
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Isaac Gym 必须先导入
import isaacgym

import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict

from curriculum.adversarial_trainer import AdversarialCurriculumTrainer


class AdversarialRunner:
    """
    对抗训练 Runner
    
    训练循环:
    1. 生成器生成环境参数
    2. 应用参数到仿真环境
    3. Solver (PPO) 在该环境中训练若干步
    4. 计算遗憾值，更新生成器
    5. 重复
    """
    
    def __init__(
        self,
        env,
        train_cfg,
        log_dir: str,
        device: str = 'cuda:0',
        use_novelty: bool = True,
    ):
        from rsl_rl.runners import OnPolicyRunner
        
        self.env = env
        self.device = device
        self.log_dir = log_dir
        self.use_novelty = use_novelty
        
        # 创建 PPO Runner
        train_cfg_dict = class_to_dict(train_cfg)
        self.ppo_runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=device)
        
        # 创建对抗训练器
        self.adversarial_trainer = AdversarialCurriculumTrainer(
            env=env,
            ppo_runner=self.ppo_runner,
            device=device,
            generator_lr=3e-4,
            novelty_threshold=0.7,
            use_novelty_filter=use_novelty,
            log_dir=log_dir,
            # === 概率下界参数 ===
            easy_terrain_prob=0.15,      # 初始简单地形概率
            easy_terrain_decay=0.995,    # 衰减率
            min_easy_prob=0.05,          # 最小概率（永不为0）
            # === 可行性过滤参数 ===
            use_feasibility_filter=True,
            max_infeasible_attempts=5,
            # === 热身协议参数 ===
            warmup_iterations=50,        # 热身 50 次课程更新
        )
        
        # 训练参数
        self.curriculum_update_freq = 10  # 每10次PPO迭代更新一次课程
        self.ppo_steps_per_curriculum = 5  # 每次课程更新后PPO训练5次
        
        print(f"\n[Adversarial Runner]")
        print(f"  Use Novelty Filter (ReMiDi): {use_novelty}")
        print(f"  Curriculum Update Freq: {self.curriculum_update_freq}")
        print(f"  Log Dir: {log_dir}")
    
    def learn(self, num_iterations: int, init_at_random_ep_len: bool = True):
        """
        对抗训练主循环
        """
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
        
        start_time = time.time()
        current_terrain = None
        
        for iteration in range(num_iterations):
            
            # ============================================
            # 1. 课程更新 (生成器生成新环境)
            # ============================================
            if iteration % self.curriculum_update_freq == 0:
                # 生成新的环境参数并应用（包含热身、概率下界、可行性过滤）
                current_terrain, is_easy, is_warmup, source = self.adversarial_trainer.generate_and_apply_curriculum()
                
                # 计算遗憾值并更新生成器
                stats = self.adversarial_trainer.compute_regret_and_update(
                    current_terrain, is_easy, is_warmup, source
                )
                
                if iteration % 50 == 0:
                    print(f"\n{'='*70}")
                    print(f"[Curriculum Update] Iteration {iteration}")
                    print(f"  Source: {source}")
                    print(f"  Terrain: type={stats.terrain_type}, diff={stats.difficulty:.2f}, "
                          f"friction={stats.friction:.2f}")
                    if not is_warmup:
                        print(f"  Regret: {stats.regret:.4f} (Ant={stats.antagonist_reward:.3f}, "
                              f"Sol={stats.solver_reward:.3f})")
                        print(f"  Novel: {stats.is_novel}, Accept Rate: {stats.accept_rate:.1%}")
                    else:
                        print(f"  [Warmup Phase] Reward: {stats.solver_reward:.3f}")
                    print(f"  Easy Terrain: prob={stats.easy_terrain_prob:.1%}, count={stats.easy_terrain_count}")
                    print(f"  Infeasible Count: {stats.infeasible_count}")
                    print(f"  Generator Loss: {stats.generator_loss:.4f}")
                    print(f"{'='*70}")
            
            # ============================================
            # 2. PPO 训练 (Solver 学习适应当前环境)
            # ============================================
            with torch.inference_mode():
                for step in range(self.ppo_runner.num_steps_per_env):
                    actions = self.ppo_runner.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
                    self.ppo_runner.alg.process_env_step(rewards, dones, infos)
                
                self.ppo_runner.alg.compute_returns(critic_obs)
            
            # PPO 更新
            mean_value_loss, mean_surrogate_loss = self.ppo_runner.alg.update()
            
            # ============================================
            # 3. 日志记录
            # ============================================
            if iteration % 20 == 0:
                elapsed = time.time() - start_time
                eta = (num_iterations - iteration) * (elapsed / max(iteration, 1))
                
                # 获取奖励信息
                ep_info = self.env.extras.get('episode', {})
                mean_reward = ep_info.get('rew_tracking_lin_vel', 0)
                
                print(f"[{iteration}/{num_iterations}] "
                      f"Reward: {mean_reward:.3f}, "
                      f"VLoss: {mean_value_loss:.4f}, "
                      f"SLoss: {mean_surrogate_loss:.4f}, "
                      f"Time: {elapsed:.0f}s, ETA: {eta/60:.1f}min")
            
            # ============================================
            # 4. 保存检查点
            # ============================================
            if iteration % 200 == 0 and iteration > 0:
                self.ppo_runner.save(os.path.join(self.log_dir, f'model_{iteration}.pt'))
                self.adversarial_trainer.save()
        
        # 最终保存
        self.ppo_runner.save(os.path.join(self.log_dir, 'model_final.pt'))
        self.adversarial_trainer.save(os.path.join(self.log_dir, 'adversarial_final.pt'))
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"[Training Complete]")
        print(f"  Total Time: {total_time/60:.1f} min")
        print(f"  Final Accept Rate: {self.adversarial_trainer.accepted / max(self.adversarial_trainer.accepted + self.adversarial_trainer.rejected, 1):.1%}")
        print(f"  Model saved to: {self.log_dir}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='go2_remidi')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--num_envs', type=int, default=512)
    parser.add_argument('--max_iterations', type=int, default=2000)
    parser.add_argument('--use_novelty', type=str, default='true', 
                        help='Use novelty filter (ReMiDi). Set to "false" for PAIRED baseline.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=1)
    
    args, unknown = parser.parse_known_args()
    
    # 解析 use_novelty
    use_novelty = args.use_novelty.lower() in ['true', '1', 'yes']
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    print("=" * 70)
    print("Adversarial Curriculum Training")
    print("=" * 70)
    print(f"  Task: {args.task}")
    print(f"  Num Envs: {args.num_envs}")
    print(f"  Max Iterations: {args.max_iterations}")
    print(f"  Use Novelty (ReMiDi): {use_novelty}")
    print(f"  Device: {args.device}")
    print("=" * 70)
    
    # 使用 legged_gym 的标准参数解析
    from isaacgym import gymapi
    
    class LeggedArgs:
        task = args.task
        headless = args.headless
        num_envs = args.num_envs
        sim_device = args.device
        rl_device = args.device
        physics_engine = gymapi.SIM_PHYSX
        use_gpu = True
        subscenes = 0
        num_threads = 0
        use_gpu_pipeline = True
        seed = args.seed
        resume = False
        experiment_name = None
        run_name = None
        load_run = None
        checkpoint = None
        max_iterations = args.max_iterations
    
    lg_args = LeggedArgs()
    
    # 创建环境
    env, env_cfg = task_registry.make_env(name=args.task, args=lg_args)
    
    # 获取训练配置
    _, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg.runner.max_iterations = args.max_iterations
    
    # 创建日志目录
    method_name = 'remidi_v2' if use_novelty else 'paired_v2'
    log_dir = os.path.join(
        'logs', f'go2_{method_name}',
        datetime.now().strftime('%b%d_%H-%M-%S') + f'_{method_name}'
    )
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建 Runner
    runner = AdversarialRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=args.device,
        use_novelty=use_novelty,
    )
    
    # 开始训练
    runner.learn(
        num_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == '__main__':
    main()
