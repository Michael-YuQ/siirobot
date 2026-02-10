"""
PAIRED Trainer for Legged Robot
实现 PAIRED (Protagonist Antagonist Induced Regret Environment Design) 算法
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional

from .generator import TerrainGenerator, GeneratorInputBuilder
from .regret_buffer import RegretBuffer, MultiLevelBuffer


class PAIREDTrainer:
    """
    PAIRED 训练器
    
    三方博弈:
    - Protagonist (Solver): 待训练的机器人策略
    - Antagonist: 专家策略 (可以是 Solver 的 EMA 版本)
    - Generator: 环境生成器，最大化遗憾值
    """
    
    def __init__(
        self,
        env,
        solver_runner,
        device: str = 'cuda:0',
        generator_lr: float = 1e-4,
        generator_update_freq: int = 10,
        antagonist_ema_decay: float = 0.995,
        use_multi_level_buffer: bool = False,
        log_dir: str = None,
    ):
        self.env = env
        self.solver_runner = solver_runner
        self.device = device
        
        # 生成器
        self.generator = TerrainGenerator(
            hidden_dims=[256, 128],
            num_terrain_types=7,
            use_lstm=True,
            lstm_hidden_size=128
        ).to(device)
        
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=generator_lr
        )
        
        # 生成器输入构建器
        self.input_builder = GeneratorInputBuilder(device=device)
        
        # Antagonist (使用 Solver 的 EMA)
        self.antagonist_ema_decay = antagonist_ema_decay
        self.antagonist_policy = None  # 延迟初始化
        
        # 缓冲区
        if use_multi_level_buffer:
            self.buffer = MultiLevelBuffer(num_levels=5, device=device)
        else:
            self.buffer = RegretBuffer(capacity=1000, device=device)
        
        self.use_multi_level_buffer = use_multi_level_buffer
        
        # 训练参数
        self.generator_update_freq = generator_update_freq
        self.iteration = 0
        
        # 日志
        self.log_dir = log_dir or f"logs/paired_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 统计
        self.stats_history = []
    
    def _init_antagonist(self):
        """初始化 Antagonist (Solver 的副本)"""
        # 深拷贝 Solver 的策略网络
        import copy
        self.antagonist_policy = copy.deepcopy(
            self.solver_runner.alg.actor_critic
        )
        self.antagonist_policy.eval()
    
    def _update_antagonist_ema(self):
        """更新 Antagonist 的 EMA"""
        if self.antagonist_policy is None:
            self._init_antagonist()
            return
        
        solver_params = dict(self.solver_runner.alg.actor_critic.named_parameters())
        
        with torch.no_grad():
            for name, param in self.antagonist_policy.named_parameters():
                if name in solver_params:
                    param.data.mul_(self.antagonist_ema_decay)
                    param.data.add_((1 - self.antagonist_ema_decay) * solver_params[name].data)
    
    def generate_terrain_params(self, batch_size: int = 1) -> Dict:
        """使用生成器生成地形参数"""
        self.generator.eval()
        
        with torch.no_grad():
            # 构建输入
            generator_input = self.input_builder.build(batch_size)
            
            # 生成参数
            terrain_params, log_prob = self.generator(generator_input, deterministic=False)
        
        return terrain_params
    
    def apply_terrain_params(self, terrain_params: Dict):
        """
        将生成的参数应用到环境
        
        注意: 这需要修改环境的配置，具体实现取决于环境接口
        """
        # 提取参数
        terrain_type = terrain_params['terrain_type']
        difficulty = terrain_params['difficulty']
        friction = terrain_params['friction']
        push_magnitude = terrain_params['push_magnitude']
        added_mass = terrain_params['added_mass']
        
        # 转换为 numpy/标量
        if isinstance(terrain_type, torch.Tensor):
            terrain_type = terrain_type.cpu().numpy()
        if isinstance(difficulty, torch.Tensor):
            difficulty = difficulty.cpu().numpy()
        if isinstance(friction, torch.Tensor):
            friction = friction.cpu().numpy()
        if isinstance(push_magnitude, torch.Tensor):
            push_magnitude = push_magnitude.cpu().numpy()
        if isinstance(added_mass, torch.Tensor):
            added_mass = added_mass.cpu().numpy()
        
        # 应用到环境配置
        # 注意: 实际实现需要根据 LeggedRobot 环境的接口调整
        env_cfg = self.env.cfg
        
        # 更新摩擦系数
        if hasattr(env_cfg, 'domain_rand'):
            # 设置摩擦系数范围为生成的值附近
            mean_friction = float(np.mean(friction))
            env_cfg.domain_rand.friction_range = [
                max(0.1, mean_friction - 0.1),
                min(2.0, mean_friction + 0.1)
            ]
            
            # 设置推力
            mean_push = float(np.mean(push_magnitude))
            env_cfg.domain_rand.max_push_vel_xy = mean_push
            
            # 设置附加质量
            mean_mass = float(np.mean(added_mass))
            env_cfg.domain_rand.added_mass_range = [
                mean_mass - 0.5,
                mean_mass + 0.5
            ]
        
        return {
            'terrain_type': terrain_type,
            'difficulty': difficulty,
            'friction': friction,
            'push_magnitude': push_magnitude,
            'added_mass': added_mass,
        }
    
    def compute_regret(
        self,
        solver_rewards: torch.Tensor,
        antagonist_rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        计算遗憾值
        
        Regret = Antagonist_reward - Solver_reward
        """
        return antagonist_rewards - solver_rewards
    
    def evaluate_policy(self, policy, num_steps: int = 100) -> float:
        """评估策略在当前环境中的表现"""
        policy.eval()
        
        obs = self.env.get_observations()
        total_reward = 0
        
        with torch.no_grad():
            for _ in range(num_steps):
                actions = policy.act_inference(obs)
                obs, _, rewards, dones, _ = self.env.step(actions)
                total_reward += rewards.mean().item()
        
        return total_reward / num_steps
    
    def update_generator(self, regret: float, terrain_params: Dict):
        """
        更新生成器
        
        目标: 最大化遗憾值
        """
        self.generator.train()
        
        # 构建输入
        generator_input = self.input_builder.build(batch_size=1)
        
        # 前向传播
        _, log_prob = self.generator(generator_input, deterministic=False)
        
        # 损失: 负的遗憾值 * log_prob (策略梯度)
        # 我们想最大化遗憾值，所以用负号
        loss = -regret * log_prob.mean()
        
        # 添加熵正则化 (鼓励探索)
        entropy = self.generator.get_entropy(generator_input).mean()
        entropy_coef = 0.01
        loss = loss - entropy_coef * entropy
        
        # 反向传播
        self.generator_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.generator_optimizer.step()
        
        return {
            'generator_loss': loss.item(),
            'entropy': entropy.item(),
        }
    
    def train_step(self) -> Dict:
        """
        执行一步 PAIRED 训练
        
        Returns:
            stats: 训练统计信息
        """
        self.iteration += 1
        
        # 1. 生成地形参数
        terrain_params = self.generate_terrain_params(batch_size=1)
        applied_params = self.apply_terrain_params(terrain_params)
        
        # 2. 评估 Solver
        solver_policy = self.solver_runner.alg.actor_critic
        solver_reward = self.evaluate_policy(solver_policy, num_steps=50)
        
        # 3. 评估 Antagonist
        if self.antagonist_policy is None:
            self._init_antagonist()
        antagonist_reward = self.evaluate_policy(self.antagonist_policy, num_steps=50)
        
        # 4. 计算遗憾值
        regret = antagonist_reward - solver_reward
        
        # 5. 更新输入构建器
        difficulty = float(applied_params['difficulty'].mean()) if isinstance(
            applied_params['difficulty'], np.ndarray
        ) else float(applied_params['difficulty'])
        
        self.input_builder.update(solver_reward, antagonist_reward, difficulty)
        
        # 6. 添加到缓冲区
        if self.use_multi_level_buffer:
            self.buffer.add(
                terrain_params=applied_params,
                solver_reward=solver_reward,
                antagonist_reward=antagonist_reward,
                iteration=self.iteration
            )
            # 检查是否晋升
            if self.buffer.check_promotion():
                self.buffer.promote()
        else:
            self.buffer.add(
                terrain_params=applied_params,
                solver_reward=solver_reward,
                antagonist_reward=antagonist_reward,
                iteration=self.iteration
            )
        
        # 7. 更新生成器
        generator_stats = {}
        if self.iteration % self.generator_update_freq == 0:
            generator_stats = self.update_generator(regret, terrain_params)
        
        # 8. 更新 Antagonist EMA
        self._update_antagonist_ema()
        
        # 统计信息
        # 构建统计信息
        buffer_stats = self.buffer.get_statistics() if hasattr(self.buffer, 'get_statistics') else {}
        
        stats = {
            'iteration': self.iteration,
            'solver_reward': solver_reward,
            'antagonist_reward': antagonist_reward,
            'regret': regret,
            'difficulty': difficulty,
            'terrain_type': int(applied_params['terrain_type'][0]) if isinstance(
                applied_params['terrain_type'], np.ndarray
            ) else int(applied_params['terrain_type']),
        }
        stats.update(generator_stats)
        stats.update(buffer_stats)
        
        self.stats_history.append(stats)
        
        return stats
    
    def save(self, path: str = None):
        """保存模型"""
        if path is None:
            path = os.path.join(self.log_dir, f'paired_checkpoint_{self.iteration}.pt')
        
        torch.save({
            'iteration': self.iteration,
            'generator_state_dict': self.generator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'antagonist_state_dict': self.antagonist_policy.state_dict() if self.antagonist_policy else None,
            'stats_history': self.stats_history,
        }, path)
        
        print(f"[PAIRED] Saved checkpoint to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.iteration = checkpoint['iteration']
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        
        if checkpoint['antagonist_state_dict'] is not None:
            if self.antagonist_policy is None:
                self._init_antagonist()
            self.antagonist_policy.load_state_dict(checkpoint['antagonist_state_dict'])
        
        self.stats_history = checkpoint.get('stats_history', [])
        
        print(f"[PAIRED] Loaded checkpoint from {path}")
