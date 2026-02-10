"""
ReMiDi (Refining Minimax Regret) Implementation
ReMiDi 算法核心实现：解决 PAIRED 的遗憾值停滞问题

核心创新:
1. 多级缓冲区 (Iterative Buffers): 维护已掌握环境的历史
2. 轨迹重叠拒绝 (Trajectory Overlap Rejection): 过滤无意义的高遗憾环境
3. 贝叶斯关卡完美极大极小遗憾 (BLP): 更精确的遗憾值估计
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import copy

from .generator import TerrainGenerator, GeneratorInputBuilder
from .regret_buffer import MultiLevelBuffer, EnvironmentRecord


@dataclass
class TrajectoryRecord:
    """轨迹记录，用于新颖性检测"""
    observations: torch.Tensor  # [T, obs_dim]
    actions: torch.Tensor       # [T, action_dim]
    rewards: torch.Tensor       # [T]
    terrain_params: Dict
    
    def compute_hash(self) -> int:
        """计算轨迹哈希，用于快速比较"""
        # 使用观测和动作的统计特征
        obs_mean = self.observations.mean(dim=0)
        obs_std = self.observations.std(dim=0)
        act_mean = self.actions.mean(dim=0)
        
        # 组合成哈希
        features = torch.cat([obs_mean, obs_std, act_mean])
        return hash(tuple(features.cpu().numpy().round(2).tolist()))


class TrajectoryNoveltyChecker:
    """
    轨迹新颖性检测器
    
    核心思想: 如果在新环境中执行的轨迹与历史缓冲区中的轨迹无法区分，
    则该环境不提供新的学习信号，应该被拒绝。
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        trajectory_length: int = 50,
        device: str = 'cuda:0'
    ):
        self.similarity_threshold = similarity_threshold
        self.trajectory_length = trajectory_length
        self.device = device
        
        # 历史轨迹缓存
        self.trajectory_cache: List[TrajectoryRecord] = []
        self.max_cache_size = 500
    
    def compute_trajectory_similarity(
        self,
        traj1: TrajectoryRecord,
        traj2: TrajectoryRecord
    ) -> float:
        """
        计算两条轨迹的相似度
        
        使用 DTW (Dynamic Time Warping) 或简化的欧氏距离
        """
        # 简化版本: 使用观测序列的欧氏距离
        min_len = min(len(traj1.observations), len(traj2.observations))
        
        obs1 = traj1.observations[:min_len]
        obs2 = traj2.observations[:min_len]
        
        # 归一化
        obs1_norm = (obs1 - obs1.mean()) / (obs1.std() + 1e-8)
        obs2_norm = (obs2 - obs2.mean()) / (obs2.std() + 1e-8)
        
        # 计算相关系数
        correlation = torch.corrcoef(
            torch.stack([obs1_norm.flatten(), obs2_norm.flatten()])
        )[0, 1]
        
        # 转换为相似度 [0, 1]
        similarity = (correlation + 1) / 2
        
        return similarity.item() if not torch.isnan(similarity) else 0.0
    
    def is_novel(self, new_trajectory: TrajectoryRecord) -> Tuple[bool, float]:
        """
        检查新轨迹是否具有新颖性
        
        Returns:
            is_novel: 是否新颖
            max_similarity: 与历史轨迹的最大相似度
        """
        if len(self.trajectory_cache) == 0:
            return True, 0.0
        
        max_similarity = 0.0
        
        for cached_traj in self.trajectory_cache:
            similarity = self.compute_trajectory_similarity(new_trajectory, cached_traj)
            max_similarity = max(max_similarity, similarity)
            
            # 早停: 如果已经找到高度相似的轨迹
            if similarity > self.similarity_threshold:
                return False, similarity
        
        return max_similarity < self.similarity_threshold, max_similarity
    
    def add_trajectory(self, trajectory: TrajectoryRecord):
        """添加轨迹到缓存"""
        self.trajectory_cache.append(trajectory)
        
        # 维护缓存大小
        if len(self.trajectory_cache) > self.max_cache_size:
            # 移除最旧的轨迹
            self.trajectory_cache.pop(0)
    
    def clear(self):
        """清空缓存"""
        self.trajectory_cache = []


class ReMiDiTrainer:
    """
    ReMiDi 训练器
    
    在 PAIRED 基础上添加:
    1. 多级缓冲区管理
    2. 轨迹重叠拒绝机制
    3. 掩码梯度更新
    """
    
    def __init__(
        self,
        env,
        solver_runner,
        device: str = 'cuda:0',
        generator_lr: float = 1e-4,
        num_buffer_levels: int = 5,
        novelty_threshold: float = 0.8,
        promotion_threshold: float = 0.8,
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
        
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=generator_lr
        )
        
        # 输入构建器
        self.input_builder = GeneratorInputBuilder(device=device)
        
        # 多级缓冲区 (ReMiDi 核心)
        self.multi_level_buffer = MultiLevelBuffer(
            num_levels=num_buffer_levels,
            capacity_per_level=200,
            device=device
        )
        self.multi_level_buffer.promotion_threshold = promotion_threshold
        
        # 轨迹新颖性检测器 (ReMiDi 核心)
        self.novelty_checker = TrajectoryNoveltyChecker(
            similarity_threshold=novelty_threshold,
            device=device
        )
        
        # Antagonist (EMA of Solver)
        self.antagonist_policy = None
        self.antagonist_ema_decay = 0.995
        
        # 统计
        self.iteration = 0
        self.rejected_count = 0
        self.accepted_count = 0
        self.stats_history = []
        
        self.log_dir = log_dir
    
    def _init_antagonist(self):
        """初始化 Antagonist"""
        try:
            self.antagonist_policy = copy.deepcopy(
                self.solver_runner.alg.actor_critic
            )
            self.antagonist_policy.eval()
        except Exception as e:
            print(f"[ReMiDi] Warning: Could not deepcopy antagonist: {e}")
            self.antagonist_policy = self.solver_runner.alg.actor_critic
    
    def _update_antagonist_ema(self):
        """更新 Antagonist EMA"""
        if self.antagonist_policy is None:
            self._init_antagonist()
            return
        
        try:
            solver_params = dict(self.solver_runner.alg.actor_critic.named_parameters())
            
            with torch.no_grad():
                for name, param in self.antagonist_policy.named_parameters():
                    if name in solver_params:
                        param.data.mul_(self.antagonist_ema_decay)
                        param.data.add_((1 - self.antagonist_ema_decay) * solver_params[name].data)
        except Exception:
            pass
    
    def collect_trajectory(
        self,
        policy,
        terrain_params: Dict,
        num_steps: int = 50
    ) -> Tuple[TrajectoryRecord, float]:
        """
        收集策略在指定环境中的轨迹
        
        Returns:
            trajectory: 轨迹记录
            mean_reward: 平均奖励
        """
        policy.eval()
        
        observations = []
        actions = []
        rewards = []
        
        obs = self.env.get_observations()
        
        with torch.no_grad():
            for _ in range(num_steps):
                action = policy.act_inference(obs)
                obs, _, reward, done, _ = self.env.step(action)
                
                observations.append(obs.mean(dim=0))  # 平均所有环境
                actions.append(action.mean(dim=0))
                rewards.append(reward.mean())
        
        trajectory = TrajectoryRecord(
            observations=torch.stack(observations),
            actions=torch.stack(actions),
            rewards=torch.stack(rewards),
            terrain_params=terrain_params
        )
        
        mean_reward = torch.stack(rewards).mean().item()
        
        return trajectory, mean_reward
    
    def train_step(self) -> Dict:
        """
        执行一步 ReMiDi 训练
        
        核心流程:
        1. 生成候选环境
        2. 收集 Solver 和 Antagonist 轨迹
        3. 检查轨迹新颖性
        4. 如果新颖，计算遗憾值并更新生成器
        5. 如果不新颖，拒绝该环境
        """
        self.iteration += 1
        
        # 1. 生成候选环境参数
        self.generator.eval()
        with torch.no_grad():
            generator_input = self.input_builder.build(batch_size=1)
            terrain_params, log_prob = self.generator(generator_input, deterministic=False)
        
        # 转换参数
        applied_params = self._apply_terrain_params(terrain_params)
        
        # 2. 收集 Solver 轨迹
        solver_policy = self.solver_runner.alg.actor_critic
        solver_traj, solver_reward = self.collect_trajectory(
            solver_policy, applied_params, num_steps=50
        )
        
        # 3. 收集 Antagonist 轨迹
        if self.antagonist_policy is None:
            self._init_antagonist()
        
        antagonist_traj, antagonist_reward = self.collect_trajectory(
            self.antagonist_policy, applied_params, num_steps=50
        )
        
        # 4. 检查轨迹新颖性 (ReMiDi 核心)
        is_novel, similarity = self.novelty_checker.is_novel(solver_traj)
        
        # 5. 计算遗憾值
        regret = antagonist_reward - solver_reward
        
        # 6. 根据新颖性决定是否更新
        generator_stats = {}
        
        if is_novel:
            self.accepted_count += 1
            
            # 添加到缓冲区
            current_level = self.multi_level_buffer.current_level
            self.multi_level_buffer.add(
                terrain_params=applied_params,
                solver_reward=solver_reward,
                antagonist_reward=antagonist_reward,
                level=current_level,
                iteration=self.iteration
            )
            
            # 添加轨迹到新颖性检测器
            self.novelty_checker.add_trajectory(solver_traj)
            
            # 更新生成器 (仅对新颖环境)
            generator_stats = self._update_generator(regret, terrain_params, log_prob)
            
            # 检查是否晋升到下一级别
            if self.multi_level_buffer.check_promotion():
                self.multi_level_buffer.promote()
                print(f"[ReMiDi] Promoted to level {self.multi_level_buffer.current_level}")
        else:
            self.rejected_count += 1
            # 不更新生成器 - 这是 ReMiDi 的关键：拒绝无意义的高遗憾环境
        
        # 更新 Antagonist
        self._update_antagonist_ema()
        
        # 更新输入构建器
        difficulty = float(np.mean(applied_params['difficulty'])) if isinstance(
            applied_params['difficulty'], np.ndarray
        ) else float(applied_params['difficulty'])
        
        self.input_builder.update(solver_reward, antagonist_reward, difficulty)
        
        # 统计
        stats = {
            'iteration': self.iteration,
            'solver_reward': solver_reward,
            'antagonist_reward': antagonist_reward,
            'regret': regret,
            'is_novel': is_novel,
            'similarity': similarity,
            'accepted_count': self.accepted_count,
            'rejected_count': self.rejected_count,
            'acceptance_rate': self.accepted_count / (self.accepted_count + self.rejected_count),
            'current_level': self.multi_level_buffer.current_level,
            'difficulty': difficulty,
            'terrain_type': int(applied_params['terrain_type'][0]) if isinstance(
                applied_params['terrain_type'], np.ndarray
            ) else int(applied_params['terrain_type']),
        }
        stats.update(generator_stats)
        
        self.stats_history.append(stats)
        
        return stats
    
    def _apply_terrain_params(self, terrain_params: Dict) -> Dict:
        """应用地形参数到环境"""
        # 转换 tensor 到 numpy
        applied = {}
        for key, value in terrain_params.items():
            if isinstance(value, torch.Tensor):
                applied[key] = value.cpu().numpy()
            else:
                applied[key] = value
        
        # 更新环境配置
        env_cfg = self.env.cfg
        if hasattr(env_cfg, 'domain_rand'):
            mean_friction = float(np.mean(applied['friction']))
            env_cfg.domain_rand.friction_range = [
                max(0.1, mean_friction - 0.1),
                min(2.0, mean_friction + 0.1)
            ]
            
            mean_push = float(np.mean(applied['push_magnitude']))
            env_cfg.domain_rand.max_push_vel_xy = mean_push
        
        return applied
    
    def _update_generator(
        self,
        regret: float,
        terrain_params: Dict,
        log_prob: torch.Tensor
    ) -> Dict:
        """更新生成器"""
        self.generator.train()
        
        # 策略梯度损失
        loss = -regret * log_prob.mean()
        
        # 熵正则化
        generator_input = self.input_builder.build(batch_size=1)
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
    
    def save(self, path: str = None):
        """保存检查点"""
        import os
        if path is None:
            path = os.path.join(self.log_dir, f'remidi_checkpoint_{self.iteration}.pt')
        
        torch.save({
            'iteration': self.iteration,
            'generator_state_dict': self.generator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'accepted_count': self.accepted_count,
            'rejected_count': self.rejected_count,
            'current_level': self.multi_level_buffer.current_level,
            'stats_history': self.stats_history,
        }, path)
        
        print(f"[ReMiDi] Saved checkpoint to {path}")
    
    def load(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.iteration = checkpoint['iteration']
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.accepted_count = checkpoint.get('accepted_count', 0)
        self.rejected_count = checkpoint.get('rejected_count', 0)
        self.multi_level_buffer.current_level = checkpoint.get('current_level', 0)
        self.stats_history = checkpoint.get('stats_history', [])
        
        print(f"[ReMiDi] Loaded checkpoint from {path}")
