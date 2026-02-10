"""
Regret Buffer for PAIRED/ReMiDi
遗憾值缓冲区：存储和管理训练过程中的环境参数和性能数据
"""
import torch
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class EnvironmentRecord:
    """环境记录"""
    terrain_params: Dict  # 地形参数
    solver_reward: float  # Solver 获得的奖励
    antagonist_reward: float  # Antagonist 获得的奖励
    regret: float  # 遗憾值 = antagonist - solver
    trajectory_hash: Optional[int] = None  # 轨迹哈希 (用于 ReMiDi)
    iteration: int = 0  # 记录时的迭代次数


class RegretBuffer:
    """
    遗憾值缓冲区
    
    功能:
    1. 存储高遗憾值环境
    2. 支持优先级采样
    3. 提供统计信息用于生成器训练
    """
    
    def __init__(
        self,
        capacity: int = 1000,
        priority_alpha: float = 0.6,
        device: str = 'cuda:0'
    ):
        self.capacity = capacity
        self.priority_alpha = priority_alpha
        self.device = device
        
        self.buffer: List[EnvironmentRecord] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(
        self,
        terrain_params: Dict,
        solver_reward: float,
        antagonist_reward: float,
        iteration: int = 0,
        trajectory_hash: Optional[int] = None
    ):
        """添加新记录"""
        regret = antagonist_reward - solver_reward
        
        record = EnvironmentRecord(
            terrain_params=terrain_params,
            solver_reward=solver_reward,
            antagonist_reward=antagonist_reward,
            regret=regret,
            trajectory_hash=trajectory_hash,
            iteration=iteration
        )
        
        # 计算优先级 (基于遗憾值)
        priority = (abs(regret) + 1e-6) ** self.priority_alpha
        
        if self.size < self.capacity:
            self.buffer.append(record)
            self.priorities[self.size] = priority
            self.size += 1
        else:
            # 替换最旧的记录
            self.buffer[self.position] = record
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[EnvironmentRecord]:
        """优先级采样"""
        if self.size == 0:
            return []
        
        batch_size = min(batch_size, self.size)
        
        # 计算采样概率
        probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
        
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)
        
        return [self.buffer[i] for i in indices]
    
    def sample_high_regret(self, batch_size: int, threshold_percentile: float = 75) -> List[EnvironmentRecord]:
        """采样高遗憾值环境"""
        if self.size == 0:
            return []
        
        regrets = np.array([r.regret for r in self.buffer[:self.size]])
        threshold = np.percentile(regrets, threshold_percentile)
        
        high_regret_indices = np.where(regrets >= threshold)[0]
        
        if len(high_regret_indices) == 0:
            return self.sample(batch_size)
        
        batch_size = min(batch_size, len(high_regret_indices))
        indices = np.random.choice(high_regret_indices, size=batch_size, replace=False)
        
        return [self.buffer[i] for i in indices]
    
    def get_statistics(self) -> Dict:
        """获取缓冲区统计信息"""
        if self.size == 0:
            return {
                'size': 0,
                'mean_regret': 0,
                'max_regret': 0,
                'min_regret': 0,
                'std_regret': 0,
                'mean_solver_reward': 0,
                'mean_antagonist_reward': 0,
            }
        
        regrets = np.array([r.regret for r in self.buffer[:self.size]])
        solver_rewards = np.array([r.solver_reward for r in self.buffer[:self.size]])
        antagonist_rewards = np.array([r.antagonist_reward for r in self.buffer[:self.size]])
        
        return {
            'size': self.size,
            'mean_regret': float(np.mean(regrets)),
            'max_regret': float(np.max(regrets)),
            'min_regret': float(np.min(regrets)),
            'std_regret': float(np.std(regrets)),
            'mean_solver_reward': float(np.mean(solver_rewards)),
            'mean_antagonist_reward': float(np.mean(antagonist_rewards)),
        }
    
    def get_terrain_distribution(self) -> Dict:
        """获取地形类型分布"""
        if self.size == 0:
            return {}
        
        terrain_counts = {}
        for record in self.buffer[:self.size]:
            t_type = record.terrain_params.get('terrain_type', 0)
            if isinstance(t_type, torch.Tensor):
                t_type = t_type.item()
            terrain_counts[t_type] = terrain_counts.get(t_type, 0) + 1
        
        total = sum(terrain_counts.values())
        return {k: v / total for k, v in terrain_counts.items()}
    
    def clear(self):
        """清空缓冲区"""
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0


class MultiLevelBuffer:
    """
    多级缓冲区 (用于 ReMiDi)
    
    维护多个难度级别的环境缓冲区，支持课程演进
    """
    
    def __init__(
        self,
        num_levels: int = 5,
        capacity_per_level: int = 200,
        device: str = 'cuda:0'
    ):
        self.num_levels = num_levels
        self.capacity_per_level = capacity_per_level
        self.device = device
        
        # 每个级别一个缓冲区
        self.level_buffers: List[RegretBuffer] = [
            RegretBuffer(capacity=capacity_per_level, device=device)
            for _ in range(num_levels)
        ]
        
        # 当前活跃级别
        self.current_level = 0
        
        # 级别晋升阈值
        self.promotion_threshold = 0.8  # solver 奖励达到 antagonist 的 80% 时晋升
    
    def add(
        self,
        terrain_params: Dict,
        solver_reward: float,
        antagonist_reward: float,
        level: Optional[int] = None,
        iteration: int = 0
    ):
        """添加到指定级别的缓冲区"""
        if level is None:
            level = self.current_level
        
        level = min(level, self.num_levels - 1)
        
        self.level_buffers[level].add(
            terrain_params=terrain_params,
            solver_reward=solver_reward,
            antagonist_reward=antagonist_reward,
            iteration=iteration
        )
    
    def check_promotion(self) -> bool:
        """检查是否应该晋升到下一级别"""
        if self.current_level >= self.num_levels - 1:
            return False
        
        current_buffer = self.level_buffers[self.current_level]
        if current_buffer.size < 10:
            return False
        
        stats = current_buffer.get_statistics()
        
        # 如果 solver 接近 antagonist 的表现，晋升
        if stats['mean_antagonist_reward'] > 0:
            ratio = stats['mean_solver_reward'] / stats['mean_antagonist_reward']
            if ratio >= self.promotion_threshold:
                return True
        
        return False
    
    def promote(self):
        """晋升到下一级别"""
        if self.current_level < self.num_levels - 1:
            self.current_level += 1
            print(f"[MultiLevelBuffer] Promoted to level {self.current_level}")
    
    def sample_current_level(self, batch_size: int) -> List[EnvironmentRecord]:
        """从当前级别采样"""
        return self.level_buffers[self.current_level].sample(batch_size)
    
    def sample_all_levels(self, batch_size: int) -> List[EnvironmentRecord]:
        """从所有级别采样 (用于防止遗忘)"""
        samples = []
        samples_per_level = max(1, batch_size // (self.current_level + 1))
        
        for level in range(self.current_level + 1):
            level_samples = self.level_buffers[level].sample(samples_per_level)
            samples.extend(level_samples)
        
        return samples[:batch_size]
    
    def get_statistics(self) -> Dict:
        """获取所有级别的统计信息"""
        stats = {
            'current_level': self.current_level,
            'levels': {}
        }
        
        for level in range(self.num_levels):
            stats['levels'][level] = self.level_buffers[level].get_statistics()
        
        return stats
