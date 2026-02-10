"""
Terrain Generator Network for PAIRED/ReMiDi
生成器网络：输出环境参数，用于自适应课程生成
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np


class TerrainGenerator(nn.Module):
    """
    地形生成器网络
    
    输出参数空间:
    - terrain_type: 离散 [0-6], 对应7种地形类型
    - difficulty: 连续 [0, 1], 控制地形难度
    - friction: 连续 [0.3, 1.5], 地面摩擦系数
    - push_magnitude: 连续 [0, 1.5], 外力扰动强度
    - added_mass: 连续 [-1.5, 2.0], 附加质量
    """
    
    def __init__(
        self,
        hidden_dims=[256, 128],
        num_terrain_types=7,
        use_lstm=False,
        lstm_hidden_size=128,
    ):
        super().__init__()
        
        self.num_terrain_types = num_terrain_types
        self.use_lstm = use_lstm
        
        # 输入: 历史统计信息 (solver性能, antagonist性能, 当前难度等)
        input_dim = 16  # 可扩展
        
        # 共享特征提取层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        
        # LSTM 用于记忆历史生成分布
        if use_lstm:
            self.lstm = nn.LSTM(prev_dim, lstm_hidden_size, batch_first=True)
            prev_dim = lstm_hidden_size
            self.lstm_hidden = None
        
        # 输出头
        # 1. 地形类型 (离散)
        self.terrain_type_head = nn.Linear(prev_dim, num_terrain_types)
        
        # 2. 连续参数 (difficulty, friction, push_magnitude, added_mass)
        # 输出均值和标准差
        self.continuous_mean = nn.Linear(prev_dim, 4)
        self.continuous_logstd = nn.Parameter(torch.zeros(4))
        
        # 参数范围
        self.param_ranges = {
            'difficulty': (0.0, 1.0),
            'friction': (0.3, 1.5),
            'push_magnitude': (0.0, 1.5),
            'added_mass': (-1.5, 2.0),
        }
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
    
    def reset_lstm(self, batch_size=1):
        """重置 LSTM 隐藏状态"""
        if self.use_lstm:
            device = next(self.parameters()).device
            self.lstm_hidden = (
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=device),
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
            )
    
    def forward(self, x, deterministic=False):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            deterministic: 是否使用确定性输出
            
        Returns:
            terrain_params: 生成的地形参数字典
            log_prob: 动作的对数概率 (用于策略梯度)
        """
        # 特征提取
        features = self.feature_extractor(x)
        
        # LSTM
        if self.use_lstm:
            features = features.unsqueeze(1)  # [batch, 1, features]
            features, self.lstm_hidden = self.lstm(features, self.lstm_hidden)
            features = features.squeeze(1)
        
        # 地形类型 (离散)
        terrain_logits = self.terrain_type_head(features)
        terrain_dist = Categorical(logits=terrain_logits)
        
        if deterministic:
            terrain_type = terrain_logits.argmax(dim=-1)
        else:
            terrain_type = terrain_dist.sample()
        
        terrain_log_prob = terrain_dist.log_prob(terrain_type)
        
        # 连续参数
        continuous_mean = self.continuous_mean(features)
        continuous_std = torch.exp(self.continuous_logstd).expand_as(continuous_mean)
        continuous_dist = Normal(continuous_mean, continuous_std)
        
        if deterministic:
            continuous_raw = continuous_mean
        else:
            continuous_raw = continuous_dist.rsample()
        
        continuous_log_prob = continuous_dist.log_prob(continuous_raw).sum(dim=-1)
        
        # 将连续参数映射到实际范围
        continuous_normalized = torch.sigmoid(continuous_raw)  # [0, 1]
        
        difficulty = continuous_normalized[:, 0]
        friction = continuous_normalized[:, 1] * 1.2 + 0.3  # [0.3, 1.5]
        push_magnitude = continuous_normalized[:, 2] * 1.5  # [0, 1.5]
        added_mass = continuous_normalized[:, 3] * 3.5 - 1.5  # [-1.5, 2.0]
        
        terrain_params = {
            'terrain_type': terrain_type,
            'difficulty': difficulty,
            'friction': friction,
            'push_magnitude': push_magnitude,
            'added_mass': added_mass,
        }
        
        total_log_prob = terrain_log_prob + continuous_log_prob
        
        return terrain_params, total_log_prob
    
    def get_entropy(self, x):
        """计算策略熵"""
        features = self.feature_extractor(x)
        
        if self.use_lstm:
            features = features.unsqueeze(1)
            features, _ = self.lstm(features, self.lstm_hidden)
            features = features.squeeze(1)
        
        # 离散熵
        terrain_logits = self.terrain_type_head(features)
        terrain_dist = Categorical(logits=terrain_logits)
        discrete_entropy = terrain_dist.entropy()
        
        # 连续熵
        continuous_mean = self.continuous_mean(features)
        continuous_std = torch.exp(self.continuous_logstd).expand_as(continuous_mean)
        continuous_dist = Normal(continuous_mean, continuous_std)
        continuous_entropy = continuous_dist.entropy().sum(dim=-1)
        
        return discrete_entropy + continuous_entropy


class GeneratorInputBuilder:
    """
    构建生成器输入特征
    """
    
    def __init__(self, device='cuda:0'):
        self.device = device
        self.history_length = 10
        
        # 历史记录
        self.solver_rewards = []
        self.antagonist_rewards = []
        self.regrets = []
        self.difficulties = []
    
    def update(self, solver_reward, antagonist_reward, difficulty):
        """更新历史记录"""
        self.solver_rewards.append(solver_reward)
        self.antagonist_rewards.append(antagonist_reward)
        self.regrets.append(antagonist_reward - solver_reward)
        self.difficulties.append(difficulty)
        
        # 保持固定长度
        if len(self.solver_rewards) > self.history_length:
            self.solver_rewards.pop(0)
            self.antagonist_rewards.pop(0)
            self.regrets.pop(0)
            self.difficulties.pop(0)
    
    def build(self, batch_size=1):
        """
        构建输入特征
        
        特征包括:
        - 最近的 solver/antagonist 奖励统计
        - 遗憾值统计
        - 难度统计
        """
        features = []
        
        # Solver 奖励统计
        if self.solver_rewards:
            features.extend([
                np.mean(self.solver_rewards),
                np.std(self.solver_rewards),
                self.solver_rewards[-1],
            ])
        else:
            features.extend([0, 0, 0])
        
        # Antagonist 奖励统计
        if self.antagonist_rewards:
            features.extend([
                np.mean(self.antagonist_rewards),
                np.std(self.antagonist_rewards),
                self.antagonist_rewards[-1],
            ])
        else:
            features.extend([0, 0, 0])
        
        # 遗憾值统计
        if self.regrets:
            features.extend([
                np.mean(self.regrets),
                np.std(self.regrets),
                np.max(self.regrets),
                self.regrets[-1],
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # 难度统计
        if self.difficulties:
            features.extend([
                np.mean(self.difficulties),
                np.std(self.difficulties),
                self.difficulties[-1],
            ])
        else:
            features.extend([0.5, 0, 0.5])
        
        # 训练进度 (可以从外部传入)
        features.append(len(self.solver_rewards) / self.history_length)
        
        # 填充到固定维度
        while len(features) < 16:
            features.append(0)
        
        features = torch.tensor(features, dtype=torch.float32, device=self.device)
        features = features.unsqueeze(0).expand(batch_size, -1)
        
        return features
    
    def reset(self):
        """重置历史"""
        self.solver_rewards = []
        self.antagonist_rewards = []
        self.regrets = []
        self.difficulties = []
