"""
参数化地形生成器网络 (Parametric Terrain Generator)

与 parametric_terrain.py 配合使用。
GAN 输出 terrain_type (离散) + 20 维连续参数向量，
parametric_terrain 将参数解码为 heightfield。
"""
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import numpy as np

from legged_gym.utils.parametric_terrain import (
    PARAM_SPEC, PARAM_NAMES, NUM_TERRAIN_TYPES, NUM_CONTINUOUS_PARAMS,
    get_param_ranges, params_dict_from_vector,
)


class ParametricTerrainGenerator(nn.Module):
    """
    参数化地形 GAN 生成器

    输入: 条件向量 (训练状态统计)
    输出:
      - terrain_type: 离散 [0, NUM_TERRAIN_TYPES)
      - param_vector: 连续 [NUM_CONTINUOUS_PARAMS] 维，每维映射到对应参数范围
    """

    def __init__(
        self,
        condition_dim: int = 16,
        hidden_dims: list = None,
        use_lstm: bool = False,
        lstm_hidden_size: int = 128,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.num_types = NUM_TERRAIN_TYPES
        self.num_params = NUM_CONTINUOUS_PARAMS
        self.use_lstm = use_lstm

        # 参数范围 (注册为 buffer，不参与梯度)
        mins, maxs, defaults = get_param_ranges()
        self.register_buffer("param_mins", torch.tensor(mins, dtype=torch.float32))
        self.register_buffer("param_maxs", torch.tensor(maxs, dtype=torch.float32))
        self.register_buffer("param_defaults", torch.tensor(defaults, dtype=torch.float32))

        # 共享特征提取
        layers = []
        prev = condition_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        self.backbone = nn.Sequential(*layers)

        # LSTM (可选)
        if use_lstm:
            self.lstm = nn.LSTM(prev, lstm_hidden_size, batch_first=True)
            prev = lstm_hidden_size
            self.lstm_hidden = None

        # 离散头: terrain_type
        self.type_head = nn.Linear(prev, self.num_types)

        # 连续头: 参数均值 + 可学习 log_std
        self.param_mean = nn.Linear(prev, self.num_params)
        self.param_logstd = nn.Parameter(torch.zeros(self.num_params))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def reset_lstm(self, batch_size=1):
        if self.use_lstm:
            dev = next(self.parameters()).device
            self.lstm_hidden = (
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=dev),
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=dev),
            )

    def forward(self, condition, deterministic=False):
        """
        Args:
            condition: [B, condition_dim] 训练状态条件向量
            deterministic: 是否确定性输出

        Returns:
            terrain_type: [B] int, 地形类型索引
            param_vector: [B, NUM_CONTINUOUS_PARAMS] float, 实际参数值
            log_prob: [B] float, 总对数概率
            info: dict, 包含分布信息 (用于熵计算等)
        """
        feat = self.backbone(condition)

        if self.use_lstm:
            feat = feat.unsqueeze(1)
            feat, self.lstm_hidden = self.lstm(feat, self.lstm_hidden)
            feat = feat.squeeze(1)

        # --- 离散: terrain_type ---
        type_logits = self.type_head(feat)
        type_dist = Categorical(logits=type_logits)
        if deterministic:
            terrain_type = type_logits.argmax(dim=-1)
        else:
            terrain_type = type_dist.sample()
        type_lp = type_dist.log_prob(terrain_type)

        # --- 连续: 参数向量 ---
        raw_mean = self.param_mean(feat)
        std = torch.exp(self.param_logstd).expand_as(raw_mean)
        param_dist = Normal(raw_mean, std)
        if deterministic:
            raw = raw_mean
        else:
            raw = param_dist.rsample()
        param_lp = param_dist.log_prob(raw).sum(dim=-1)

        # sigmoid -> [0,1] -> 映射到 [min, max]
        normalized = torch.sigmoid(raw)
        param_vector = self.param_mins + normalized * (self.param_maxs - self.param_mins)

        log_prob = type_lp + param_lp

        info = {
            "type_logits": type_logits,
            "type_dist": type_dist,
            "param_dist": param_dist,
            "raw_params": raw,
            "normalized_params": normalized,
        }

        return terrain_type, param_vector, log_prob, info

    def get_entropy(self, condition):
        """计算策略熵 (用于正则化)"""
        feat = self.backbone(condition)
        if self.use_lstm:
            feat = feat.unsqueeze(1)
            feat, _ = self.lstm(feat, self.lstm_hidden)
            feat = feat.squeeze(1)

        type_logits = self.type_head(feat)
        type_entropy = Categorical(logits=type_logits).entropy()

        raw_mean = self.param_mean(feat)
        std = torch.exp(self.param_logstd).expand_as(raw_mean)
        param_entropy = Normal(raw_mean, std).entropy().sum(dim=-1)

        return type_entropy + param_entropy

    def to_numpy_params(self, terrain_type, param_vector):
        """
        将 tensor 输出转为 numpy dict，供 parametric_terrain 使用。

        Args:
            terrain_type: [B] tensor
            param_vector: [B, NUM_CONTINUOUS_PARAMS] tensor

        Returns:
            list of (int, dict) — 每个元素是 (type_idx, params_dict)
        """
        types = terrain_type.detach().cpu().numpy()
        vecs = param_vector.detach().cpu().numpy()
        results = []
        for i in range(len(types)):
            t = int(types[i])
            d = params_dict_from_vector(vecs[i])
            results.append((t, d))
        return results


class ParametricConditionBuilder:
    """
    构建 GAN 条件向量 — 与原 GeneratorInputBuilder 兼容但扩展了维度。

    条件向量 (16维):
      [0-2]  solver reward: mean, std, latest
      [3-5]  antagonist reward: mean, std, latest
      [6-9]  regret: mean, std, max, latest
      [10-12] difficulty: mean, std, latest
      [13]   training progress (0~1)
      [14]   terrain diversity (当前地形类型的熵)
      [15]   reserved
    """

    def __init__(self, device="cuda:0", history_length=10):
        self.device = device
        self.history_length = history_length
        self.solver_rewards = []
        self.antagonist_rewards = []
        self.regrets = []
        self.difficulties = []
        self.terrain_type_counts = np.zeros(NUM_TERRAIN_TYPES)

    def update(self, solver_reward, antagonist_reward, difficulty, terrain_type=None):
        self.solver_rewards.append(solver_reward)
        self.antagonist_rewards.append(antagonist_reward)
        self.regrets.append(antagonist_reward - solver_reward)
        self.difficulties.append(difficulty)
        if terrain_type is not None:
            self.terrain_type_counts[terrain_type % NUM_TERRAIN_TYPES] += 1
        # 保持固定长度
        for lst in [self.solver_rewards, self.antagonist_rewards,
                    self.regrets, self.difficulties]:
            while len(lst) > self.history_length:
                lst.pop(0)

    def build(self, batch_size=1):
        feats = []

        def _stats(arr):
            if arr:
                return [np.mean(arr), np.std(arr), arr[-1]]
            return [0.0, 0.0, 0.0]

        feats.extend(_stats(self.solver_rewards))
        feats.extend(_stats(self.antagonist_rewards))

        if self.regrets:
            feats.extend([np.mean(self.regrets), np.std(self.regrets),
                          np.max(self.regrets), self.regrets[-1]])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

        feats.extend(_stats(self.difficulties))

        # training progress
        feats.append(len(self.solver_rewards) / max(1, self.history_length))

        # terrain diversity (归一化熵)
        total = self.terrain_type_counts.sum()
        if total > 0:
            probs = self.terrain_type_counts / total
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs)) / np.log(NUM_TERRAIN_TYPES)
        else:
            entropy = 0.0
        feats.append(entropy)

        # pad to 16
        while len(feats) < 16:
            feats.append(0.0)

        t = torch.tensor(feats[:16], dtype=torch.float32, device=self.device)
        return t.unsqueeze(0).expand(batch_size, -1)

    def reset(self):
        self.solver_rewards.clear()
        self.antagonist_rewards.clear()
        self.regrets.clear()
        self.difficulties.clear()
        self.terrain_type_counts[:] = 0
