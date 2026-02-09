"""
Adversarial Curriculum Trainer
真正的对抗-生成框架：生成器生成难环境，Solver 学习适应

核心修复:
1. 生成器的参数真正应用到环境
2. Solver 在生成器指定的环境中训练
3. 形成闭环对抗
"""
import torch
import torch.nn as nn
import numpy as np
import copy
import os
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from .generator import TerrainGenerator, GeneratorInputBuilder


@dataclass
class TrainingStats:
    iteration: int
    solver_reward: float
    antagonist_reward: float
    regret: float
    generator_loss: float
    terrain_type: int
    difficulty: float
    friction: float
    is_novel: bool
    accept_rate: float
    # === 概率下界统计 ===
    is_easy_terrain: bool = False
    easy_terrain_prob: float = 0.0
    easy_terrain_count: int = 0
    # === 可行性过滤统计 ===
    is_feasible: bool = True
    infeasible_reason: str = ""
    infeasible_count: int = 0


class FeasibilityFilter:
    """
    可行性过滤器 - 检查环境参数是否物理可行
    
    目的：丢弃物理上不可能完成的环境，避免无效梯度
    
    检查规则：
    1. 摩擦系数不能太低（否则任何策略都会滑倒）
    2. 扰动力不能太强（否则任何策略都会被推倒）
    3. 负载不能太重（否则电机无法驱动）
    4. 组合检查：低摩擦 + 强扰动 = 不可行
    """
    
    def __init__(
        self,
        min_friction: float = 0.15,
        max_push_magnitude: float = 2.5,
        max_added_mass: float = 3.5,
        # 组合约束
        low_friction_threshold: float = 0.3,
        max_push_with_low_friction: float = 1.0,
    ):
        self.min_friction = min_friction
        self.max_push_magnitude = max_push_magnitude
        self.max_added_mass = max_added_mass
        self.low_friction_threshold = low_friction_threshold
        self.max_push_with_low_friction = max_push_with_low_friction
        
        # 统计
        self.total_checked = 0
        self.total_rejected = 0
        self.rejection_reasons = {}
    
    def is_feasible(self, params: Dict) -> Tuple[bool, str]:
        """
        检查参数是否物理可行
        
        Args:
            params: 环境参数字典
            
        Returns:
            (is_feasible, reason): 是否可行，不可行的原因
        """
        self.total_checked += 1
        
        friction = params.get('friction', 1.0)
        push_mag = params.get('push_magnitude', 0.0)
        added_mass = params.get('added_mass', 0.0)
        difficulty = params.get('difficulty', 0.5)
        
        # 1. 摩擦系数检查
        if friction < self.min_friction:
            reason = f"Friction too low: {friction:.2f} < {self.min_friction}"
            self._record_rejection(reason)
            return False, reason
        
        # 2. 扰动力检查
        if push_mag > self.max_push_magnitude:
            reason = f"Push too strong: {push_mag:.2f} > {self.max_push_magnitude}"
            self._record_rejection(reason)
            return False, reason
        
        # 3. 负载检查
        if added_mass > self.max_added_mass:
            reason = f"Mass too heavy: {added_mass:.2f} > {self.max_added_mass}"
            self._record_rejection(reason)
            return False, reason
        
        # 4. 组合检查：低摩擦 + 强扰动
        if friction < self.low_friction_threshold and push_mag > self.max_push_with_low_friction:
            reason = f"Low friction ({friction:.2f}) + strong push ({push_mag:.2f})"
            self._record_rejection(reason)
            return False, reason
        
        # 5. 组合检查：低摩擦 + 高难度
        if friction < self.low_friction_threshold and difficulty > 0.8:
            reason = f"Low friction ({friction:.2f}) + high difficulty ({difficulty:.2f})"
            self._record_rejection(reason)
            return False, reason
        
        # 6. 组合检查：重负载 + 强扰动
        if added_mass > 2.0 and push_mag > 1.5:
            reason = f"Heavy load ({added_mass:.2f}) + strong push ({push_mag:.2f})"
            self._record_rejection(reason)
            return False, reason
        
        return True, "OK"
    
    def _record_rejection(self, reason: str):
        """记录拒绝原因"""
        self.total_rejected += 1
        # 简化原因用于统计
        simple_reason = reason.split(':')[0] if ':' in reason else reason
        self.rejection_reasons[simple_reason] = self.rejection_reasons.get(simple_reason, 0) + 1
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_checked': self.total_checked,
            'total_rejected': self.total_rejected,
            'rejection_rate': self.total_rejected / max(self.total_checked, 1),
            'rejection_reasons': self.rejection_reasons.copy(),
        }


class AdversarialCurriculumTrainer:
    """
    对抗课程训练器
    
    真正实现:
    - 生成器生成环境参数 → 应用到仿真环境
    - Solver 在该环境中训练
    - 计算遗憾值 → 更新生成器
    
    AT-PC 稳定性机制:
    - 概率下界 (Probabilistic Lower Bounds): 以一定概率采样简单地形
    - 防止灾难性遗忘 (Catastrophic Forgetting)
    """
    
    def __init__(
        self,
        env,
        ppo_runner,
        device: str = 'cuda:0',
        generator_lr: float = 3e-4,
        novelty_threshold: float = 0.7,
        use_novelty_filter: bool = True,  # ReMiDi 开关
        log_dir: str = None,
        # === 概率下界参数 ===
        easy_terrain_prob: float = 0.15,  # 采样简单地形的概率
        easy_terrain_decay: float = 0.995,  # 概率衰减率 (随训练进行逐渐降低)
        min_easy_prob: float = 0.05,  # 最小简单地形概率 (永不为0)
        # === 可行性过滤参数 ===
        use_feasibility_filter: bool = True,
        max_infeasible_attempts: int = 5,  # 最大不可行尝试次数
        # === 热身协议参数 ===
        warmup_iterations: int = 0,  # 热身迭代次数 (0 表示不使用显式热身)
    ):
        self.env = env
        self.ppo_runner = ppo_runner
        self.device = device
        self.log_dir = log_dir
        self.use_novelty_filter = use_novelty_filter
        
        # === 概率下界机制 (AT-PC 稳定性) ===
        self.easy_terrain_prob = easy_terrain_prob
        self.easy_terrain_decay = easy_terrain_decay
        self.min_easy_prob = min_easy_prob
        self.current_easy_prob = easy_terrain_prob
        self.easy_terrain_count = 0  # 统计简单地形采样次数
        
        # === 可行性过滤 ===
        self.use_feasibility_filter = use_feasibility_filter
        self.max_infeasible_attempts = max_infeasible_attempts
        self.feasibility_filter = FeasibilityFilter() if use_feasibility_filter else None
        self.infeasible_count = 0  # 统计不可行环境次数
        
        # === 热身协议 ===
        self.warmup_iterations = warmup_iterations
        self.warmup_completed = False
        self.warmup_count = 0
        
        # 简单地形的参数定义
        self.easy_terrain_params = {
            'terrain_type': 0,  # 平地
            'difficulty': 0.1,  # 低难度
            'friction': 1.0,    # 正常摩擦
            'push_magnitude': 0.0,  # 无扰动
            'added_mass': 0.0,  # 无负载
        }
        
        # 保存原始环境配置
        self.original_cfg = self._save_env_config()
        
        # 生成器
        self.generator = TerrainGenerator(
            hidden_dims=[128, 64],
            num_terrain_types=7,
            use_lstm=True,
            lstm_hidden_size=64
        ).to(device)
        
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=generator_lr
        )
        
        # 输入构建器
        self.input_builder = GeneratorInputBuilder(device=device)
        
        # Antagonist: Solver 的 EMA 版本
        self.antagonist = None
        self.antagonist_ema = 0.99
        
        # 新颖性检测 (ReMiDi)
        self.novelty_threshold = novelty_threshold
        self.trajectory_buffer = []
        self.max_buffer_size = 200
        
        # 统计
        self.iteration = 0
        self.accepted = 0
        self.rejected = 0
        self.stats_history = []
        
        # 初始化
        self.generator.reset_lstm(batch_size=1)
    
    def _sample_easy_terrain(self) -> bool:
        """
        概率下界机制：决定是否采样简单地形
        
        Returns:
            True: 使用简单地形 (防止遗忘)
            False: 使用生成器生成的地形
        """
        # 以当前概率采样简单地形
        use_easy = np.random.random() < self.current_easy_prob
        
        if use_easy:
            self.easy_terrain_count += 1
        
        return use_easy
    
    def _decay_easy_prob(self):
        """
        衰减简单地形采样概率
        
        随着训练进行，逐渐降低简单地形的比例，
        但永远保持一个最小值 (min_easy_prob)
        """
        self.current_easy_prob = max(
            self.min_easy_prob,
            self.current_easy_prob * self.easy_terrain_decay
        )
    
    def _get_easy_terrain_params(self) -> Dict:
        """
        获取简单地形参数
        
        可以添加轻微随机化，避免完全相同的简单环境
        """
        params = self.easy_terrain_params.copy()
        
        # 轻微随机化 (保持简单但不完全相同)
        params['friction'] = np.random.uniform(0.9, 1.1)
        params['difficulty'] = np.random.uniform(0.05, 0.15)
        params['push_magnitude'] = np.random.uniform(0.0, 0.1)
        params['added_mass'] = np.random.uniform(-0.2, 0.2)
        
        return params
    
    def _get_warmup_terrain_params(self) -> Dict:
        """
        获取热身阶段的随机地形参数
        
        热身阶段使用中低难度的均匀随机参数，
        让 Solver 获得基础运动能力
        """
        return {
            'terrain_type': np.random.randint(0, 7),
            'difficulty': np.random.uniform(0.1, 0.5),  # 中低难度
            'friction': np.random.uniform(0.5, 1.2),    # 正常范围
            'push_magnitude': np.random.uniform(0.0, 0.5),  # 轻微扰动
            'added_mass': np.random.uniform(-0.5, 0.5),  # 轻微负载变化
        }
    
    def is_in_warmup(self) -> bool:
        """检查是否在热身阶段"""
        if self.warmup_iterations <= 0:
            return False
        return not self.warmup_completed
    
    def _save_env_config(self) -> Dict:
        """保存原始环境配置"""
        cfg = self.env.cfg
        return {
            'friction_range': list(cfg.domain_rand.friction_range) if hasattr(cfg.domain_rand, 'friction_range') else [0.5, 1.25],
            'added_mass_range': list(cfg.domain_rand.added_mass_range) if hasattr(cfg.domain_rand, 'added_mass_range') else [-1.0, 1.0],
            'push_robots': cfg.domain_rand.push_robots if hasattr(cfg.domain_rand, 'push_robots') else False,
            'max_push_vel_xy': cfg.domain_rand.max_push_vel_xy if hasattr(cfg.domain_rand, 'max_push_vel_xy') else 1.0,
        }
    
    def _apply_terrain_to_env(self, params: Dict):
        """
        将生成器参数真正应用到环境
        
        关键: 直接修改环境的物理属性，而不只是配置
        """
        cfg = self.env.cfg
        
        # 1. 摩擦系数
        friction = float(params['friction'])
        cfg.domain_rand.friction_range = [
            max(0.1, friction - 0.1),
            min(2.0, friction + 0.1)
        ]
        
        # 直接修改环境中的摩擦系数 (如果支持)
        if hasattr(self.env, 'friction_coeffs'):
            self.env.friction_coeffs[:] = friction
        
        # 2. 外力扰动
        push_mag = float(params['push_magnitude'])
        cfg.domain_rand.max_push_vel_xy = push_mag
        if push_mag > 0.1:
            cfg.domain_rand.push_robots = True
            cfg.domain_rand.push_interval_s = 8
        else:
            cfg.domain_rand.push_robots = False
        
        # 3. 附加质量
        added_mass = float(params['added_mass'])
        cfg.domain_rand.added_mass_range = [
            added_mass - 0.3,
            added_mass + 0.3
        ]
        
        # 4. 地形难度 (如果支持动态地形)
        difficulty = float(params['difficulty'])
        if hasattr(cfg, 'terrain') and hasattr(cfg.terrain, 'difficulty_scale'):
            cfg.terrain.difficulty_scale = difficulty
    
    def _init_antagonist(self):
        """初始化 Antagonist 为 Solver 的副本"""
        self.antagonist = copy.deepcopy(self.ppo_runner.alg.actor_critic)
        self.antagonist.eval()
    
    def _update_antagonist(self):
        """EMA 更新 Antagonist"""
        if self.antagonist is None:
            self._init_antagonist()
            return
        
        with torch.no_grad():
            for p_ant, p_sol in zip(
                self.antagonist.parameters(),
                self.ppo_runner.alg.actor_critic.parameters()
            ):
                p_ant.data.mul_(self.antagonist_ema)
                p_ant.data.add_((1 - self.antagonist_ema) * p_sol.data)
    
    def _evaluate_policy(self, policy, num_steps: int = 24) -> Tuple[float, torch.Tensor]:
        """
        评估策略在当前环境中的表现
        
        注意：减少评估步数以加快训练速度
        
        Returns:
            mean_reward: 平均奖励
            trajectory_features: 轨迹特征 (用于新颖性检测)
        """
        policy.eval()
        
        obs = self.env.get_observations()
        total_reward = 0
        obs_list = []
        
        with torch.no_grad():
            for _ in range(num_steps):
                actions = policy.act_inference(obs)
                obs, _, rewards, dones, _ = self.env.step(actions)
                total_reward += rewards.mean().item()
                obs_list.append(obs.mean(dim=0))
        
        mean_reward = total_reward / num_steps
        trajectory_features = torch.stack(obs_list).flatten()
        
        return mean_reward, trajectory_features
    
    def _check_novelty(self, traj_features: torch.Tensor) -> Tuple[bool, float]:
        """
        检查轨迹新颖性 (ReMiDi 核心) - GPU 加速版本
        
        如果新轨迹与历史轨迹过于相似，则拒绝
        """
        if not self.use_novelty_filter or len(self.trajectory_buffer) == 0:
            return True, 0.0
        
        # 确保在 GPU 上计算
        traj_features = traj_features.to(self.device)
        traj_norm = traj_features / (traj_features.norm() + 1e-8)
        
        # 批量计算相似度 (GPU 加速)
        recent_trajs = self.trajectory_buffer[-50:]
        if len(recent_trajs) > 0:
            # 堆叠成矩阵，一次性计算所有相似度
            cached_stack = torch.stack([t.to(self.device) for t in recent_trajs])
            cached_norms = cached_stack / (cached_stack.norm(dim=1, keepdim=True) + 1e-8)
            
            # 矩阵乘法计算所有相似度
            similarities = torch.mv(cached_norms, traj_norm)
            similarities = (similarities + 1) / 2  # 归一化到 [0, 1]
            
            max_similarity = similarities.max().item()
        else:
            max_similarity = 0.0
        
        is_novel = max_similarity < self.novelty_threshold
        return is_novel, max_similarity
    
    def _add_to_buffer(self, traj_features: torch.Tensor):
        """添加轨迹到缓冲区"""
        self.trajectory_buffer.append(traj_features.detach())
        if len(self.trajectory_buffer) > self.max_buffer_size:
            self.trajectory_buffer.pop(0)
    
    def generate_and_apply_curriculum(self) -> Tuple[Dict, bool, bool, str]:
        """
        生成课程并应用到环境
        
        AT-PC 稳定性机制：
        1. 热身协议：训练初期使用随机中低难度环境
        2. 概率下界：以一定概率采样简单地形，防止灾难性遗忘
        3. 可行性过滤：丢弃物理上不可行的环境
        
        Returns:
            terrain_params: 生成的地形参数
            is_easy_terrain: 是否使用了简单地形
            is_warmup: 是否在热身阶段
            source: 参数来源 ("WARMUP", "EASY", "GENERATOR", "FALLBACK")
        """
        # === 热身协议：训练初期使用随机环境 ===
        if self.is_in_warmup():
            self.warmup_count += 1
            params = self._get_warmup_terrain_params()
            self.last_log_prob = None
            
            # 检查是否完成热身
            if self.warmup_count >= self.warmup_iterations:
                self.warmup_completed = True
                print(f"\n{'='*50}")
                print(f"[Warmup Complete] {self.warmup_count} iterations")
                print(f"{'='*50}\n")
            
            self._apply_terrain_to_env(params)
            return params, False, True, "WARMUP"
        
        # === 概率下界机制：决定是否使用简单地形 ===
        use_easy = self._sample_easy_terrain()
        
        if use_easy:
            # 使用简单地形（防止遗忘基础技能）
            params = self._get_easy_terrain_params()
            self.last_log_prob = None
            self._apply_terrain_to_env(params)
            self._decay_easy_prob()
            return params, True, False, "EASY"
        
        # === 使用生成器生成的地形 ===
        self.generator.eval()
        
        # 可行性过滤：最多尝试 max_infeasible_attempts 次
        for attempt in range(self.max_infeasible_attempts):
            with torch.no_grad():
                gen_input = self.input_builder.build(batch_size=1)
                terrain_params, self.last_log_prob = self.generator(gen_input, deterministic=False)
            
            # 转换为 numpy
            params = {
                'terrain_type': terrain_params['terrain_type'].item(),
                'difficulty': terrain_params['difficulty'].item(),
                'friction': terrain_params['friction'].item(),
                'push_magnitude': terrain_params['push_magnitude'].item(),
                'added_mass': terrain_params['added_mass'].item(),
            }
            
            # 可行性检查
            if self.use_feasibility_filter and self.feasibility_filter is not None:
                is_feasible, reason = self.feasibility_filter.is_feasible(params)
                if not is_feasible:
                    self.infeasible_count += 1
                    continue  # 重新生成
            
            # 可行，应用到环境
            self._apply_terrain_to_env(params)
            self._decay_easy_prob()
            return params, False, False, "GENERATOR"
        
        # 多次尝试都不可行，使用简单地形作为 fallback
        self.infeasible_count += 1
        params = self._get_easy_terrain_params()
        self.last_log_prob = None
        self._apply_terrain_to_env(params)
        self._decay_easy_prob()
        return params, True, False, "FALLBACK"
    
    def compute_regret_and_update(self, terrain_params: Dict, is_easy_terrain: bool = False, is_warmup: bool = False, source: str = "GENERATOR") -> TrainingStats:
        """
        计算遗憾值并更新生成器
        
        这是对抗训练的核心:
        1. Solver 在当前环境中的表现
        2. Antagonist 在当前环境中的表现
        3. Regret = Antagonist - Solver
        4. 生成器最大化 Regret
        
        注意：
        - 热身阶段：不更新生成器，不初始化 Antagonist
        - 简单地形（概率下界采样）：不更新生成器
        - Fallback：不更新生成器
        """
        self.iteration += 1
        
        # === 热身阶段：只训练 Solver ===
        if is_warmup:
            # 热身阶段不需要 Antagonist 和 Regret
            solver_reward, _ = self._evaluate_policy(
                self.ppo_runner.alg.actor_critic, num_steps=24
            )
            
            stats = TrainingStats(
                iteration=self.iteration,
                solver_reward=solver_reward,
                antagonist_reward=0.0,
                regret=0.0,
                generator_loss=0.0,
                terrain_type=terrain_params['terrain_type'],
                difficulty=terrain_params['difficulty'],
                friction=terrain_params['friction'],
                is_novel=True,
                accept_rate=0.0,
                is_easy_terrain=False,
                easy_terrain_prob=self.current_easy_prob,
                easy_terrain_count=self.easy_terrain_count,
                is_feasible=True,
                infeasible_reason="",
                infeasible_count=self.infeasible_count,
            )
            self.stats_history.append(stats)
            return stats
        
        # === 正常对抗训练 ===
        # 确保 Antagonist 存在
        if self.antagonist is None:
            self._init_antagonist()
        
        # 评估 Solver
        solver_reward, solver_traj = self._evaluate_policy(
            self.ppo_runner.alg.actor_critic, num_steps=24
        )
        
        # 评估 Antagonist
        antagonist_reward, _ = self._evaluate_policy(
            self.antagonist, num_steps=24
        )
        
        # 计算遗憾值
        regret = antagonist_reward - solver_reward
        
        # === 概率下界/Fallback：简单地形不参与对抗更新 ===
        if is_easy_terrain or source == "FALLBACK":
            # 简单地形：只训练 Solver，不更新生成器
            is_novel = True  # 简单地形总是"新颖"的（不参与新颖性检测）
            generator_loss = 0.0
            # 不添加到轨迹缓冲区，不更新生成器
        else:
            # 生成器地形：正常的对抗更新流程
            # 检查新颖性 (ReMiDi)
            is_novel, similarity = self._check_novelty(solver_traj)
            
            # 更新生成器
            generator_loss = 0.0
            if is_novel:
                self.accepted += 1
                self._add_to_buffer(solver_traj)
                generator_loss = self._update_generator(regret)
            else:
                self.rejected += 1
        
        # 更新 Antagonist（无论是否简单地形都更新）
        self._update_antagonist()
        
        # 更新输入构建器
        self.input_builder.update(
            solver_reward, antagonist_reward, terrain_params['difficulty']
        )
        
        # 统计
        accept_rate = self.accepted / max(self.accepted + self.rejected, 1)
        
        # 获取可行性过滤统计
        feasibility_stats = self.feasibility_filter.get_stats() if self.feasibility_filter else {}
        
        stats = TrainingStats(
            iteration=self.iteration,
            solver_reward=solver_reward,
            antagonist_reward=antagonist_reward,
            regret=regret,
            generator_loss=generator_loss,
            terrain_type=terrain_params['terrain_type'],
            difficulty=terrain_params['difficulty'],
            friction=terrain_params['friction'],
            is_novel=is_novel,
            accept_rate=accept_rate,
            # === 概率下界统计 ===
            is_easy_terrain=is_easy_terrain,
            easy_terrain_prob=self.current_easy_prob,
            easy_terrain_count=self.easy_terrain_count,
            # === 可行性过滤统计 ===
            is_feasible=(source != "FALLBACK"),
            infeasible_reason="" if source != "FALLBACK" else "Max attempts exceeded",
            infeasible_count=self.infeasible_count,
        )
        
        self.stats_history.append(stats)
        
        return stats
    
    def _update_generator(self, regret: float) -> float:
        """
        更新生成器 (策略梯度)
        
        目标: 最大化遗憾值
        Loss = -regret * log_prob
        """
        self.generator.train()
        
        # 重新计算 log_prob (需要梯度)
        gen_input = self.input_builder.build(batch_size=1)
        _, log_prob = self.generator(gen_input, deterministic=False)
        
        # 策略梯度损失
        loss = -regret * log_prob.mean()
        
        # 熵正则化 (鼓励探索)
        entropy = self.generator.get_entropy(gen_input).mean()
        entropy_coef = 0.02
        loss = loss - entropy_coef * entropy
        
        # 反向传播
        self.generator_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.5)
        self.generator_optimizer.step()
        
        return loss.item()
    
    def save(self, path: str = None):
        """保存检查点"""
        if path is None:
            path = os.path.join(self.log_dir, f'adversarial_ckpt_{self.iteration}.pt')
        
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict(),
            'optimizer': self.generator_optimizer.state_dict(),
            'accepted': self.accepted,
            'rejected': self.rejected,
            'stats_history': [(s.iteration, s.regret, s.accept_rate) for s in self.stats_history[-100:]],
        }, path)
    
    def load(self, path: str):
        """加载检查点"""
        ckpt = torch.load(path, map_location=self.device)
        self.iteration = ckpt['iteration']
        self.generator.load_state_dict(ckpt['generator'])
        self.generator_optimizer.load_state_dict(ckpt['optimizer'])
        self.accepted = ckpt.get('accepted', 0)
        self.rejected = ckpt.get('rejected', 0)
