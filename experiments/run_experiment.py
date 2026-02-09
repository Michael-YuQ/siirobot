"""
Unified experiment entry point - one script runs one experiment.

Usage:
    python -m experiments.run_experiment \
        --generator G1 --method atpc --seed 42 \
        --max_iterations 2000 --num_envs 512 --headless
"""
import os
import sys
import time
import json
import argparse
import copy
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import isaacgym

import torch
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict

from experiments.config import TRAIN_CFG, METHODS, COMMON_TEST_TERRAINS
from experiments.generators import get_generator, make_generator_network, BaseTerrainGenerator
from experiments.uploader import ResultUploader


def _json_default(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class PluggableAdversarialTrainer:
    def __init__(self, env, ppo_runner, terrain_generator, device="cuda:0",
                 cfg=None, log_dir=None):
        self.env = env
        self.ppo_runner = ppo_runner
        self.device = device
        self.log_dir = log_dir
        self.terrain_gen = terrain_generator
        c = cfg or TRAIN_CFG
        self.gen_net = make_generator_network(terrain_generator, condition_dim=16, device=device)
        self.gen_optimizer = torch.optim.Adam(self.gen_net.parameters(), lr=c["generator_lr"])
        from legged_gym.curriculum.generator import GeneratorInputBuilder
        self.input_builder = GeneratorInputBuilder(device=device)
        self.antagonist = None
        self.antagonist_ema = 0.99
        self.use_novelty = c.get("use_novelty", True)
        self.novelty_threshold = c["novelty_threshold"]
        self.trajectory_buffer = []
        self.max_buffer_size = c["trajectory_buffer_size"]
        self.current_easy_prob = c["easy_terrain_prob"]
        self.easy_decay = c["easy_terrain_decay"]
        self.min_easy_prob = c["min_easy_prob"]
        self.easy_count = 0
        self.warmup_iters = c["warmup_iterations"]
        self.warmup_done = False
        self.warmup_count = 0
        from legged_gym.curriculum.adversarial_trainer import FeasibilityFilter
        self.feasibility_filter = FeasibilityFilter()
        self.iteration = 0
        self.accepted = 0
        self.rejected = 0
        self.stats_history = []

    def _init_antagonist(self):
        self.antagonist = copy.deepcopy(self.ppo_runner.alg.actor_critic)
        self.antagonist.eval()

    def _update_antagonist(self):
        if self.antagonist is None:
            self._init_antagonist()
            return
        with torch.no_grad():
            for pa, ps in zip(self.antagonist.parameters(),
                              self.ppo_runner.alg.actor_critic.parameters()):
                pa.data.mul_(self.antagonist_ema).add_((1 - self.antagonist_ema) * ps.data)

    def _evaluate_policy(self, policy, num_steps=24):
        policy.eval()
        obs = self.env.get_observations()
        total_r = 0.0
        obs_list = []
        with torch.no_grad():
            for _ in range(num_steps):
                actions = policy.act_inference(obs)
                obs, _, rewards, dones, _ = self.env.step(actions)
                total_r += rewards.mean().item()
                obs_list.append(obs.mean(dim=0))
        traj = torch.stack(obs_list).flatten()
        return total_r / num_steps, traj

    def _check_novelty(self, traj):
        if not self.use_novelty or len(self.trajectory_buffer) == 0:
            return True, 0.0
        traj = traj.to(self.device)
        tn = traj / (traj.norm() + 1e-8)
        recent = self.trajectory_buffer[-50:]
        stack = torch.stack([t.to(self.device) for t in recent])
        norms = stack / (stack.norm(dim=1, keepdim=True) + 1e-8)
        sims = torch.mv(norms, tn)
        sims = (sims + 1) / 2
        max_sim = sims.max().item()
        return max_sim < self.novelty_threshold, max_sim

    def _add_to_buffer(self, traj):
        self.trajectory_buffer.append(traj.detach())
        if len(self.trajectory_buffer) > self.max_buffer_size:
            self.trajectory_buffer.pop(0)

    def _apply_params(self, params_dict):
        self.terrain_gen.apply_to_env(self.env, params_dict.get("terrain_type", 0), params_dict)
        cfg = self.env.cfg
        diff = params_dict.get("difficulty", 0.5)
        if hasattr(cfg, "terrain") and hasattr(cfg.terrain, "difficulty_scale"):
            cfg.terrain.difficulty_scale = diff
