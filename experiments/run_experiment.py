"""
统一实验入口 — 一个脚本跑一个实验

用法:
    python -m experiments.run_experiment \
        --generator G1 --method atpc --seed 42 \
        --max_iterations 2000 --num_envs 512 \
        --upload_url http://your-server:8080/upload \
        --headless

参数:
    --generator   G1/G2/G3/G4
    --method      dr/paired/atpc
    --seed        随机种子
    --upload_url  上传地址 (可选, HTTP POST)
    --scp_target  SCP 上传目标 (可选, user@host:/path)
"""
import os
import sys
import time
import json
import argparse
import copy
from datetime import datetime

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import isaacgym  # 必须先导入

import torch
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict

from experiments.config import TRAIN_CFG, METHODS, COMMON_TEST_TERRAINS
from experiments.generators import get_generator, make_generator_network, BaseTerrainGenerator
from experiments.uploader import ResultUploader


# ============================================================
# 可插拔生成器的对抗训练器
# ============================================================

class PluggableAdversarialTrainer:
    """
    在 AdversarialCurriculumTrainer 基础上，
    将内部固定的 TerrainGenerator 替换为 experiments.generators 中的任意生成器。
    """

    def __init__(
        self,
        env,
        ppo_runner,
        terrain_generator: BaseTerrainGenerator,
        device: str = "cuda:0",
        cfg: dict = None,
        log_dir: str = None,
    ):
        self.env = env
        self.ppo_runner = ppo_runner
        self.device = device
        self.log_dir = log_dir
        self.terrain_gen = terrain_generator
        c = cfg or TRAIN_CFG

        # GAN 网络 (适配当前生成器的参数空间)
        self.gen_net = make_generator_network(
            terrain_generator, condition_dim=16, device=device
        )
        self.gen_optimizer = torch.optim.Adam(
            self.gen_net.parameters(), lr=c["generator_lr"]
        )

        # 输入构建器 (复用原有逻辑)
        from legged_gym.curriculum.generator import GeneratorInputBuilder
        self.input_builder = GeneratorInputBuilder(device=device)

        # Antagonist (EMA)
        self.antagonist = None
        self.antagonist_ema = 0.99

        # 新颖性检测
        self.use_novelty = c.get("use_novelty", True)
        self.novelty_threshold = c["novelty_threshold"]
        self.trajectory_buffer = []
        self.max_buffer_size = c["trajectory_buffer_size"]

        # 概率下界
        self.current_easy_prob = c["easy_terrain_prob"]
        self.easy_decay = c["easy_terrain_decay"]
        self.min_easy_prob = c["min_easy_prob"]
        self.easy_count = 0

        # 热身
        self.warmup_iters = c["warmup_iterations"]
        self.warmup_done = False
        self.warmup_count = 0

        # 可行性过滤
        from legged_gym.curriculum.adversarial_trainer import FeasibilityFilter
        self.feasibility_filter = FeasibilityFilter()

        # 统计
        self.iteration = 0
        self.accepted = 0
        self.rejected = 0
        self.stats_history = []

    # ------ Antagonist ------
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

    # ------ 评估 ------
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

    # ------ 新颖性 ------
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

    # ------ 应用参数到环境 ------
    def _apply_params(self, params_dict):
        """将生成器输出的参数应用到 env"""
        self.terrain_gen.apply_to_env(self.env, params_dict.get("terrain_type", 0), params_dict)
        cfg = self.env.cfg
        # difficulty → terrain scale
        diff = params_dict.get("difficulty", 0.5)
        if hasattr(cfg, "terrain") and hasattr(cfg.terrain, "difficulty_scale"):
            cfg.terrain.difficulty_scale = diff

    # ------ 生成课程 ------
    def generate_and_apply(self):
        """返回 (params_dict, is_easy, is_warmup, source)"""
        # 热身
        if not self.warmup_done and self.warmup_count < self.warmup_iters:
            self.warmup_count += 1
            if self.warmup_count >= self.warmup_iters:
                self.warmup_done = True
            params = self._random_easy_params()
            params["difficulty"] = np.random.uniform(0.1, 0.5)
            params["terrain_type"] = np.random.randint(0, self.terrain_gen.num_terrain_types)
            self._apply_params(params)
            return params, False, True, "WARMUP"

        # 概率下界
        if np.random.random() < self.current_easy_prob:
            self.easy_count += 1
            params = self._random_easy_params()
            self._apply_params(params)
            self._decay_easy()
            return params, True, False, "EASY"

        # 生成器
        self.gen_net.eval()
        for _ in range(5):
            with torch.no_grad():
                cond = self.input_builder.build(batch_size=1)
                t_type, p_vec, lp = self.gen_net(cond, deterministic=False)
            params = self.gen_net.to_params_dict(p_vec)
            if isinstance(params, list):
                params = params[0]
            params["terrain_type"] = t_type.item()
            ok, _ = self.feasibility_filter.is_feasible(params)
            if ok:
                self._apply_params(params)
                self._decay_easy()
                self._last_lp = lp
                return params, False, False, "GENERATOR"

        # fallback
        params = self._random_easy_params()
        self._apply_params(params)
        self._decay_easy()
        self._last_lp = None
        return params, True, False, "FALLBACK"

    def _random_easy_params(self):
        return {
            "terrain_type": 0,
            "difficulty": np.random.uniform(0.05, 0.15),
            "friction": np.random.uniform(0.9, 1.1),
            "push_magnitude": np.random.uniform(0.0, 0.1),
            "added_mass": np.random.uniform(-0.2, 0.2),
        }

    def _decay_easy(self):
        self.current_easy_prob = max(self.min_easy_prob,
                                     self.current_easy_prob * self.easy_decay)

    # ------ 计算遗憾 & 更新 ------
    def compute_and_update(self, params, is_easy, is_warmup, source):
        self.iteration += 1
        sol_r, sol_traj = self._evaluate_policy(
            self.ppo_runner.alg.actor_critic, num_steps=24
        )
        if is_warmup:
            return {"solver_reward": sol_r, "regret": 0, "gen_loss": 0,
                    "is_novel": True, "accept_rate": 0, "source": source,
                    "easy_prob": self.current_easy_prob, **params}

        if self.antagonist is None:
            self._init_antagonist()
        ant_r, _ = self._evaluate_policy(self.antagonist, num_steps=24)
        regret = ant_r - sol_r

        gen_loss = 0.0
        is_novel = True
        if not is_easy and source != "FALLBACK":
            is_novel, _ = self._check_novelty(sol_traj)
            if is_novel:
                self.accepted += 1
                self._add_to_buffer(sol_traj)
                gen_loss = self._update_generator(regret)
            else:
                self.rejected += 1

        self._update_antagonist()
        self.input_builder.update(sol_r, ant_r, params.get("difficulty", 0.5))
        ar = self.accepted / max(self.accepted + self.rejected, 1)

        return {"solver_reward": sol_r, "antagonist_reward": ant_r,
                "regret": regret, "gen_loss": gen_loss,
                "is_novel": is_novel, "accept_rate": ar, "source": source,
                "easy_prob": self.current_easy_prob, **params}

    def _update_generator(self, regret):
        self.gen_net.train()
        cond = self.input_builder.build(batch_size=1)
        _, _, lp = self.gen_net(cond, deterministic=False)
        loss = -regret * lp.mean()
        entropy = self.gen_net.get_entropy(cond).mean()
        loss = loss - 0.02 * entropy
        self.gen_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gen_net.parameters(), 0.5)
        self.gen_optimizer.step()
        return loss.item()

    # ------ 保存/加载 ------
    def save(self, path=None):
        if path is None:
            path = os.path.join(self.log_dir, f"adversarial_state.pt")
        torch.save({
            "iteration": self.iteration,
            "gen_net": self.gen_net.state_dict(),
            "gen_opt": self.gen_optimizer.state_dict(),
            "accepted": self.accepted,
            "rejected": self.rejected,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.iteration = ckpt["iteration"]
        self.gen_net.load_state_dict(ckpt["gen_net"])
        self.gen_optimizer.load_state_dict(ckpt["gen_opt"])
        self.accepted = ckpt.get("accepted", 0)
        self.rejected = ckpt.get("rejected", 0)


# ============================================================
# DR Baseline Runner (无对抗，纯 PPO + Domain Randomization)
# ============================================================

class DRBaselineRunner:
    """DR 方法不需要生成器，直接用 PPO 训练"""

    def __init__(self, env, ppo_runner, log_dir):
        self.env = env
        self.ppo_runner = ppo_runner
        self.log_dir = log_dir
        self.iteration = 0

    def step_curriculum(self):
        """DR 没有课程，返回空统计"""
        self.iteration += 1
        return {"source": "DR", "solver_reward": 0, "regret": 0}

    def save(self, path=None):
        pass  # DR 没有额外状态


# ============================================================
# 主训练循环
# ============================================================

def run_one_experiment(args):
    """运行单个实验"""
    method_cfg = METHODS[args.method]
    gen = get_generator(args.generator)
    cfg = TRAIN_CFG.copy()

    experiment_id = f"{args.method}-{args.generator}-seed{args.seed}"
    print("=" * 70)
    print(f"  Experiment: {experiment_id}")
    print(f"  Method: {method_cfg['name']}")
    print(f"  Generator: {args.generator} ({gen.name}, {gen.param_dim}D, {gen.num_terrain_types} types)")
    print(f"  Seed: {args.seed}")
    print(f"  Iterations: {args.max_iterations}")
    print("=" * 70)

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 日志目录
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    log_dir = os.path.join("logs", "experiments", f"{experiment_id}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # 保存实验配置
    with open(os.path.join(log_dir, "experiment_config.json"), "w") as f:
        json.dump({
            "method": args.method, "generator": args.generator,
            "seed": args.seed, "max_iterations": args.max_iterations,
            "train_cfg": cfg, "method_cfg": method_cfg,
        }, f, indent=2, default=str)

    # 上传器
    uploader = None
    if args.upload:
        uploader = ResultUploader(experiment_id=experiment_id)

    # 创建 IsaacGym 环境
    task_name = f"go2_{method_cfg['task_suffix']}"
    from isaacgym import gymapi

    class EnvArgs:
        task = task_name
        headless = args.headless
        num_envs = cfg["num_envs"]
        sim_device = args.device
        rl_device = args.device
        physics_engine = gymapi.SIM_PHYSX
        use_gpu = True
        subscenes = 0
        num_threads = 0
        use_gpu_pipeline = True
        seed = args.seed
        resume = False
        experiment_name = experiment_id
        run_name = experiment_id
        load_run = None
        checkpoint = None
        max_iterations = args.max_iterations

    env_args = EnvArgs()
    env, env_cfg = task_registry.make_env(name=task_name, args=env_args)

    # 创建 PPO Runner
    from rsl_rl.runners import OnPolicyRunner
    _, train_cfg = task_registry.get_cfgs(name=task_name)
    train_cfg.runner.max_iterations = args.max_iterations
    train_cfg_dict = class_to_dict(train_cfg)
    ppo_runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.device)

    # 创建训练器 (根据方法)
    if args.method == "dr":
        trainer = DRBaselineRunner(env, ppo_runner, log_dir)
        use_adversarial = False
    else:
        use_novelty = method_cfg["use_novelty"]
        cfg["use_novelty"] = use_novelty
        trainer = PluggableAdversarialTrainer(
            env=env, ppo_runner=ppo_runner,
            terrain_generator=gen, device=args.device,
            cfg=cfg, log_dir=log_dir,
        )
        use_adversarial = True

    # ============ 训练循环 ============
    device = args.device
    obs = env.get_observations()
    priv_obs = env.get_privileged_observations()
    critic_obs = priv_obs if priv_obs is not None else obs
    obs, critic_obs = obs.to(device), critic_obs.to(device)
    ppo_runner.alg.actor_critic.train()

    # 随机初始化 episode 长度
    env.episode_length_buf = torch.randint_like(
        env.episode_length_buf, high=int(env.max_episode_length)
    )

    stats_log = []
    start_time = time.time()
    curriculum_freq = cfg["curriculum_update_freq"]

    for it in range(args.max_iterations):
        # --- 课程更新 ---
        cur_stats = {}
        if use_adversarial and it % curriculum_freq == 0:
            params, is_easy, is_warmup, source = trainer.generate_and_apply()
            cur_stats = trainer.compute_and_update(params, is_easy, is_warmup, source)

        # --- PPO rollout ---
        with torch.inference_mode():
            for _ in range(ppo_runner.num_steps_per_env):
                actions = ppo_runner.alg.act(obs, critic_obs)
                obs, priv_obs, rewards, dones, infos = env.step(actions)
                critic_obs = priv_obs if priv_obs is not None else obs
                obs, critic_obs = obs.to(device), critic_obs.to(device)
                ppo_runner.alg.process_env_step(rewards, dones, infos)
            ppo_runner.alg.compute_returns(critic_obs)

        # --- PPO 更新 ---
        v_loss, s_loss = ppo_runner.alg.update()

        # --- 日志 ---
        ep = env.extras.get("episode", {})
        mean_rew = ep.get("rew_tracking_lin_vel", 0)
        entry = {"iter": it, "reward": mean_rew, "v_loss": v_loss,
                 "s_loss": s_loss, **cur_stats}
        stats_log.append(entry)

        if it % 50 == 0:
            elapsed = time.time() - start_time
            eta = (args.max_iterations - it) * elapsed / max(it, 1)
            src = cur_stats.get("source", "DR")
            reg = cur_stats.get("regret", 0)
            ar = cur_stats.get("accept_rate", 0)
            print(f"[{it}/{args.max_iterations}] rew={mean_rew:.3f} "
                  f"src={src} regret={reg:.3f} AR={ar:.1%} "
                  f"time={elapsed:.0f}s ETA={eta/60:.1f}m")

        # --- 保存 checkpoint ---
        if it > 0 and it % cfg["save_interval"] == 0:
            ppo_runner.save(os.path.join(log_dir, f"model_{it}.pt"))
            if use_adversarial:
                trainer.save()
            # 保存训练统计
            with open(os.path.join(log_dir, "training_stats.json"), "w") as f:
                json.dump(stats_log, f)

        # --- 上传 ---
        if uploader and it > 0 and it % cfg["upload_interval"] == 0:
            ppo_runner.save(os.path.join(log_dir, f"model_{it}.pt"))
            with open(os.path.join(log_dir, "training_stats.json"), "w") as f:
                json.dump(stats_log, f)
            uploader.upload_checkpoint(log_dir, it)

    # ============ 训练结束 ============
    ppo_runner.save(os.path.join(log_dir, "model_final.pt"))
    if use_adversarial:
        trainer.save(os.path.join(log_dir, "adversarial_final.pt"))
    with open(os.path.join(log_dir, "training_stats.json"), "w") as f:
        json.dump(stats_log, f)

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  {experiment_id} DONE — {total_time/60:.1f} min")
    print(f"  Saved to: {log_dir}")
    print(f"{'='*70}")

    if uploader:
        uploader.upload_final(log_dir)

    return log_dir


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AT-PC 跨生成器实验")
    parser.add_argument("--generator", type=str, required=True,
                        choices=["G1", "G2", "G3", "G4"])
    parser.add_argument("--method", type=str, required=True,
                        choices=["dr", "paired", "atpc"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int,
                        default=TRAIN_CFG["max_iterations"])
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--upload", action="store_true", default=True,
                        help="上传结果到服务器 (默认开启)")
    parser.add_argument("--no_upload", action="store_true",
                        help="禁用上传")
    args = parser.parse_args()
    if args.no_upload:
        args.upload = False
    run_one_experiment(args)


if __name__ == "__main__":
    main()
