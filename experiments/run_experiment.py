"""
Unified experiment entry point - one script runs one experiment.

Usage:
    python -m experiments.run_experiment \
        --generator G1 --method atpc --seed 42 \
        --max_iterations 2000 --headless
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
        self.gen_net.reset_lstm(batch_size=1)
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
        self._eval_steps = c.get("eval_steps", 100)
        self.iteration = 0
        self.accepted = 0
        self.rejected = 0

    def _init_antagonist(self):
        ac = self.ppo_runner.alg.actor_critic
        # Detach all params, deepcopy, then re-attach
        ac.eval()
        sd = {k: v.clone().detach() for k, v in ac.state_dict().items()}
        # Build a new model with same architecture via save/load trick
        import io
        buf = io.BytesIO()
        torch.save(ac, buf)
        buf.seek(0)
        self.antagonist = torch.load(buf, map_location=self.device)
        self.antagonist.eval()

    def _update_antagonist(self):
        if self.antagonist is None:
            self._init_antagonist()
            return
        with torch.no_grad():
            for pa, ps in zip(self.antagonist.parameters(),
                              self.ppo_runner.alg.actor_critic.parameters()):
                pa.data.mul_(self.antagonist_ema).add_((1 - self.antagonist_ema) * ps.data)

    def _evaluate_policy(self, policy, num_steps=None):
        if num_steps is None:
            num_steps = self._eval_steps
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
        return sims.max().item() < self.novelty_threshold, sims.max().item()

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

    def generate_and_apply(self, current_iter=0):
        if not self.warmup_done and current_iter < self.warmup_iters:
            self.warmup_count += 1
            if current_iter + 50 >= self.warmup_iters:  # next call will be past warmup
                self.warmup_done = True
            params = self._random_easy_params()
            params["difficulty"] = float(np.random.uniform(0.1, 0.5))
            params["terrain_type"] = int(np.random.randint(0, self.terrain_gen.num_terrain_types))
            self._apply_params(params)
            return params, False, True, "WARMUP"
        if np.random.random() < self.current_easy_prob:
            self.easy_count += 1
            params = self._random_easy_params()
            self._apply_params(params)
            self._decay_easy()
            return params, True, False, "EASY"
        self.gen_net.eval()
        for _ in range(5):
            with torch.no_grad():
                cond = self.input_builder.build(batch_size=1)
                t_type, p_vec, lp = self.gen_net(cond, deterministic=False)
            params = self.gen_net.to_params_dict(p_vec)
            if isinstance(params, list):
                params = params[0]
            params["terrain_type"] = int(t_type.item())
            ok, _ = self.feasibility_filter.is_feasible(params)
            if ok:
                self._apply_params(params)
                self._decay_easy()
                self._last_lp = lp
                return params, False, False, "GENERATOR"
        params = self._random_easy_params()
        self._apply_params(params)
        self._decay_easy()
        self._last_lp = None
        return params, True, False, "FALLBACK"

    def _random_easy_params(self):
        return {
            "terrain_type": 0,
            "difficulty": float(np.random.uniform(0.05, 0.15)),
            "friction": float(np.random.uniform(0.9, 1.1)),
            "push_magnitude": float(np.random.uniform(0.0, 0.1)),
            "added_mass": float(np.random.uniform(-0.2, 0.2)),
        }

    def _decay_easy(self):
        self.current_easy_prob = max(self.min_easy_prob,
                                     self.current_easy_prob * self.easy_decay)

    def compute_and_update(self, params, is_easy, is_warmup, source):
        self.iteration += 1
        sol_r, sol_traj = self._evaluate_policy(self.ppo_runner.alg.actor_critic)
        if is_warmup:
            return {"solver_reward": sol_r, "regret": 0, "gen_loss": 0,
                    "is_novel": True, "accept_rate": 0, "source": source,
                    "easy_prob": self.current_easy_prob, **params}
        if self.antagonist is None:
            self._init_antagonist()
        ant_r, _ = self._evaluate_policy(self.antagonist)
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

    def save(self, path=None):
        if path is None:
            path = os.path.join(self.log_dir, "adversarial_state.pt")
        torch.save({"iteration": self.iteration,
                     "gen_net": self.gen_net.state_dict(),
                     "gen_opt": self.gen_optimizer.state_dict(),
                     "accepted": self.accepted, "rejected": self.rejected}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.iteration = ckpt["iteration"]
        self.gen_net.load_state_dict(ckpt["gen_net"])
        self.gen_optimizer.load_state_dict(ckpt["gen_opt"])
        self.accepted = ckpt.get("accepted", 0)
        self.rejected = ckpt.get("rejected", 0)


class DRBaselineRunner:
    def __init__(self, env, ppo_runner, log_dir):
        self.env = env
        self.ppo_runner = ppo_runner
        self.log_dir = log_dir
        self.iteration = 0

    def step_curriculum(self):
        self.iteration += 1
        return {"source": "DR", "solver_reward": 0, "regret": 0}

    def save(self, path=None):
        pass


def run_one_experiment(args):
    method_cfg = METHODS[args.method]
    gen = get_generator(args.generator)
    cfg = TRAIN_CFG.copy()
    experiment_id = f"{args.method}-{args.generator}-seed{args.seed}"
    print("=" * 70)
    print(f"  Experiment: {experiment_id}")
    print(f"  Method: {method_cfg['name']}")
    print(f"  Generator: {args.generator} ({gen.name}, {gen.param_dim}D, {gen.num_terrain_types} types)")
    print(f"  Seed: {args.seed}, Iterations: {args.max_iterations}")
    print("=" * 70)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    log_dir = os.path.join("logs", "experiments", f"{experiment_id}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "experiment_config.json"), "w") as f:
        json.dump({"method": args.method, "generator": args.generator,
                    "seed": args.seed, "max_iterations": args.max_iterations,
                    "train_cfg": cfg, "method_cfg": method_cfg}, f, indent=2, default=str)
    uploader = None
    if args.upload:
        uploader = ResultUploader(experiment_id=experiment_id)
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

    env, env_cfg = task_registry.make_env(name=task_name, args=EnvArgs())
    from rsl_rl.runners import OnPolicyRunner
    _, train_cfg = task_registry.get_cfgs(name=task_name)
    train_cfg.runner.max_iterations = args.max_iterations
    # Set num_steps_per_env BEFORE creating runner (rollout buffer is pre-allocated)
    if "num_steps_per_env" in cfg:
        train_cfg.runner.num_steps_per_env = cfg["num_steps_per_env"]
    train_cfg_dict = class_to_dict(train_cfg)
    ppo_runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.device)
    if args.method == "dr":
        trainer = DRBaselineRunner(env, ppo_runner, log_dir)
        use_adversarial = False
    else:
        use_novelty = method_cfg["use_novelty"]
        cfg["use_novelty"] = use_novelty
        trainer = PluggableAdversarialTrainer(
            env=env, ppo_runner=ppo_runner, terrain_generator=gen,
            device=args.device, cfg=cfg, log_dir=log_dir)
        use_adversarial = True
    device = args.device
    obs = env.get_observations()
    priv_obs = env.get_privileged_observations()
    critic_obs = priv_obs if priv_obs is not None else obs
    obs, critic_obs = obs.to(device), critic_obs.to(device)
    ppo_runner.alg.actor_critic.train()
    env.episode_length_buf = torch.randint_like(
        env.episode_length_buf, high=int(env.max_episode_length))
    stats_log = []
    start_time = time.time()
    # All methods use same curriculum_freq now (50)
    curriculum_freq = cfg["curriculum_update_freq"]
    for it in range(args.max_iterations):
        cur_stats = {}
        if use_adversarial and it % curriculum_freq == 0:
            params, is_easy, is_warmup, source = trainer.generate_and_apply(current_iter=it)
            cur_stats = trainer.compute_and_update(params, is_easy, is_warmup, source)
        # DR baseline: evaluate solver periodically for fair comparison
        if not use_adversarial and it % cfg.get("dr_eval_freq", 50) == 0:
            eval_steps = cfg.get("eval_steps", 100)
            with torch.no_grad():
                policy = ppo_runner.alg.actor_critic
                policy.eval()
                eval_obs = env.get_observations()
                eval_r = 0.0
                for _ in range(eval_steps):
                    act = policy.act_inference(eval_obs)
                    eval_obs, _, rew, _, _ = env.step(act)
                    eval_r += rew.mean().item()
                policy.train()
            cur_stats["solver_reward"] = eval_r / eval_steps
            cur_stats["source"] = "DR"
        with torch.inference_mode():
            for _ in range(ppo_runner.num_steps_per_env):
                actions = ppo_runner.alg.act(obs, critic_obs)
                obs, priv_obs, rewards, dones, infos = env.step(actions)
                critic_obs = priv_obs if priv_obs is not None else obs
                obs, critic_obs = obs.to(device), critic_obs.to(device)
                ppo_runner.alg.process_env_step(rewards, dones, infos)
            ppo_runner.alg.compute_returns(critic_obs)
        v_loss, s_loss = ppo_runner.alg.update()
        ep = env.extras.get("episode", {})
        # ---- reward logging (v3 fix) ----
        # env.extras["episode"] contains episode sums as Tensors, need .item()
        mean_rew = 0.0
        if ep:
            for key in ["rew_tracking_lin_vel", "reward", "r"]:
                if key in ep:
                    val = ep[key]
                    if hasattr(val, 'item'):
                        mean_rew = val.mean().item() if val.dim() > 0 else val.item()
                    elif isinstance(val, (int, float)):
                        mean_rew = float(val)
                    break
            if mean_rew == 0.0:
                for key, val in ep.items():
                    if key.startswith("rew_"):
                        if hasattr(val, 'item'):
                            mean_rew = val.mean().item() if val.dim() > 0 else val.item()
                        elif isinstance(val, (int, float)):
                            mean_rew = float(val)
                        if mean_rew != 0.0:
                            break
        # Direct env reward buffer — most reliable
        step_reward = env.rew_buf.mean().item() if hasattr(env, 'rew_buf') else 0.0
        if it == 0 and ep:
            print(f"[DEBUG] Available episode keys: {list(ep.keys())}")
            for k, v in ep.items():
                vtype = type(v).__name__
                vshape = v.shape if hasattr(v, 'shape') else 'scalar'
                print(f"  {k}: type={vtype} shape={vshape}")
        entry = {"iter": it, "reward": float(mean_rew), "step_reward": float(step_reward),
                 "v_loss": float(v_loss), "s_loss": float(s_loss), **cur_stats}
        stats_log.append(entry)
        if it % 50 == 0:
            elapsed = time.time() - start_time
            eta = (args.max_iterations - it) * elapsed / max(it, 1)
            src = cur_stats.get("source", "DR")
            reg = cur_stats.get("regret", 0)
            ar = cur_stats.get("accept_rate", 0)
            print(f"[{it}/{args.max_iterations}] rew={step_reward:.3f} "
                  f"src={src} regret={reg:.3f} AR={ar:.1%} "
                  f"time={elapsed:.0f}s ETA={eta/60:.1f}m")
            # Upload lightweight stats JSON every 50 iters
            if uploader:
                stats_path = os.path.join(log_dir, "training_stats.json")
                with open(stats_path, "w") as f:
                    json.dump(stats_log, f, default=_json_default)
                uploader.upload_file(stats_path)
        if it > 0 and it % cfg["save_interval"] == 0:
            ppo_runner.save(os.path.join(log_dir, f"model_{it}.pt"))
            if use_adversarial:
                trainer.save()
            with open(os.path.join(log_dir, "training_stats.json"), "w") as f:
                json.dump(stats_log, f, default=_json_default)
        if uploader and it > 0 and it % cfg["upload_interval"] == 0:
            ppo_runner.save(os.path.join(log_dir, f"model_{it}.pt"))
            with open(os.path.join(log_dir, "training_stats.json"), "w") as f:
                json.dump(stats_log, f, default=_json_default)
            uploader.upload_checkpoint(log_dir, it)
    ppo_runner.save(os.path.join(log_dir, "model_final.pt"))
    if use_adversarial:
        trainer.save(os.path.join(log_dir, "adversarial_final.pt"))
    with open(os.path.join(log_dir, "training_stats.json"), "w") as f:
        json.dump(stats_log, f, default=_json_default)
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  {experiment_id} DONE - {total_time/60:.1f} min")
    print(f"  Saved to: {log_dir}")
    print(f"{'='*70}")
    if uploader:
        uploader.upload_final(log_dir)
    return log_dir


def main():
    parser = argparse.ArgumentParser(description="AT-PC experiment")
    parser.add_argument("--generator", type=str, required=True, choices=["G1", "G2", "G3", "G4"])
    parser.add_argument("--method", type=str, required=True, choices=["dr", "paired", "atpc"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=TRAIN_CFG["max_iterations"])
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--upload", action="store_true", default=True)
    parser.add_argument("--no_upload", action="store_true")
    args = parser.parse_args()
    if args.no_upload:
        args.upload = False
    run_one_experiment(args)


if __name__ == "__main__":
    main()
