"""
统一评估脚本 — 加载训练好的模型，在所有测试地形上评估

用法:
    # 评估单个实验
    python -m experiments.evaluate_experiment \
        --log_dir logs/experiments/atpc-G1-seed42_0210_120000 \
        --generator G1 --method atpc

    # 评估并上传
    python -m experiments.evaluate_experiment \
        --log_dir logs/experiments/atpc-G1-seed42_0210_120000 \
        --generator G1 --method atpc \
        --upload_url http://server:8080/upload
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import isaacgym

from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict

from experiments.config import METHODS, COMMON_TEST_TERRAINS
from experiments.generators import get_generator
from experiments.uploader import ResultUploader


class UnifiedEvaluator:
    """在指定测试地形上评估策略"""

    def __init__(self, env, device="cuda:0"):
        self.env = env
        self.device = device
        self.num_envs = env.num_envs

    def evaluate_on_terrain(self, policy, terrain_params, num_episodes=50,
                            max_steps=1000):
        """在单个地形配置上评估"""
        # 应用地形参数到环境
        cfg = self.env.cfg
        friction = terrain_params.get("friction", 1.0)
        cfg.domain_rand.friction_range = [friction, friction]
        if hasattr(self.env, "friction_coeffs"):
            self.env.friction_coeffs[:] = friction

        push = terrain_params.get("push_magnitude", 0.0)
        cfg.domain_rand.max_push_vel_xy = push
        cfg.domain_rand.push_robots = push > 0.1

        mass = terrain_params.get("added_mass", 0.0)
        cfg.domain_rand.added_mass_range = [mass, mass]

        diff = terrain_params.get("difficulty", 0.5)
        if hasattr(cfg.terrain, "difficulty_scale"):
            cfg.terrain.difficulty_scale = diff

        # 评估
        policy.eval()
        obs = self.env.get_observations()
        ep_rewards = []
        ep_lengths = []
        successes = 0
        completed = 0

        ep_rew = torch.zeros(self.num_envs, device=self.device)
        ep_len = torch.zeros(self.num_envs, device=self.device)
        step = 0

        while completed < num_episodes and step < max_steps * 5:
            with torch.no_grad():
                actions = policy.act_inference(obs)
            obs, _, rewards, dones, infos = self.env.step(actions)
            ep_rew += rewards
            ep_len += 1
            step += 1

            done_idx = dones.nonzero(as_tuple=False).flatten()
            for idx in done_idx:
                i = idx.item()
                ep_rewards.append(ep_rew[i].item())
                ep_lengths.append(ep_len[i].item())
                if "time_outs" in infos and infos["time_outs"][i]:
                    successes += 1
                ep_rew[i] = 0
                ep_len[i] = 0
                completed += 1
                if completed >= num_episodes:
                    break

        if completed == 0:
            return {"mean_reward": 0, "success_rate": 0, "mean_ep_len": 0,
                    "num_episodes": 0}

        return {
            "mean_reward": float(np.mean(ep_rewards)),
            "std_reward": float(np.std(ep_rewards)),
            "success_rate": successes / completed,
            "mean_ep_len": float(np.mean(ep_lengths)),
            "num_episodes": completed,
        }


def find_best_checkpoint(log_dir):
    """找到最佳 checkpoint (优先 model_final.pt)"""
    final = os.path.join(log_dir, "model_final.pt")
    if os.path.exists(final):
        return final
    # 找最大迭代数的 checkpoint
    pts = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not pts:
        raise FileNotFoundError(f"No checkpoint found in {log_dir}")
    pts.sort(key=lambda x: int(x.replace("model_", "").replace(".pt", ""))
             if x.replace("model_", "").replace(".pt", "").isdigit() else -1)
    return os.path.join(log_dir, pts[-1])


def main():
    parser = argparse.ArgumentParser(description="AT-PC 实验评估")
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--generator", type=str, required=True,
                        choices=["G1", "G2", "G3", "G4"])
    parser.add_argument("--method", type=str, required=True,
                        choices=["dr", "paired", "atpc"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="指定 checkpoint 路径，默认自动查找")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--upload", action="store_true", default=True)
    parser.add_argument("--no_upload", action="store_true")
    args = parser.parse_args()
    if args.no_upload:
        args.upload = False

    method_cfg = METHODS[args.method]
    task_name = f"go2_{method_cfg['task_suffix']}"

    # checkpoint
    ckpt_path = args.checkpoint or find_best_checkpoint(args.log_dir)
    print(f"Loading checkpoint: {ckpt_path}")

    # 创建环境
    from isaacgym import gymapi

    class EvalArgs:
        task = task_name
        headless = True
        num_envs = args.num_envs
        sim_device = args.device
        rl_device = args.device
        physics_engine = gymapi.SIM_PHYSX
        use_gpu = True
        subscenes = 0
        num_threads = 0
        use_gpu_pipeline = True
        seed = 42
        resume = False
        experiment_name = "eval"
        run_name = "eval"
        load_run = None
        checkpoint = None
        max_iterations = 1

    env, env_cfg = task_registry.make_env(name=task_name, args=EvalArgs())

    # 加载策略
    from rsl_rl.runners import OnPolicyRunner
    _, train_cfg = task_registry.get_cfgs(name=task_name)
    train_cfg_dict = class_to_dict(train_cfg)
    ppo_runner = OnPolicyRunner(env, train_cfg_dict, args.log_dir, device=args.device)
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.alg.actor_critic

    # 评估
    evaluator = UnifiedEvaluator(env, device=args.device)
    results = {}

    print(f"\nEvaluating {args.method}-{args.generator} on {len(COMMON_TEST_TERRAINS)} test terrains...")
    for terrain_name, terrain_params in COMMON_TEST_TERRAINS.items():
        print(f"  {terrain_name}...", end=" ", flush=True)
        res = evaluator.evaluate_on_terrain(
            policy, terrain_params, num_episodes=args.num_episodes
        )
        results[terrain_name] = res
        print(f"reward={res['mean_reward']:.2f}, success={res['success_rate']:.1%}")

    # 汇总
    all_rewards = [r["mean_reward"] for r in results.values()]
    all_success = [r["success_rate"] for r in results.values()]
    summary = {
        "experiment": f"{args.method}-{args.generator}",
        "checkpoint": ckpt_path,
        "mean_reward_across_terrains": float(np.mean(all_rewards)),
        "mean_success_across_terrains": float(np.mean(all_success)),
        "per_terrain": results,
    }

    # 保存
    out_path = os.path.join(args.log_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print(f"  Mean reward: {summary['mean_reward_across_terrains']:.3f}")
    print(f"  Mean success: {summary['mean_success_across_terrains']:.1%}")

    # 上传
    if args.upload:
        uploader = ResultUploader(
            experiment_id=f"eval-{args.method}-{args.generator}",
        )
        uploader.upload_final(args.log_dir)


if __name__ == "__main__":
    main()
