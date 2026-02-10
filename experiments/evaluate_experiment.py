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
        --upload
"""
import os
import sys
import json
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import isaacgym

import torch
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict

from experiments.config import METHODS, COMMON_TEST_TERRAINS
from experiments.generators import get_generator
from experiments.uploader import ResultUploader


class UnifiedEvaluator:
    """在指定测试地形上评估策略
    
    注意：由于 IsaacGym 地形在环境创建时生成，无法在运行时切换地形类型。
    因此评估主要测试 domain randomization 参数（摩擦、推力、负载）的影响。
    """

    def __init__(self, env, device="cuda:0"):
        self.env = env
        self.device = device
        self.num_envs = env.num_envs
        # 获取最大 episode 长度用于判断成功
        self.max_ep_len = int(env.max_episode_length)
        # 成功阈值：完成 80% 的 episode 长度算成功
        self.success_threshold = int(self.max_ep_len * 0.8)

    def _apply_domain_params(self, terrain_params):
        """应用 domain randomization 参数到环境"""
        # 摩擦系数
        friction = terrain_params.get("friction", 1.0)
        if hasattr(self.env, "friction_coeffs"):
            self.env.friction_coeffs[:] = friction
        
        # 推力设置
        push = terrain_params.get("push_magnitude", 0.0)
        if hasattr(self.env.cfg, "domain_rand"):
            self.env.cfg.domain_rand.max_push_vel_xy = push
            self.env.cfg.domain_rand.push_robots = push > 0.01
        
        # 附加质量 - 需要在 reset 时生效
        mass = terrain_params.get("added_mass", 0.0)
        if hasattr(self.env.cfg, "domain_rand"):
            self.env.cfg.domain_rand.added_mass_range = [mass, mass]
            self.env.cfg.domain_rand.randomize_base_mass = abs(mass) > 0.01

    def evaluate_on_terrain(self, policy, terrain_params, num_episodes=50,
                            max_steps=2000):
        """在单个地形配置上评估
        
        Args:
            policy: 要评估的策略
            terrain_params: 地形参数字典
            num_episodes: 评估的 episode 数量
            max_steps: 最大仿真步数（防止死循环）
        
        Returns:
            dict: 包含 mean_reward, success_rate, mean_ep_len 等指标
        """
        # 应用参数
        self._apply_domain_params(terrain_params)
        
        # 重置环境以应用新参数
        obs = self.env.reset()[0] if isinstance(self.env.reset(), tuple) else self.env.reset()
        if hasattr(obs, 'to'):
            obs = obs.to(self.device)

        # 评估
        policy.eval()
        ep_rewards = []
        ep_lengths = []
        successes = 0
        completed = 0

        ep_rew = torch.zeros(self.num_envs, device=self.device)
        ep_len = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        step = 0

        while completed < num_episodes and step < max_steps:
            with torch.no_grad():
                actions = policy.act_inference(obs)
            obs, _, rewards, dones, infos = self.env.step(actions)
            if hasattr(obs, 'to'):
                obs = obs.to(self.device)
            
            ep_rew += rewards
            ep_len += 1
            step += 1

            # 处理完成的 episodes
            done_idx = dones.nonzero(as_tuple=False).flatten()
            for idx in done_idx:
                i = idx.item()
                ep_rewards.append(ep_rew[i].item())
                ep_lengths.append(ep_len[i].item())
                
                # 成功判定：timeout 或者 episode 长度超过阈值
                is_timeout = "time_outs" in infos and infos["time_outs"][i]
                is_long_enough = ep_len[i].item() >= self.success_threshold
                if is_timeout or is_long_enough:
                    successes += 1
                
                # 重置计数器
                ep_rew[i] = 0
                ep_len[i] = 0
                completed += 1
                
                if completed >= num_episodes:
                    break

        if completed == 0:
            return {
                "mean_reward": 0.0, "std_reward": 0.0,
                "success_rate": 0.0, "mean_ep_len": 0.0,
                "num_episodes": 0
            }

        return {
            "mean_reward": float(np.mean(ep_rewards)),
            "std_reward": float(np.std(ep_rewards)),
            "success_rate": float(successes / completed),
            "mean_ep_len": float(np.mean(ep_lengths)),
            "std_ep_len": float(np.std(ep_lengths)),
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
    parser.add_argument("--upload", action="store_true", default=False)
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
    print(f"  (Note: terrain_type/difficulty changes require env rebuild, testing domain_rand params)")
    print()
    
    for terrain_name, terrain_params in COMMON_TEST_TERRAINS.items():
        print(f"  {terrain_name}...", end=" ", flush=True)
        res = evaluator.evaluate_on_terrain(
            policy, terrain_params, num_episodes=args.num_episodes
        )
        results[terrain_name] = res
        print(f"reward={res['mean_reward']:.2f}, ep_len={res['mean_ep_len']:.0f}, success={res['success_rate']:.1%}")

    # 汇总
    all_rewards = [r["mean_reward"] for r in results.values()]
    all_success = [r["success_rate"] for r in results.values()]
    all_ep_len = [r["mean_ep_len"] for r in results.values()]
    
    summary = {
        "experiment": f"{args.method}-{args.generator}",
        "checkpoint": ckpt_path,
        "mean_reward_across_terrains": float(np.mean(all_rewards)),
        "std_reward_across_terrains": float(np.std(all_rewards)),
        "mean_success_across_terrains": float(np.mean(all_success)),
        "mean_ep_len_across_terrains": float(np.mean(all_ep_len)),
        "per_terrain": results,
    }

    # 保存
    out_path = os.path.join(args.log_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print(f"  Mean reward: {summary['mean_reward_across_terrains']:.3f}")
    print(f"  Mean ep_len: {summary['mean_ep_len_across_terrains']:.0f}")
    print(f"  Mean success: {summary['mean_success_across_terrains']:.1%}")

    # 上传
    if args.upload:
        uploader = ResultUploader(
            experiment_id=f"eval-{args.method}-{args.generator}",
        )
        uploader.upload_final(args.log_dir)


if __name__ == "__main__":
    main()
