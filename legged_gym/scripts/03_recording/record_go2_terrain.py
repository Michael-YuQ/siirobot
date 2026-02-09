#!/usr/bin/env python3
"""
录制 Go2 在复杂地形行走的视频
使用 gym.write_viewer_image_to_file 方法
"""
import os
import sys
import shutil
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isaacgym
from isaacgym import gymapi

import torch
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict


def main():
    device = 'cuda:0'
    
    # 找到模型
    model_path = None
    log_dir = None
    for model_dir in ['logs/heroes_comparison_paired', 'logs/rough_go2']:
        if os.path.exists(model_dir):
            subdirs = sorted(os.listdir(model_dir), reverse=True)
            for subdir in subdirs:
                for model_name in ['model_500.pt', 'model_1000.pt']:
                    path = os.path.join(model_dir, subdir, model_name)
                    if os.path.exists(path):
                        model_path = path
                        log_dir = os.path.join(model_dir, subdir)
                        print(f"Using model: {model_path}")
                        break
                if model_path:
                    break
        if model_path:
            break
    
    if model_path is None:
        print("No model found!")
        return
    
    # 创建环境
    class Args:
        task = 'go2_remidi'
        headless = False
        num_envs = 4
        sim_device = device
        rl_device = device
        physics_engine = gymapi.SIM_PHYSX
        use_gpu = True
        subscenes = 0
        num_threads = 0
        use_gpu_pipeline = True
        seed = 42
        resume = True
        experiment_name = 'test'
        run_name = None
        load_run = None
        checkpoint = None
        max_iterations = 100
    
    args = Args()
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 4
    env_cfg.terrain.num_rows = 8
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True
    
    print("Creating environment...")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 加载策略
    print("Loading policy...")
    from rsl_rl.runners import OnPolicyRunner
    train_cfg_dict = class_to_dict(train_cfg)
    ppo_runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=device)
    ppo_runner.load(model_path)
    policy = ppo_runner.get_inference_policy(device=device)
    
    gym = env.gym
    sim = env.sim
    viewer = env.viewer
    
    if viewer is None:
        print("Error: No viewer!")
        return
    
    # 输出目录
    output_dir = 'results/videos/go2_camera'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    frames_dir = os.path.join(output_dir, f'frames_{timestamp}')
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"\n录制到: {output_dir}")
    print(f"帧目录: {frames_dir}")
    print("按 ESC 停止录制\n")
    
    # 设置相机
    cam_pos = gymapi.Vec3(3.0, 2.0, 1.5)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    
    obs = env.get_observations()
    frame_count = 0
    num_steps = 600  # 20秒 @ 30fps
    
    for step in range(num_steps):
        # 执行策略
        with torch.no_grad():
            actions = policy(obs.detach())
        
        obs, _, rewards, dones, infos = env.step(actions.detach())
        
        # 渲染
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        
        # 每隔一帧保存图片
        if step % 2 == 0:
            frame_path = os.path.join(frames_dir, f'frame_{frame_count:04d}.png')
            gym.write_viewer_image_to_file(viewer, frame_path)
            frame_count += 1
        
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}, Reward: {rewards.mean().item():.3f}, Frames: {frame_count}")
        
        # 检查退出
        if gym.query_viewer_has_closed(viewer):
            print("Viewer closed")
            break
    
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    
    # 使用 ffmpeg 合成视频
    print(f"\n合成视频 ({frame_count} 帧)...")
    video_path = os.path.join(output_dir, f'go2_terrain_{timestamp}.mp4')
    frame_pattern = os.path.join(frames_dir, 'frame_%04d.png')
    
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-framerate', '30',
            '-i', frame_pattern,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '23',
            video_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"视频已保存: {video_path}")
            # 清理帧文件
            shutil.rmtree(frames_dir)
        else:
            print(f"ffmpeg 错误: {result.stderr}")
            print(f"帧文件保留在: {frames_dir}")
    except FileNotFoundError:
        print("ffmpeg 未找到，帧文件保留在:", frames_dir)
        # 尝试用 imageio
        try:
            import imageio
            from PIL import Image
            frames = []
            for i in range(frame_count):
                img = Image.open(os.path.join(frames_dir, f'frame_{i:04d}.png'))
                frames.append(np.array(img))
            imageio.mimsave(video_path, frames, fps=30)
            print(f"视频已保存 (imageio): {video_path}")
            shutil.rmtree(frames_dir)
        except Exception as e:
            print(f"imageio 也失败了: {e}")
    
    print("\n完成!")


if __name__ == '__main__':
    main()
