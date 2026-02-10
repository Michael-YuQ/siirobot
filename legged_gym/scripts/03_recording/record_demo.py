"""
录制演示视频
展示机器狗在不同物理条件下的表现

注意：地形几何是固定的（在环境创建时生成）
测试的是不同的物理参数：摩擦、扰动、负载

使用方法:
    python scripts/record_demo.py --model=remidi_v2
"""
import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isaacgym
from isaacgym import gymapi

import torch
from PIL import Image, ImageDraw, ImageFont

from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict


# 测试场景 - 物理参数变化
TEST_SCENARIOS = [
    {
        'name': 'normal',
        'description': 'Normal Conditions',
        'friction': 1.0,
        'push_magnitude': 0.0,
        'added_mass': 0.0,
    },
    {
        'name': 'low_friction',
        'description': 'Low Friction (Slippery)',
        'friction': 0.2,
        'push_magnitude': 0.0,
        'added_mass': 0.0,
    },
    {
        'name': 'external_push',
        'description': 'External Push (2.0 m/s)',
        'friction': 1.0,
        'push_magnitude': 2.0,
        'added_mass': 0.0,
    },
    {
        'name': 'heavy_load',
        'description': 'Heavy Load (+3kg)',
        'friction': 1.0,
        'push_magnitude': 0.0,
        'added_mass': 3.0,
    },
    {
        'name': 'combined_challenge',
        'description': 'Combined: Low Friction + Push + Load',
        'friction': 0.4,
        'push_magnitude': 1.5,
        'added_mass': 2.0,
    },
]

MODEL_CONFIGS = {
    'remidi_v2': {
        'task': 'go2_remidi',
        'model_path': 'logs/go2_remidi_v2/Jan06_15-58-16_remidi_v2/model_final.pt',
        'name': 'ReMiDi v2',
    },
    'paired_v2': {
        'task': 'go2_paired',
        'model_path': 'logs/go2_paired_v2/Jan07_18-08-08_paired_v2/model_final.pt',
        'name': 'PAIRED v2',
    },
}


def apply_physics_params(env, scenario):
    """应用物理参数"""
    cfg = env.cfg
    
    # 摩擦系数
    friction = scenario['friction']
    cfg.domain_rand.friction_range = [max(0.1, friction - 0.05), min(2.0, friction + 0.05)]
    
    # 外力扰动
    push_mag = scenario['push_magnitude']
    cfg.domain_rand.max_push_vel_xy = push_mag
    if push_mag > 0.1:
        cfg.domain_rand.push_robots = True
        cfg.domain_rand.push_interval_s = 3  # 更频繁的扰动
    else:
        cfg.domain_rand.push_robots = False
    
    # 附加质量
    added_mass = scenario['added_mass']
    cfg.domain_rand.added_mass_range = [added_mass - 0.2, added_mass + 0.2]
    cfg.domain_rand.randomize_base_mass = True
    
    print(f"  Applied: friction={friction}, push={push_mag}, mass=+{added_mass}kg")


def add_text_overlay(frame_path, text, output_path):
    """在帧上添加文字说明"""
    try:
        img = Image.open(frame_path)
        draw = ImageDraw.Draw(img)
        
        # 尝试使用系统字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        except:
            font = ImageFont.load_default()
        
        # 添加半透明背景
        text_bbox = draw.textbbox((10, 10), text, font=font)
        padding = 10
        draw.rectangle(
            [text_bbox[0] - padding, text_bbox[1] - padding, 
             text_bbox[2] + padding, text_bbox[3] + padding],
            fill=(0, 0, 0, 180)
        )
        
        # 添加文字
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)
        
        img.save(output_path)
    except Exception as e:
        # 如果添加文字失败，直接复制原图
        import shutil
        shutil.copy(frame_path, output_path)


def record_scenario(env, policy, scenario, output_dir, num_steps=400, device='cuda:0'):
    """录制单个场景"""
    
    scenario_name = scenario['name']
    description = scenario['description']
    
    print(f"\n{'='*50}")
    print(f"Recording: {scenario_name}")
    print(f"Description: {description}")
    print(f"{'='*50}")
    
    # 应用物理参数
    apply_physics_params(env, scenario)
    
    gym = env.gym
    sim = env.sim
    viewer = env.viewer
    
    if viewer is None:
        print("Error: No viewer")
        return None
    
    # 设置相机
    cam_pos = gymapi.Vec3(2.5, 2.5, 1.5)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.4)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    
    # 创建帧目录
    frames_dir = os.path.join(output_dir, f'{scenario_name}_frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # 重置
    obs = env.get_observations()
    policy.eval()
    
    frame_count = 0
    
    for step in range(num_steps):
        with torch.no_grad():
            actions = policy.act_inference(obs)
        
        obs, _, rewards, dones, infos = env.step(actions)
        
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        
        # 每 2 步截图
        if step % 2 == 0:
            frame_path = os.path.join(frames_dir, f'frame_{frame_count:04d}.png')
            gym.write_viewer_image_to_file(viewer, frame_path)
            
            # 添加文字说明
            labeled_path = os.path.join(frames_dir, f'labeled_{frame_count:04d}.png')
            add_text_overlay(frame_path, description, labeled_path)
            os.remove(frame_path)
            os.rename(labeled_path, frame_path)
            
            frame_count += 1
        
        if step % 100 == 0:
            print(f"  Step {step}/{num_steps}")
        
        if gym.query_viewer_has_closed(viewer):
            break
    
    print(f"  Captured {frame_count} frames")
    
    # 合成视频
    video_path = os.path.join(output_dir, f'{scenario_name}.mp4')
    frame_pattern = os.path.join(frames_dir, 'frame_%04d.png')
    
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-framerate', '30',
            '-i', frame_pattern,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '20',
            video_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            import shutil
            shutil.rmtree(frames_dir)
            print(f"  Video: {video_path}")
            return video_path
        else:
            print(f"  ffmpeg failed, frames in: {frames_dir}")
            return frames_dir
    except FileNotFoundError:
        print(f"  ffmpeg not found, frames in: {frames_dir}")
        return frames_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='remidi_v2', choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--scenario', type=str, default='all')
    parser.add_argument('--num_envs', type=int, default=16)
    parser.add_argument('--num_steps', type=int, default=400)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    model_config = MODEL_CONFIGS[args.model]
    
    print("=" * 60)
    print(f"Demo Recording: {model_config['name']}")
    print("=" * 60)
    print("\nNote: Terrain geometry is fixed at environment creation.")
    print("These demos show robot behavior under different PHYSICS conditions:")
    print("  - Friction changes")
    print("  - External push disturbances")
    print("  - Added payload mass")
    print("=" * 60)
    
    if not os.path.exists(model_config['model_path']):
        print(f"Error: Model not found: {model_config['model_path']}")
        return
    
    # 创建环境
    class Args:
        task = model_config['task']
        headless = False
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
        experiment_name = None
        run_name = None
        load_run = None
        checkpoint = None
        max_iterations = 1000
    
    lg_args = Args()
    
    print("\nCreating environment...")
    env, env_cfg = task_registry.make_env(name=model_config['task'], args=lg_args)
    
    print(f"Loading model: {model_config['model_path']}")
    _, train_cfg = task_registry.get_cfgs(name=model_config['task'])
    
    from rsl_rl.runners import OnPolicyRunner
    train_cfg_dict = class_to_dict(train_cfg)
    ppo_runner = OnPolicyRunner(env, train_cfg_dict, log_dir=None, device=args.device)
    ppo_runner.load(model_config['model_path'])
    policy = ppo_runner.alg.actor_critic
    
    print("Model loaded!")
    
    output_dir = f"results/videos/{args.model}_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.scenario == 'all':
        scenarios = TEST_SCENARIOS
    else:
        scenarios = [s for s in TEST_SCENARIOS if s['name'] == args.scenario]
    
    print(f"\nRecording {len(scenarios)} scenario(s)...")
    
    for scenario in scenarios:
        record_scenario(env, policy, scenario, output_dir, args.num_steps, args.device)
    
    print(f"\n{'='*60}")
    print("Recording Complete!")
    print(f"Videos: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
