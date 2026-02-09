#!/usr/bin/env python3
"""
将帧图片合成为视频
"""

import os
import glob
import argparse
from pathlib import Path

def frames_to_video(frames_dir, output_path, fps=30):
    """使用 imageio 将帧图片合成视频"""
    try:
        import imageio
        from PIL import Image
        import numpy as np
    except ImportError:
        print("Please install: pip install imageio imageio-ffmpeg pillow")
        return False
    
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frames:
        print(f"No frames found in {frames_dir}")
        return False
    
    print(f"Creating video from {len(frames)} frames...")
    
    # 使用 mpeg4 编码器（广泛支持）
    writer = imageio.get_writer(
        output_path, 
        fps=fps, 
        codec='mpeg4',
        quality=8,  # 0-10, higher is better
        macro_block_size=1  # 避免尺寸调整
    )
    
    for i, frame_path in enumerate(frames):
        img = np.array(Image.open(frame_path).convert('RGB'))
        writer.append_data(img)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(frames)} frames")
    
    writer.close()
    print(f"Video saved to {output_path}")
    return True

def process_all_videos(base_dir, output_dir, fps=30):
    """处理所有帧目录"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有 *_frames 目录
    frame_dirs = glob.glob(os.path.join(base_dir, "*_frames"))
    
    if not frame_dirs:
        print(f"No frame directories found in {base_dir}")
        return
    
    print(f"Found {len(frame_dirs)} frame directories")
    
    for frames_dir in sorted(frame_dirs):
        terrain_name = os.path.basename(frames_dir).replace("_frames", "")
        output_path = os.path.join(output_dir, f"{terrain_name}.mp4")
        
        print(f"\nProcessing: {terrain_name}")
        frames_to_video(frames_dir, output_path, fps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="remidi_v2", 
                       choices=["remidi_v2", "paired_v2", "all"])
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    
    base_path = "results/videos"
    
    if args.model == "all":
        models = ["remidi_v2", "paired_v2"]
    else:
        models = [args.model]
    
    for model in models:
        frames_base = os.path.join(base_path, model)
        output_dir = os.path.join(base_path, f"{model}_mp4")
        
        if os.path.exists(frames_base):
            print(f"\n{'='*50}")
            print(f"Processing {model}")
            print(f"{'='*50}")
            process_all_videos(frames_base, output_dir, fps=args.fps)
        else:
            print(f"Directory not found: {frames_base}")

if __name__ == "__main__":
    main()
