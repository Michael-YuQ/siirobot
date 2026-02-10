"""
实验配置 — 所有实验共享的常量和参数
"""

# ============================================================
# 服务器配置 (修改为你的服务器地址)
# ============================================================
UPLOAD_URL = "http://111.170.6.103:10005/upload"  # 网盘上传 endpoint
UPLOAD_BASE = "http://111.170.6.103:10005"          # 网盘 API 基地址
UPLOAD_ENABLED = False  # 默认关闭，命令行传入 --upload_url 时自动开启

# ============================================================
# 训练配置 (与论文第五章一致)
# ============================================================
TRAIN_CFG = {
    "num_envs": 512,
    "max_iterations": 2000,
    "save_interval": 200,       # 每 200 iter 保存 checkpoint
    "upload_interval": 500,     # 每 500 iter 上传到服务器
    "eval_interval": 500,       # 每 500 iter 做一次中间评估
    "curriculum_update_freq": 10,
    "ppo_lr": 1e-3,
    "generator_lr": 3e-4,
    "novelty_threshold": 0.7,
    "trajectory_buffer_size": 200,
    "warmup_iterations": 50,
    "easy_terrain_prob": 0.15,
    "easy_terrain_decay": 0.995,
    "min_easy_prob": 0.05,
}

# ============================================================
# 实验种子
# ============================================================
SEEDS = [42, 123, 456]

# ============================================================
# 方法定义
# ============================================================
METHODS = {
    "dr": {
        "name": "DR Baseline",
        "use_adversarial": False,
        "use_novelty": False,
        "task_suffix": "dr_baseline",
    },
    "paired": {
        "name": "PAIRED",
        "use_adversarial": True,
        "use_novelty": False,
        "task_suffix": "paired",
    },
    "atpc": {
        "name": "AT-PC",
        "use_adversarial": True,
        "use_novelty": True,
        "task_suffix": "remidi",
    },
}

# ============================================================
# 地形生成器定义
# ============================================================
GENERATORS = {
    "G1": {
        "name": "IsaacGym 原生 (7维)",
        "param_dim": 7,
        "terrain_types": 7,
        "description": "经典 legged_gym terrain_utils，difficulty 单标量控制",
    },
    "G2": {
        "name": "Isaac Lab 扩展 (15维)",
        "param_dim": 15,
        "terrain_types": 9,
        "description": "每种地形独立 (min,max) range 参数",
    },
    "G3": {
        "name": "参数化梅花桩 (21维)",
        "param_dim": 21,
        "terrain_types": 9,
        "description": "parametric_terrain.py，梅花桩参数最丰富",
    },
    "G4": {
        "name": "Trimesh 重复物体 (18维)",
        "param_dim": 18,
        "terrain_types": 6,
        "description": "圆柱/方块/锥体 + 轨道/浮环/坑",
    },
}

# ============================================================
# 通用测试地形 (与论文第五章一致)
# ============================================================
COMMON_TEST_TERRAINS = {
    "flat_easy": {
        "terrain_type": 0, "difficulty": 0.1,
        "friction": 1.0, "push_magnitude": 0.0, "added_mass": 0.0,
    },
    "steep_slope": {
        "terrain_type": 0, "difficulty": 0.9,
        "friction": 1.0, "push_magnitude": 0.0, "added_mass": 0.0,
    },
    "high_stairs": {
        "terrain_type": 2, "difficulty": 0.95,
        "friction": 1.0, "push_magnitude": 0.0, "added_mass": 0.0,
    },
    "slippery_surface": {
        "terrain_type": 0, "difficulty": 0.3,
        "friction": 0.2, "push_magnitude": 0.0, "added_mass": 0.0,
    },
    "strong_push": {
        "terrain_type": 0, "difficulty": 0.3,
        "friction": 1.0, "push_magnitude": 2.0, "added_mass": 0.0,
    },
    "heavy_load": {
        "terrain_type": 0, "difficulty": 0.3,
        "friction": 1.0, "push_magnitude": 0.0, "added_mass": 3.0,
    },
    "slippery_slope": {
        "terrain_type": 0, "difficulty": 0.7,
        "friction": 0.3, "push_magnitude": 0.0, "added_mass": 0.0,
    },
    "stairs_with_push": {
        "terrain_type": 2, "difficulty": 0.7,
        "friction": 1.0, "push_magnitude": 1.5, "added_mass": 0.0,
    },
    "obstacles_loaded": {
        "terrain_type": 4, "difficulty": 0.7,
        "friction": 1.0, "push_magnitude": 0.5, "added_mass": 2.0,
    },
    "extreme_challenge": {
        "terrain_type": 4, "difficulty": 0.95,
        "friction": 0.3, "push_magnitude": 1.5, "added_mass": 2.5,
    },
}
