"""
四种地形生成器的统一接口

每个生成器实现:
- get_param_spec(): 返回参数空间定义
- make_generator_network(): 返回对应的 GAN 网络
- apply_params_to_env(): 将参数应用到 IsaacGym 环境
- make_terrain_from_params(): 从参数生成 heightfield
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from typing import Dict, Tuple, List

from isaacgym import terrain_utils


# ============================================================
# 基类
# ============================================================

class BaseTerrainGenerator:
    """地形生成器基类"""

    name: str = "base"
    param_dim: int = 0
    num_terrain_types: int = 0

    def get_param_spec(self) -> Dict[str, Tuple[float, float, float, str]]:
        """返回 {name: (min, max, default, desc)}"""
        raise NotImplementedError

    def make_terrain(self, width_px, length_px, hs, vs, terrain_type, params):
        """生成 heightfield"""
        raise NotImplementedError

    def apply_to_env(self, env, terrain_type, params):
        """将参数应用到 env 的 domain_rand 等配置"""
        p = params
        if hasattr(env.cfg, 'domain_rand'):
            dr = env.cfg.domain_rand
            if 'friction' in p:
                dr.friction_range = [p['friction'], p['friction']]
            if 'push_magnitude' in p:
                dr.max_push_vel_xy = p['push_magnitude']
                dr.push_robots = p['push_magnitude'] > 0.01
            if 'added_mass' in p:
                dr.added_mass_range = [p['added_mass'], p['added_mass']]
                dr.randomize_base_mass = abs(p['added_mass']) > 0.01


# ============================================================
# G1: IsaacGym 原生 (7维)
# ============================================================

class G1_OriginalGenerator(BaseTerrainGenerator):
    """与现有 terrain.py 完全一致的参数空间"""

    name = "G1"
    param_dim = 5  # 连续参数 (不含离散 terrain_type)
    num_terrain_types = 7

    def get_param_spec(self):
        return {
            "difficulty":      (0.0, 1.0, 0.5,  "整体难度"),
            "friction":        (0.3, 1.5, 1.0,  "摩擦系数"),
            "push_magnitude":  (0.0, 1.5, 0.0,  "外力扰动"),
            "added_mass":      (-1.5, 2.0, 0.0, "附加质量"),
        }

    def make_terrain(self, width_px, length_px, hs, vs, terrain_type, params):
        terrain = terrain_utils.SubTerrain(
            "terrain", width=width_px, length=length_px,
            vertical_scale=vs, horizontal_scale=hs,
        )
        diff = params.get("difficulty", 0.5)
        # 与原 terrain.py make_terrain 一致
        proportions = [0.1, 0.2, 0.55, 0.75, 0.95, 1.0]
        choice = terrain_type / max(self.num_terrain_types - 1, 1)

        slope = diff * 0.4
        step_height = 0.05 + 0.18 * diff
        discrete_h = 0.05 + diff * 0.2
        stone_size = 1.5 * (1.05 - diff)
        stone_dist = 0.05 if diff == 0 else 0.1
        gap_size = 1.0 * diff

        if choice < proportions[0]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, -0.05, 0.05, 0.005, 0.2)
        elif choice < proportions[2]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31,
                                                  step_height=step_height, platform_size=3.)
        elif choice < proportions[3]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31,
                                                  step_height=-step_height, platform_size=3.)
        elif choice < proportions[4]:
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_h, 1., 2., 20, platform_size=3.)
        elif choice < proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stone_size,
                                                   stone_distance=stone_dist, max_height=0., platform_size=4.)
        else:
            terrain.height_field_raw[:] = 0

        return terrain.height_field_raw


# ============================================================
# G2: Isaac Lab 扩展 (15维)
# ============================================================

class G2_IsaacLabGenerator(BaseTerrainGenerator):
    """参考 Isaac Lab 的参数化方式，每种地形有独立 range 参数"""

    name = "G2"
    param_dim = 13
    num_terrain_types = 9

    TYPES = [
        "pyramid_slope", "rough_slope", "stairs_up", "stairs_down",
        "discrete_obstacles", "stepping_stones", "gap", "wave", "inverted_stairs",
    ]

    def get_param_spec(self):
        return {
            "difficulty":          (0.0, 1.0, 0.5,   "整体难度"),
            "step_height":         (0.02, 0.25, 0.1,  "台阶高度 [m]"),
            "step_width":          (0.2, 0.6, 0.31,   "台阶宽度 [m]"),
            "stone_width":         (0.2, 2.0, 1.0,    "石头宽度 [m]"),
            "stone_distance":      (0.02, 0.4, 0.1,   "石头间距 [m]"),
            "stone_height_max":    (0.0, 0.15, 0.0,   "石头最大高度 [m]"),
            "obstacle_width":      (0.1, 1.5, 0.5,    "障碍物宽度 [m]"),
            "obstacle_height":     (0.05, 0.4, 0.15,  "障碍物高度 [m]"),
            "num_obstacles":       (3.0, 30.0, 15.0,  "障碍物数量"),
            "gap_width":           (0.1, 1.5, 0.5,    "缝隙宽度 [m]"),
            "wave_amplitude":      (0.02, 0.2, 0.05,  "波浪振幅 [m]"),
            "friction":            (0.3, 1.5, 1.0,    "摩擦系数"),
            "push_magnitude":      (0.0, 1.5, 0.0,    "外力扰动"),
            "added_mass":          (-1.5, 2.0, 0.0,   "附加质量"),
        }

    def make_terrain(self, width_px, length_px, hs, vs, terrain_type, params):
        terrain = terrain_utils.SubTerrain(
            "terrain", width=width_px, length=length_px,
            vertical_scale=vs, horizontal_scale=hs,
        )
        p = params
        diff = p.get("difficulty", 0.5)
        t = self.TYPES[terrain_type % len(self.TYPES)]

        if t == "pyramid_slope":
            slope = p.get("step_height", 0.1) * 2 * diff
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif t == "rough_slope":
            slope = p.get("step_height", 0.1) * 2 * diff
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, -0.05, 0.05, 0.005, 0.2)
        elif t == "stairs_up":
            sh = p.get("step_height", 0.1) * (0.3 + 0.7 * diff)
            sw = p.get("step_width", 0.31)
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=sw, step_height=sh, platform_size=3.)
        elif t == "stairs_down":
            sh = -p.get("step_height", 0.1) * (0.3 + 0.7 * diff)
            sw = p.get("step_width", 0.31)
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=sw, step_height=sh, platform_size=3.)
        elif t == "discrete_obstacles":
            oh = p.get("obstacle_height", 0.15) * (0.3 + 0.7 * diff)
            ow_min = p.get("obstacle_width", 0.5) * 0.5
            ow_max = p.get("obstacle_width", 0.5) * 1.5
            n = int(p.get("num_obstacles", 15))
            terrain_utils.discrete_obstacles_terrain(terrain, oh, ow_min, ow_max, n, platform_size=3.)
        elif t == "stepping_stones":
            ss = p.get("stone_width", 1.0) * (1.1 - diff)
            sd = p.get("stone_distance", 0.1) * (0.5 + diff)
            sh = p.get("stone_height_max", 0.0)
            terrain_utils.stepping_stones_terrain(terrain, stone_size=ss, stone_distance=sd,
                                                   max_height=sh, platform_size=4.)
        elif t == "gap":
            gw = p.get("gap_width", 0.5) * diff
            gap_px = max(1, int(gw / hs))
            cx, cy = terrain.length // 2, terrain.width // 2
            terrain.height_field_raw[cx - gap_px:cx + gap_px, :] = int(-1.0 / vs)
        elif t == "wave":
            amp = p.get("wave_amplitude", 0.05) * (0.5 + diff)
            for ix in range(terrain.length):
                for iy in range(terrain.width):
                    h = amp * np.sin(2 * np.pi * ix * hs)
                    terrain.height_field_raw[ix, iy] = int(h / vs)
        elif t == "inverted_stairs":
            sh = -p.get("step_height", 0.1) * (0.3 + 0.7 * diff)
            sw = p.get("step_width", 0.31)
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=sw, step_height=sh, platform_size=3.)

        return terrain.height_field_raw


# ============================================================
# G3: 参数化梅花桩 (21维) — 直接使用 parametric_terrain.py
# ============================================================

class G3_ParametricGenerator(BaseTerrainGenerator):
    """使用已实现的 parametric_terrain.py"""

    name = "G3"
    param_dim = 20
    num_terrain_types = 9

    def get_param_spec(self):
        from legged_gym.utils.parametric_terrain import PARAM_SPEC
        return {k: v[:3] + (v[3],) for k, v in PARAM_SPEC.items()}

    def make_terrain(self, width_px, length_px, hs, vs, terrain_type, params):
        from legged_gym.utils.parametric_terrain import make_parametric_terrain
        return make_parametric_terrain(width_px, length_px, hs, vs, terrain_type, params)


# ============================================================
# G4: Trimesh 重复物体 (18维)
# ============================================================

class G4_TrimeshGenerator(BaseTerrainGenerator):
    """参考 Isaac Lab MeshRepeatedObjects 系列"""

    name = "G4"
    param_dim = 14
    num_terrain_types = 6

    TYPES = [
        "repeated_cylinders", "repeated_boxes", "repeated_cones",
        "rails", "pit", "random_grid",
    ]

    def get_param_spec(self):
        return {
            "difficulty":       (0.0, 1.0, 0.5,   "整体难度"),
            "num_objects":      (5.0, 50.0, 20.0,  "物体数量"),
            "object_height":    (0.05, 0.5, 0.15,  "物体高度 [m]"),
            "object_radius":    (0.05, 0.5, 0.15,  "物体半径 [m]"),
            "object_size_x":    (0.1, 1.0, 0.3,    "方块 X 尺寸 [m]"),
            "object_size_y":    (0.1, 1.0, 0.3,    "方块 Y 尺寸 [m]"),
            "max_tilt_angle":   (0.0, 45.0, 0.0,   "最大倾斜角 [度]"),
            "height_noise":     (0.0, 0.1, 0.0,    "高度噪声 [m]"),
            "platform_width":   (0.5, 3.0, 1.5,    "平台宽度 [m]"),
            "rail_thickness":   (0.05, 0.3, 0.1,   "轨道厚度 [m]"),
            "rail_height":      (0.05, 0.3, 0.1,   "轨道高度 [m]"),
            "pit_depth":        (0.1, 1.0, 0.3,    "坑深度 [m]"),
            "grid_width":       (0.3, 1.5, 0.5,    "网格宽度 [m]"),
            "grid_height":      (0.05, 0.3, 0.1,   "网格高度 [m]"),
            "friction":         (0.3, 1.5, 1.0,    "摩擦系数"),
            "push_magnitude":   (0.0, 1.5, 0.0,    "外力扰动"),
            "added_mass":       (-1.5, 2.0, 0.0,   "附加质量"),
        }

    def make_terrain(self, width_px, length_px, hs, vs, terrain_type, params):
        terrain = terrain_utils.SubTerrain(
            "terrain", width=width_px, length=length_px,
            vertical_scale=vs, horizontal_scale=hs,
        )
        p = params
        diff = p.get("difficulty", 0.5)
        t = self.TYPES[terrain_type % len(self.TYPES)]

        if t in ("repeated_cylinders", "repeated_boxes", "repeated_cones"):
            self._gen_repeated_objects(terrain, p, diff, t, hs, vs)
        elif t == "rails":
            self._gen_rails(terrain, p, diff, hs, vs)
        elif t == "pit":
            self._gen_pit(terrain, p, diff, vs)
        elif t == "random_grid":
            self._gen_random_grid(terrain, p, diff, hs, vs)

        return terrain.height_field_raw

    def _gen_repeated_objects(self, terrain, p, diff, obj_type, hs, vs):
        terrain.height_field_raw[:] = 0
        n = int(p.get("num_objects", 20) * (0.5 + 0.5 * diff))
        h = p.get("object_height", 0.15) * (0.3 + 0.7 * diff)
        r = p.get("object_radius", 0.15)
        sx = p.get("object_size_x", 0.3)
        sy = p.get("object_size_y", 0.3)
        noise = p.get("height_noise", 0.0)
        plat = int(p.get("platform_width", 1.5) / hs / 2)
        W, L = terrain.width, terrain.length
        cx, cy = L // 2, W // 2
        h_val = int(h / vs)

        for _ in range(n):
            px = np.random.randint(0, max(1, L - 3))
            py = np.random.randint(0, max(1, W - 3))
            if abs(px - cx) < plat and abs(py - cy) < plat:
                continue
            hn = int(np.random.uniform(-noise, noise) / vs) if noise > 0 else 0

            if obj_type == "repeated_cylinders":
                r_px = max(1, int(r / hs))
                for ix in range(max(0, px - r_px), min(L, px + r_px)):
                    for iy in range(max(0, py - r_px), min(W, py + r_px)):
                        if (ix - px)**2 + (iy - py)**2 <= r_px**2:
                            terrain.height_field_raw[ix, iy] = h_val + hn
            elif obj_type == "repeated_boxes":
                bx = max(1, int(sx / hs))
                by = max(1, int(sy / hs))
                x1, x2 = max(0, px), min(L, px + bx)
                y1, y2 = max(0, py), min(W, py + by)
                terrain.height_field_raw[x1:x2, y1:y2] = h_val + hn
            elif obj_type == "repeated_cones":
                r_px = max(1, int(r / hs))
                for ix in range(max(0, px - r_px), min(L, px + r_px)):
                    for iy in range(max(0, py - r_px), min(W, py + r_px)):
                        dist = np.sqrt((ix - px)**2 + (iy - py)**2)
                        if dist <= r_px:
                            cone_h = h_val * (1.0 - dist / r_px)
                            terrain.height_field_raw[ix, iy] = int(cone_h) + hn

    def _gen_rails(self, terrain, p, diff, hs, vs):
        terrain.height_field_raw[:] = 0
        thickness = max(1, int(p.get("rail_thickness", 0.1) / hs))
        height = p.get("rail_height", 0.1) * (0.3 + 0.7 * diff)
        h_val = int(height / vs)
        W, L = terrain.width, terrain.length
        spacing = max(thickness + 2, int(1.0 / hs))
        for y in range(0, W, spacing):
            y2 = min(W, y + thickness)
            terrain.height_field_raw[:, y:y2] = h_val

    def _gen_pit(self, terrain, p, diff, vs):
        terrain.height_field_raw[:] = 0
        depth = p.get("pit_depth", 0.3) * (0.3 + 0.7 * diff)
        plat = int(p.get("platform_width", 1.5) / terrain.horizontal_scale / 2)
        W, L = terrain.width, terrain.length
        cx, cy = L // 2, W // 2
        terrain.height_field_raw[:] = int(-depth / vs)
        terrain.height_field_raw[cx - plat:cx + plat, cy - plat:cy + plat] = 0

    def _gen_random_grid(self, terrain, p, diff, hs, vs):
        terrain.height_field_raw[:] = 0
        gw = max(2, int(p.get("grid_width", 0.5) / hs))
        gh = p.get("grid_height", 0.1) * (0.3 + 0.7 * diff)
        W, L = terrain.width, terrain.length
        for x in range(0, L, gw):
            for y in range(0, W, gw):
                h = np.random.uniform(-gh, gh)
                x2, y2 = min(L, x + gw), min(W, y + gw)
                terrain.height_field_raw[x:x2, y:y2] = int(h / vs)


# ============================================================
# 统一 GAN 网络工厂
# ============================================================

def make_generator_network(generator: BaseTerrainGenerator, condition_dim=16, device="cuda:0"):
    """为指定的地形生成器创建对应的 GAN 网络"""
    spec = generator.get_param_spec()
    param_names = list(spec.keys())
    # 去掉 domain_rand 参数 (friction, push, mass)，这些不是地形几何参数
    # 但仍然作为 GAN 输出的一部分
    n_continuous = len(param_names)
    n_types = generator.num_terrain_types

    mins = torch.tensor([spec[k][0] for k in param_names], dtype=torch.float32)
    maxs = torch.tensor([spec[k][1] for k in param_names], dtype=torch.float32)

    net = _UnifiedGeneratorNet(
        condition_dim=condition_dim,
        n_types=n_types,
        n_continuous=n_continuous,
        param_mins=mins,
        param_maxs=maxs,
        param_names=param_names,
    ).to(device)

    return net


class _UnifiedGeneratorNet(nn.Module):
    """统一的生成器网络，适配任意参数空间维度"""

    def __init__(self, condition_dim, n_types, n_continuous, param_mins, param_maxs, param_names,
                 use_lstm=True, lstm_hidden_size=64):
        super().__init__()
        self.n_types = n_types
        self.n_continuous = n_continuous
        self.param_names = param_names
        self.use_lstm = use_lstm
        self.register_buffer("param_mins", param_mins)
        self.register_buffer("param_maxs", param_maxs)

        self.backbone = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )

        # LSTM for generation history memory (matches original code)
        feat_dim = 64
        if use_lstm:
            self.lstm = nn.LSTM(64, lstm_hidden_size, batch_first=True)
            self.lstm_hidden = None
            feat_dim = lstm_hidden_size

        self.type_head = nn.Linear(feat_dim, n_types)
        self.param_mean = nn.Linear(feat_dim, n_continuous)
        self.param_logstd = nn.Parameter(torch.zeros(n_continuous))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def reset_lstm(self, batch_size=1):
        if self.use_lstm:
            device = next(self.parameters()).device
            self.lstm_hidden = (
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=device),
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=device),
            )

    def forward(self, condition, deterministic=False):
        feat = self.backbone(condition)

        if self.use_lstm:
            feat = feat.unsqueeze(1)  # [B, 1, 64]
            feat, self.lstm_hidden = self.lstm(feat, self.lstm_hidden)
            feat = feat.squeeze(1)

        type_logits = self.type_head(feat)
        type_dist = Categorical(logits=type_logits)
        terrain_type = type_logits.argmax(-1) if deterministic else type_dist.sample()
        type_lp = type_dist.log_prob(terrain_type)

        raw_mean = self.param_mean(feat)
        std = torch.exp(self.param_logstd).expand_as(raw_mean)
        param_dist = Normal(raw_mean, std)
        raw = raw_mean if deterministic else param_dist.rsample()
        param_lp = param_dist.log_prob(raw).sum(-1)

        normalized = torch.sigmoid(raw)
        param_vector = self.param_mins + normalized * (self.param_maxs - self.param_mins)

        log_prob = type_lp + param_lp
        return terrain_type, param_vector, log_prob

    def get_entropy(self, condition):
        feat = self.backbone(condition)
        if self.use_lstm:
            feat = feat.unsqueeze(1)
            feat, _ = self.lstm(feat, self.lstm_hidden)
            feat = feat.squeeze(1)
        te = Categorical(logits=self.type_head(feat)).entropy()
        raw_mean = self.param_mean(feat)
        std = torch.exp(self.param_logstd).expand_as(raw_mean)
        pe = Normal(raw_mean, std).entropy().sum(-1)
        return te + pe

    def to_params_dict(self, param_vector):
        """tensor → dict"""
        v = param_vector.detach().cpu().numpy()
        if v.ndim == 1:
            return {self.param_names[i]: float(v[i]) for i in range(len(self.param_names))}
        return [{self.param_names[i]: float(v[b, i]) for i in range(len(self.param_names))} for b in range(len(v))]


# ============================================================
# 工厂函数
# ============================================================

GENERATOR_REGISTRY = {
    "G1": G1_OriginalGenerator,
    "G2": G2_IsaacLabGenerator,
    "G3": G3_ParametricGenerator,
    "G4": G4_TrimeshGenerator,
}


def get_generator(name: str) -> BaseTerrainGenerator:
    return GENERATOR_REGISTRY[name]()
