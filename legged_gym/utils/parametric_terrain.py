"""
参数化地形生成器 (Parametric Terrain Generator)

设计目标：GAN 输出一组连续参数向量，本模块将参数解码为 heightfield 地形。
每种地形类型有独立的参数子空间，GAN 可以精细控制地形的每个属性。

参数空间总维度: 20 (连续) + 1 (离散地形类型)

地形类型:
  0: stepping_stones  — 梅花桩
  1: pillars          — 柱阵（圆柱/方柱随机排列）
  2: hurdles          — 跨栏（横向障碍条）
  3: gaps             — 缝隙地形
  4: stairs           — 阶梯
  5: tilted_ramps     — 倾斜坡道
  6: mixed_obstacles  — 混合障碍（组合多种元素）
  7: wave             — 波浪地形
  8: stepping_pillars — 梅花桩+柱阵混合
"""
import numpy as np
from isaacgym import terrain_utils
from typing import Dict, Optional


# ============================================================
# 参数空间定义
# ============================================================

TERRAIN_TYPES = [
    "stepping_stones",
    "pillars",
    "hurdles",
    "gaps",
    "stairs",
    "tilted_ramps",
    "mixed_obstacles",
    "wave",
    "stepping_pillars",
]
NUM_TERRAIN_TYPES = len(TERRAIN_TYPES)

# 每种地形的参数定义: (name, min, max, default, description)
PARAM_SPEC = {
    # === 通用参数 (所有地形共享, index 0-4) ===
    "difficulty":       (0.0, 1.0, 0.5,   "整体难度缩放"),
    "platform_size":    (1.0, 5.0, 3.0,   "中心安全平台大小 [m]"),
    "surface_noise":    (0.0, 0.1, 0.02,  "表面噪声幅度 [m]"),
    "border_height":    (0.0, 0.3, 0.0,   "边界墙高度 [m]"),
    "height_offset":    (-0.2, 0.2, 0.0,  "整体高度偏移 [m]"),

    # === 梅花桩参数 (index 5-9) ===
    "stone_size":       (0.2, 2.0, 1.0,   "石头尺寸 [m]"),
    "stone_distance":   (0.02, 0.4, 0.1,  "石头间距 [m]"),
    "stone_height_var": (0.0, 0.15, 0.0,  "石头高度变化 [m]"),
    "stone_shape":      (0.0, 1.0, 0.5,   "石头形状: 0=方形, 1=圆形"),
    "stone_pattern":    (0.0, 1.0, 0.5,   "排列模式: 0=网格, 0.5=交错, 1=随机"),

    # === 柱阵/跨栏参数 (index 10-13) ===
    "obstacle_height":  (0.05, 0.4, 0.15, "障碍物高度 [m]"),
    "obstacle_width":   (0.1, 1.5, 0.5,   "障碍物宽度 [m]"),
    "obstacle_spacing": (0.3, 2.0, 1.0,   "障碍物间距 [m]"),
    "obstacle_count":   (3.0, 30.0, 15.0, "障碍物数量"),

    # === 阶梯/坡道参数 (index 14-16) ===
    "step_height":      (0.02, 0.25, 0.1, "台阶高度 [m]"),
    "step_width":       (0.2, 0.6, 0.31,  "台阶宽度 [m]"),
    "slope_angle":      (0.0, 0.5, 0.2,   "坡度 [rad]"),

    # === 缝隙/波浪参数 (index 17-19) ===
    "gap_width":        (0.1, 1.5, 0.5,   "缝隙宽度 [m]"),
    "wave_amplitude":   (0.02, 0.2, 0.05, "波浪振幅 [m]"),
    "wave_frequency":   (0.5, 3.0, 1.0,   "波浪频率 [1/m]"),
}

PARAM_NAMES = list(PARAM_SPEC.keys())
NUM_CONTINUOUS_PARAMS = len(PARAM_NAMES)


def get_param_ranges():
    """返回参数范围，供 GAN 使用"""
    mins = np.array([PARAM_SPEC[k][0] for k in PARAM_NAMES])
    maxs = np.array([PARAM_SPEC[k][1] for k in PARAM_NAMES])
    defaults = np.array([PARAM_SPEC[k][2] for k in PARAM_NAMES])
    return mins, maxs, defaults


def params_dict_from_vector(vec: np.ndarray) -> Dict[str, float]:
    """将参数向量转为字典"""
    return {name: float(vec[i]) for i, name in enumerate(PARAM_NAMES)}


def clip_params(params: Dict[str, float]) -> Dict[str, float]:
    """将参数裁剪到合法范围"""
    clipped = {}
    for name, val in params.items():
        if name in PARAM_SPEC:
            lo, hi = PARAM_SPEC[name][0], PARAM_SPEC[name][1]
            clipped[name] = float(np.clip(val, lo, hi))
        else:
            clipped[name] = val
    return clipped


# ============================================================
# 地形生成函数
# ============================================================

def make_parametric_terrain(
    width_pixels: int,
    length_pixels: int,
    horizontal_scale: float,
    vertical_scale: float,
    terrain_type: int,
    params: Dict[str, float],
) -> np.ndarray:
    """
    根据参数生成 heightfield。

    Args:
        width_pixels: 地形宽度 (像素)
        length_pixels: 地形长度 (像素)
        horizontal_scale: 水平分辨率 [m/pixel]
        vertical_scale: 垂直分辨率 [m/unit]
        terrain_type: 地形类型索引 (0-8)
        params: 参数字典

    Returns:
        height_field: int16 heightfield array
    """
    terrain = terrain_utils.SubTerrain(
        "terrain",
        width=width_pixels,
        length=length_pixels,
        vertical_scale=vertical_scale,
        horizontal_scale=horizontal_scale,
    )

    p = clip_params(params)
    diff = p.get("difficulty", 0.5)

    # 根据地形类型分发
    type_name = TERRAIN_TYPES[terrain_type % NUM_TERRAIN_TYPES]

    if type_name == "stepping_stones":
        _gen_stepping_stones(terrain, p, diff)
    elif type_name == "pillars":
        _gen_pillars(terrain, p, diff)
    elif type_name == "hurdles":
        _gen_hurdles(terrain, p, diff)
    elif type_name == "gaps":
        _gen_gaps(terrain, p, diff)
    elif type_name == "stairs":
        _gen_stairs(terrain, p, diff)
    elif type_name == "tilted_ramps":
        _gen_tilted_ramps(terrain, p, diff)
    elif type_name == "mixed_obstacles":
        _gen_mixed_obstacles(terrain, p, diff)
    elif type_name == "wave":
        _gen_wave(terrain, p, diff)
    elif type_name == "stepping_pillars":
        _gen_stepping_pillars(terrain, p, diff)

    # 添加表面噪声
    noise_amp = p.get("surface_noise", 0.0)
    if noise_amp > 0.001:
        noise = np.random.uniform(
            -noise_amp, noise_amp,
            size=terrain.height_field_raw.shape
        )
        terrain.height_field_raw += (noise / vertical_scale).astype(np.int16)

    # 高度偏移
    offset = p.get("height_offset", 0.0)
    if abs(offset) > 0.001:
        terrain.height_field_raw += int(offset / vertical_scale)

    return terrain.height_field_raw


# ============================================================
# 各地形类型的生成实现
# ============================================================

def _gen_stepping_stones(terrain, p, diff):
    """
    梅花桩地形 — 参数化版本

    支持:
    - 石头大小、间距、高度变化
    - 形状 (方形/圆形)
    - 排列模式 (网格/交错/随机)
    """
    stone_size = p.get("stone_size", 1.0) * (1.1 - diff)
    stone_dist = p.get("stone_distance", 0.1) * (0.5 + diff)
    height_var = p.get("stone_height_var", 0.0) * diff
    shape = p.get("stone_shape", 0.5)
    pattern = p.get("stone_pattern", 0.5)
    platform = p.get("platform_size", 3.0)

    hs = terrain.horizontal_scale
    vs = terrain.vertical_scale
    W, L = terrain.width, terrain.length

    # 先把整个地形设为深坑（掉下去 = 失败）
    terrain.height_field_raw[:] = int(-1.0 / vs)

    # 中心安全平台
    plat_px = int(platform / hs / 2)
    cx, cy = L // 2, W // 2
    terrain.height_field_raw[
        cx - plat_px: cx + plat_px,
        cy - plat_px: cy + plat_px
    ] = 0

    # 石头尺寸 (像素)
    stone_px = max(2, int(stone_size / hs))
    dist_px = max(1, int(stone_dist / hs))
    step = stone_px + dist_px

    if pattern < 0.33:
        # 网格排列
        _place_stones_grid(terrain, stone_px, step, height_var, vs, shape)
    elif pattern < 0.66:
        # 交错排列 (蜂窝状)
        _place_stones_staggered(terrain, stone_px, step, height_var, vs, shape)
    else:
        # 随机排列
        _place_stones_random(terrain, stone_px, step, height_var, vs, shape)


def _place_stones_grid(terrain, stone_px, step, height_var, vs, shape):
    """网格排列梅花桩"""
    W, L = terrain.width, terrain.length
    for x in range(0, L, step):
        for y in range(0, W, step):
            h = int(np.random.uniform(-height_var, height_var) / vs) if height_var > 0 else 0
            _place_single_stone(terrain, x, y, stone_px, h, shape)


def _place_stones_staggered(terrain, stone_px, step, height_var, vs, shape):
    """交错排列梅花桩 (蜂窝状)"""
    W, L = terrain.width, terrain.length
    row_idx = 0
    for x in range(0, L, step):
        offset = (step // 2) if (row_idx % 2 == 1) else 0
        for y in range(offset, W, step):
            h = int(np.random.uniform(-height_var, height_var) / vs) if height_var > 0 else 0
            _place_single_stone(terrain, x, y, stone_px, h, shape)
        row_idx += 1


def _place_stones_random(terrain, stone_px, step, height_var, vs, shape):
    """随机排列梅花桩"""
    W, L = terrain.width, terrain.length
    num_stones = (L * W) // (step * step)
    for _ in range(num_stones):
        x = np.random.randint(0, max(1, L - stone_px))
        y = np.random.randint(0, max(1, W - stone_px))
        h = int(np.random.uniform(-height_var, height_var) / vs) if height_var > 0 else 0
        _place_single_stone(terrain, x, y, stone_px, h, shape)


def _place_single_stone(terrain, x, y, size, height, shape):
    """放置单个石头"""
    W, L = terrain.width, terrain.length
    x1 = max(0, x)
    x2 = min(L, x + size)
    y1 = max(0, y)
    y2 = min(W, y + size)

    if shape > 0.5:
        # 圆形石头
        cx_s = (x1 + x2) / 2.0
        cy_s = (y1 + y2) / 2.0
        r = size / 2.0
        for ix in range(x1, x2):
            for iy in range(y1, y2):
                if (ix - cx_s) ** 2 + (iy - cy_s) ** 2 <= r * r:
                    terrain.height_field_raw[ix, iy] = height
    else:
        # 方形石头
        terrain.height_field_raw[x1:x2, y1:y2] = height


def _gen_pillars(terrain, p, diff):
    """
    柱阵地形 — 随机分布的柱子，狗需要在柱子间穿行
    """
    height = p.get("obstacle_height", 0.15) * (0.5 + diff)
    width = p.get("obstacle_width", 0.5) * (1.2 - 0.4 * diff)
    count = int(p.get("obstacle_count", 15) * (0.5 + diff))
    platform = p.get("platform_size", 3.0)

    hs = terrain.horizontal_scale
    vs = terrain.vertical_scale
    W, L = terrain.width, terrain.length

    # 平坦基底
    terrain.height_field_raw[:] = 0

    pillar_px = max(2, int(width / hs))
    h_val = int(height / vs)

    for _ in range(count):
        px = np.random.randint(0, max(1, L - pillar_px))
        py = np.random.randint(0, max(1, W - pillar_px))
        # 避开中心平台
        plat_px = int(platform / hs / 2)
        cx, cy = L // 2, W // 2
        if abs(px - cx) < plat_px and abs(py - cy) < plat_px:
            continue
        terrain.height_field_raw[px:px + pillar_px, py:py + pillar_px] = h_val


def _gen_hurdles(terrain, p, diff):
    """
    跨栏地形 — 横向障碍条，狗需要跨越
    """
    height = p.get("obstacle_height", 0.15) * (0.3 + 0.7 * diff)
    bar_width = max(1, int(p.get("obstacle_width", 0.5) / terrain.horizontal_scale))
    spacing = max(3, int(p.get("obstacle_spacing", 1.0) / terrain.horizontal_scale))
    platform = p.get("platform_size", 3.0)

    vs = terrain.vertical_scale
    hs = terrain.horizontal_scale
    W, L = terrain.width, terrain.length

    terrain.height_field_raw[:] = 0
    h_val = int(height / vs)

    plat_px = int(platform / hs / 2)
    cx = L // 2

    # 从中心向两侧放置跨栏
    for x in range(cx + plat_px, L - bar_width, spacing):
        terrain.height_field_raw[x:x + bar_width, :] = h_val
    for x in range(cx - plat_px - bar_width, bar_width, -spacing):
        terrain.height_field_raw[x:x + bar_width, :] = h_val


def _gen_gaps(terrain, p, diff):
    """
    缝隙地形 — 地面上的缝隙，狗需要跳过
    """
    gap_w = p.get("gap_width", 0.5) * (0.3 + 0.7 * diff)
    spacing = max(3, int(p.get("obstacle_spacing", 1.0) / terrain.horizontal_scale))
    platform = p.get("platform_size", 3.0)

    hs = terrain.horizontal_scale
    vs = terrain.vertical_scale
    W, L = terrain.width, terrain.length

    terrain.height_field_raw[:] = 0
    gap_px = max(1, int(gap_w / hs))
    plat_px = int(platform / hs / 2)
    cx = L // 2

    for x in range(cx + plat_px, L - gap_px, spacing):
        terrain.height_field_raw[x:x + gap_px, :] = int(-1.0 / vs)
    for x in range(cx - plat_px - gap_px, gap_px, -spacing):
        terrain.height_field_raw[x:x + gap_px, :] = int(-1.0 / vs)


def _gen_stairs(terrain, p, diff):
    """
    阶梯地形
    """
    step_h = p.get("step_height", 0.1) * (0.3 + 0.7 * diff)
    step_w = p.get("step_width", 0.31)
    platform = p.get("platform_size", 3.0)

    # 使用 isaacgym 内置
    terrain_utils.pyramid_stairs_terrain(
        terrain,
        step_width=step_w,
        step_height=step_h,
        platform_size=platform,
    )


def _gen_tilted_ramps(terrain, p, diff):
    """
    倾斜坡道
    """
    slope = p.get("slope_angle", 0.2) * (0.3 + 0.7 * diff)
    platform = p.get("platform_size", 3.0)

    terrain_utils.pyramid_sloped_terrain(
        terrain,
        slope=slope,
        platform_size=platform,
    )


def _gen_wave(terrain, p, diff):
    """
    波浪地形 — 正弦波起伏
    """
    amp = p.get("wave_amplitude", 0.05) * (0.5 + diff)
    freq = p.get("wave_frequency", 1.0)

    hs = terrain.horizontal_scale
    vs = terrain.vertical_scale
    W, L = terrain.width, terrain.length

    for ix in range(L):
        for iy in range(W):
            x_m = ix * hs
            y_m = iy * hs
            h = amp * (np.sin(2 * np.pi * freq * x_m) +
                       np.sin(2 * np.pi * freq * y_m * 0.7))
            terrain.height_field_raw[ix, iy] = int(h / vs)


def _gen_mixed_obstacles(terrain, p, diff):
    """
    混合障碍 — 组合梅花桩 + 跨栏 + 小柱子
    """
    terrain.height_field_raw[:] = 0

    # 前半部分: 跨栏
    half_p = dict(p)
    half_p["obstacle_height"] = p.get("obstacle_height", 0.15) * 0.7
    _gen_hurdles_partial(terrain, half_p, diff, x_start=0, x_end=terrain.length // 2)

    # 后半部分: 离散障碍
    obs_h = p.get("obstacle_height", 0.15) * (0.3 + 0.7 * diff)
    count = int(p.get("obstacle_count", 15) * 0.5)
    vs = terrain.vertical_scale
    hs = terrain.horizontal_scale
    W, L = terrain.width, terrain.length
    h_val = int(obs_h / vs)
    obs_px = max(2, int(p.get("obstacle_width", 0.5) / hs))

    for _ in range(count):
        px = np.random.randint(L // 2, max(L // 2 + 1, L - obs_px))
        py = np.random.randint(0, max(1, W - obs_px))
        terrain.height_field_raw[px:px + obs_px, py:py + obs_px] = h_val


def _gen_hurdles_partial(terrain, p, diff, x_start, x_end):
    """在指定 x 范围内生成跨栏"""
    height = p.get("obstacle_height", 0.15) * (0.3 + 0.7 * diff)
    bar_width = max(1, int(p.get("obstacle_width", 0.5) / terrain.horizontal_scale))
    spacing = max(3, int(p.get("obstacle_spacing", 1.0) / terrain.horizontal_scale))
    vs = terrain.vertical_scale
    h_val = int(height / vs)

    for x in range(x_start, x_end - bar_width, spacing):
        terrain.height_field_raw[x:x + bar_width, :] = h_val


def _gen_stepping_pillars(terrain, p, diff):
    """
    梅花桩 + 柱阵混合 — 石头之间穿插柱子
    """
    # 先生成梅花桩
    _gen_stepping_stones(terrain, p, diff)

    # 再在空隙中加柱子
    pillar_h = p.get("obstacle_height", 0.15) * diff
    pillar_w = max(2, int(p.get("obstacle_width", 0.3) / terrain.horizontal_scale))
    count = int(p.get("obstacle_count", 10) * diff * 0.5)
    vs = terrain.vertical_scale
    W, L = terrain.width, terrain.length
    h_val = int(pillar_h / vs)

    for _ in range(count):
        px = np.random.randint(0, max(1, L - pillar_w))
        py = np.random.randint(0, max(1, W - pillar_w))
        # 只在深坑区域放柱子（不覆盖石头）
        if terrain.height_field_raw[px, py] < 0:
            terrain.height_field_raw[px:px + pillar_w, py:py + pillar_w] = h_val
