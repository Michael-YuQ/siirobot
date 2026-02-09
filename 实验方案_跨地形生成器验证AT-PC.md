# AT-PC 跨地形生成器泛化性验证实验方案

## 实验目标

验证 AT-PC 的核心主张：**不管底层用什么参数化地形生成器，AT-PC 都能比 DR 和 PAIRED 更好地利用参数空间来训练出更鲁棒的策略。**

这不是对比地形生成器本身的优劣，而是证明 AT-PC 作为"参数优化方法"的通用性——它能在任意参数化地形生成器上发挥优势。

---

## 实验设计概览

### 自变量（控制变量）

| 维度 | 水平 | 说明 |
|------|------|------|
| 训练方法 | DR / PAIRED / AT-PC | 3 种参数生成策略 |
| 地形生成器 | G1 / G2 / G3 / G4 | 4 种参数化地形生成器 |

共 3 × 4 = **12 组实验**，每组跑 3 个 seed → 共 **36 次训练**。

### 因变量（评估指标）

| 指标 | 说明 |
|------|------|
| 零样本成功率 | 在 held-out 测试地形上的成功率 |
| 平均奖励 | Episode 累积奖励均值 |
| 收敛速度 | 达到 70% 成功率所需迭代数 |
| 参数空间覆盖率 | 训练过程中实际探索的参数空间比例 |
| Accept Rate | AT-PC 的新颖性过滤效果 |

---

## 四种地形生成器定义

### G1: IsaacGym 原生地形（你现有的 terrain.py）

最经典的基线，参数空间最小。

**参数空间 (7维):**
- terrain_type: 离散 [0-6]，7 种地形
- difficulty: 连续 [0, 1]
- friction: 连续 [0.3, 1.5]
- push_magnitude: 连续 [0, 1.5]
- added_mass: 连续 [-1.5, 2.0]

**地形类型:** 平滑斜坡、粗糙斜坡、上楼梯、下楼梯、离散障碍、梅花桩、缝隙

**特点:** 参数空间小，difficulty 一个标量控制所有几何参数，地形多样性有限。

**对应代码:** `legged_gym/utils/terrain.py` + `legged_gym/curriculum/generator.py`

### G2: Isaac Lab 风格扩展地形

参考 Isaac Lab 的参数化方式，每种地形有独立的 (min, max) range 参数。

**参数空间 (15维):**
- terrain_type: 离散 [0-8]，9 种地形
- difficulty: 连续 [0, 1]
- step_height_range: 连续 [0.02, 0.25]
- step_width: 连续 [0.2, 0.6]
- stone_width_range: 连续 [0.2, 2.0]
- stone_distance_range: 连续 [0.02, 0.4]
- stone_height_max: 连续 [0, 0.15]
- obstacle_width_range: 连续 [0.1, 1.5]
- obstacle_height_range: 连续 [0.05, 0.4]
- num_obstacles: 连续 [3, 30]
- gap_width_range: 连续 [0.1, 1.5]
- wave_amplitude: 连续 [0.02, 0.2]
- friction: 连续 [0.3, 1.5]
- push_magnitude: 连续 [0, 1.5]
- added_mass: 连续 [-1.5, 2.0]

**新增地形类型:** 波浪地形、随机网格、反向金字塔阶梯

**特点:** 参数空间中等，每种地形有独立参数，difficulty 仍用于插值但不再是唯一控制。

**对应代码:** 需新建 `terrain_g2.py`，扩展 `generator.py` 输出维度

### G3: 参数化梅花桩专精地形（你的 parametric_terrain.py）

你已经实现的 20 维参数空间，重点在梅花桩及其变体。

**参数空间 (21维):**
- terrain_type: 离散 [0-8]，9 种地形
- 20 维连续参数（通用 5 + 梅花桩 5 + 柱阵/跨栏 4 + 阶梯/坡道 3 + 缝隙/波浪 3）

**特点:** 参数空间大，梅花桩参数最丰富（大小、间距、高度变化、形状、排列模式），适合精细控制。

**对应代码:** `legged_gym/utils/parametric_terrain.py` + `legged_gym/curriculum/parametric_generator.py`

### G4: Isaac Lab Trimesh 风格（重复物体地形）

参考 Isaac Lab 的 MeshRepeatedObjects 系列，用 3D mesh 物体（圆柱、方块、锥体）参数化生成。

**参数空间 (18维):**
- terrain_type: 离散 [0-5]，6 种地形
- object_type: 离散 [cylinder, box, cone]
- num_objects_range: 连续 [5, 50]
- object_height_range: 连续 [0.05, 0.5]
- object_radius_range: 连续 [0.05, 0.5] (圆柱/锥体)
- object_size_range: 连续 [0.1, 1.0] (方块)
- max_yx_angle: 连续 [0, 45] 度（物体倾斜角）
- max_height_noise: 连续 [0, 0.1]
- platform_width: 连续 [0.5, 3.0]
- rail_thickness: 连续 [0.05, 0.3]
- rail_height: 连续 [0.05, 0.3]
- pit_depth: 连续 [0.1, 1.0]
- gap_width: 连续 [0.1, 1.5]
- friction: 连续 [0.3, 1.5]
- push_magnitude: 连续 [0, 1.5]
- added_mass: 连续 [-1.5, 2.0]
- ring_width: 连续 [0.2, 1.0]
- ring_height: 连续 [0.1, 0.5]

**地形类型:** 重复圆柱、重复方块、重复锥体、轨道(rails)、浮环(floating ring)、坑(pit)

**特点:** 参数空间大，3D mesh 物体比 heightfield 更真实，物体可以倾斜。

**对应代码:** 需新建 `terrain_g4.py`，扩展 `generator.py`

---

## 实验矩阵

```
              G1(原生7维)  G2(扩展15维)  G3(梅花桩21维)  G4(Trimesh18维)
DR            DR-G1        DR-G2         DR-G3           DR-G4
PAIRED        PA-G1        PA-G2         PA-G3           PA-G4
AT-PC         AT-G1        AT-G2         AT-G3           AT-G4
```

每个格子跑 3 个 seed (42, 123, 456)，共 36 次训练。

---

## 训练配置

所有实验统一以下配置（与论文第五章一致）：

| 参数 | 值 |
|------|-----|
| 并行环境数 | 512 |
| 训练迭代数 | 2000 |
| PPO 步数/环境 | 24 |
| 学习率 | 1e-3 |
| 生成器学习率 | 3e-4 |
| 新颖性阈值 (AT-PC) | 0.7 |
| 轨迹缓冲区大小 | 200 |
| 热身迭代数 | 50 |
| 概率下界初始值 | 0.15 |
| 随机种子 | 42, 123, 456 |

**关键控制:** 三种方法在同一个地形生成器上使用完全相同的参数空间和范围，唯一区别是参数的采样策略（DR=随机, PAIRED=最大化遗憾, AT-PC=最大化遗憾+新颖性过滤）。

---

## 测试地形集

### 通用测试集（所有生成器共用，10 种）

沿用论文第五章的 10 种测试地形，确保与已有结果可比。

### 生成器专属测试集（每个生成器额外 4 种）

针对每个生成器的参数空间特点，设计 held-out 测试地形：

**G1 专属:**
- steep_stairs_down: 下楼梯 difficulty=0.95
- dense_obstacles: 离散障碍 difficulty=0.9, 30 个矩形
- narrow_stones: 梅花桩 difficulty=0.95 (极小石头)
- combined_slope_push: 斜坡 + 强扰动

**G2 专属:**
- extreme_step_height: step_height=0.25, step_width=0.2
- wide_gap: gap_width=1.4
- high_wave: amplitude=0.18, num_waves=3
- inverted_stairs_slippery: 反向阶梯 + friction=0.3

**G3 专属:**
- random_round_stones: stone_shape=1.0, stone_pattern=1.0, stone_distance=0.35
- staggered_tall_stones: stone_pattern=0.5, stone_height_var=0.12
- stepping_pillars_hard: 梅花桩+柱阵混合, difficulty=0.9
- tiny_stones_far: stone_size=0.25, stone_distance=0.3

**G4 专属:**
- tilted_cylinders: max_yx_angle=30°, num_objects=40
- dense_cones: object_type=cone, num_objects=50, height=0.3
- narrow_rails: rail_thickness=0.08, rail_height=0.25
- floating_ring_gap: ring_width=0.8, ring_height=0.4, gap_width=1.0

---

## 评估协议

每个测试地形跑 20 个 episode，记录：
- 成功率 (主指标)
- 平均奖励
- 平均 Episode 长度
- 标准差

### 统计检验

- 配对 t-检验: AT-PC vs PAIRED, AT-PC vs DR (每个生成器内)
- 双因素方差分析 (Two-way ANOVA): 方法 × 生成器 交互效应
- Cohen's d 效应量

---

## 核心假设与预期结果

### H1: AT-PC 在所有生成器上都优于 DR 和 PAIRED

**预期:** AT-G1 > PA-G1 > DR-G1, AT-G2 > PA-G2 > DR-G2, ... 对所有 G 成立。

**验证方式:** 通用测试集上的平均成功率。

**论文意义:** 证明 AT-PC 是通用的参数优化方法，不依赖特定地形生成器。

### H2: 参数空间越大，AT-PC 的优势越明显

**预期:** AT-PC 相对 PAIRED 的提升幅度: G3(21维) > G4(18维) > G2(15维) > G1(7维)

**验证方式:** 计算每个生成器上 AT-PC vs PAIRED 的成功率差值。

**论文意义:** 证明 AT-PC 的新颖性过滤在高维参数空间中更有价值——参数空间越大，PAIRED 越容易陷入遗憾值停滞（因为高维空间中"看起来不同但效果相同"的参数组合更多），而 AT-PC 的轨迹新颖性检测能有效避免这个问题。

### H3: AT-PC 的参数空间覆盖率显著高于 PAIRED

**预期:** AT-PC 探索的参数空间体积 > PAIRED > DR

**验证方式:** 记录训练过程中所有被接受的参数向量，计算凸包体积或 PCA 方差。

**论文意义:** 直接证明新颖性过滤迫使生成器探索更广的参数空间。

### H4: AT-PC 在组合挑战场景中优势最大

**预期:** 在生成器专属测试集的组合挑战中，AT-PC 提升幅度 > 单一挑战

**验证方式:** 对比组合挑战 vs 单一挑战的提升幅度。

**论文意义:** 与论文现有结论一致，进一步在更多地形生成器上验证。

---

## 结果呈现方式

### 表格 1: 通用测试集成功率矩阵

```
              G1      G2      G3      G4      平均
DR            60.0%   ?       ?       ?       ?
PAIRED        80.5%   ?       ?       ?       ?
AT-PC         87.5%   ?       ?       ?       ?
```

### 表格 2: 生成器专属测试集成功率

每个生成器一张子表，展示 14 种测试地形（10 通用 + 4 专属）的详细结果。

### 图 1: AT-PC 相对 PAIRED 的提升幅度 vs 参数空间维度

X 轴: 参数空间维度 (7, 15, 18, 21)
Y 轴: 成功率提升百分点
预期: 正相关趋势

### 图 2: 参数空间覆盖率对比

每个生成器一个子图，展示 DR/PAIRED/AT-PC 的参数分布（PCA 降维到 2D）。

### 图 3: Accept Rate 演化曲线

4 个生成器 × AT-PC 的 Accept Rate 随训练迭代的变化。
预期: 高维生成器的 Accept Rate 下降更快（因为更容易生成"看似不同实则相同"的参数）。

### 图 4: 收敛速度对比

12 组实验的训练曲线（Mean Reward vs Iteration）。

---

## 实验时间估算

| 项目 | 时间 |
|------|------|
| 单次训练 (2000 iter) | ~40 min |
| 36 次训练 | ~24 h |
| 评估 (每组 14 地形 × 20 ep) | ~2 h/组, 共 ~24 h |
| G2, G4 地形生成器实现 | ~4 h |
| 数据分析与绘图 | ~4 h |
| **总计** | **~56 h (约 2.5 天)** |

注: 可以 4 组并行跑（不同 seed），实际墙钟时间约 1.5 天。

---

## 实验优先级

如果时间有限，按以下优先级执行：

1. **P0 (必做):** G1 + G3 的完整 3×3 对比 → 证明 AT-PC 在原生和扩展参数空间上都有效
2. **P1 (重要):** G2 的完整对比 → 补充中等参数空间的数据点
3. **P2 (锦上添花):** G4 的完整对比 → 展示在 Trimesh 风格地形上的泛化性
4. **P3 (可选):** 参数空间覆盖率分析 → 提供更深入的机制解释

---

## 与论文现有实验的关系

本实验方案是论文第六章实验的**扩展实验**，定位为：

- 第六章: 在 G1 (原生地形) 上验证 AT-PC vs DR vs PAIRED → **已完成**
- 本方案: 在 G1/G2/G3/G4 四种生成器上验证 → **证明 AT-PC 的通用性**

论文中可以作为 **第六章的 6.X 节** 或 **附录** 呈现，标题建议：

> "6.X 跨地形生成器泛化性验证"
> 或
> "附录 B: AT-PC 在不同参数化地形生成器上的表现"
