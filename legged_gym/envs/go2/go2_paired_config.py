"""
PAIRED Configuration for Go2
用于 PAIRED 算法的 Go2 配置
"""
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GO2PAIREDCfg(LeggedRobotCfg):
    """
    PAIRED 配置
    - 地形参数由生成器动态控制
    - Domain Randomization 范围可被生成器覆盖
    """
    
    class env(LeggedRobotCfg.env):
        num_envs = 512
        num_observations = 48
        num_actions = 12
        episode_length_s = 20
    
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = False  # PAIRED 自己管理课程
        selected = False
        
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        
        # 初始地形比例 (会被生成器覆盖)
        terrain_proportions = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1]
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        
    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 20.}
        damping = {'joint': 0.5}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1

    class domain_rand(LeggedRobotCfg.domain_rand):
        # 初始范围 (会被生成器动态调整)
        randomize_friction = True
        friction_range = [0.5, 1.25]
        
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.5]
        
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        only_positive_rewards = True
        tracking_sigma = 0.25
        
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.5
            torques = -0.0002
            dof_vel = -0.0001
            dof_acc = -2.5e-7
            feet_air_time = 1.0
            collision = -1.0
            action_rate = -0.01
            dof_pos_limits = -10.0
            termination = -0.0

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
        clip_observations = 100.
        clip_actions = 100.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05


class GO2PAIREDCfgPPO(LeggedRobotCfgPPO):
    seed = 42
    
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 2000
        
        save_interval = 100
        experiment_name = 'go2_paired'
        run_name = 'paired_v1'
        
        resume = False
        load_run = -1
        checkpoint = -1


# PAIRED 特定配置
class PAIREDConfig:
    """PAIRED 算法配置"""
    
    # 生成器配置
    generator_lr = 1e-4
    generator_hidden_dims = [256, 128]
    generator_update_freq = 10  # 每 N 次 solver 更新后更新生成器
    
    # Antagonist 配置
    antagonist_ema_decay = 0.995  # EMA 衰减率
    
    # 缓冲区配置
    buffer_capacity = 1000
    use_multi_level_buffer = False  # PAIRED 不使用多级缓冲
    
    # 遗憾值配置
    regret_clip = 10.0  # 裁剪遗憾值防止梯度爆炸
    
    # 熵正则化
    generator_entropy_coef = 0.01
