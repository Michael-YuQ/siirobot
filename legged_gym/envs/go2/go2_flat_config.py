"""
Standard PPO (Flat) Baseline Configuration for Go2
仅在平坦地面训练，作为性能下界基线
"""
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GO2FlatCfg(LeggedRobotCfg):
    """
    Standard PPO 基线配置
    - 仅平坦地面
    - 无域随机化
    - 无外力扰动
    - 作为性能下界
    """
    
    class env(LeggedRobotCfg.env):
        num_envs = 1024
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
        mesh_type = 'plane'  # 仅平坦地面！
        curriculum = False
        selected = False
        
        # 物理属性 - 固定值
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        
        measure_heights = False  # 平地不需要高度测量
        
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
        # === 关闭所有域随机化 ===
        randomize_friction = False
        friction_range = [1.0, 1.0]  # 固定摩擦
        
        randomize_base_mass = False
        added_mass_range = [0.0, 0.0]  # 无附加质量
        
        push_robots = False  # 无外力扰动
        push_interval_s = 15
        max_push_vel_xy = 0.0

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        only_positive_rewards = True
        tracking_sigma = 0.25
        
        class scales(LeggedRobotCfg.rewards.scales):
            # 主要奖励
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            
            # 稳定性惩罚
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.5
            
            # 能耗惩罚
            torques = -0.0002
            dof_vel = -0.0001
            dof_acc = -2.5e-7
            
            # 其他
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
        add_noise = True  # 保留观测噪声（现实中传感器有噪声）
        noise_level = 1.0
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05


class GO2FlatCfgPPO(LeggedRobotCfgPPO):
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
        experiment_name = 'go2_flat'
        run_name = 'flat_baseline'
        
        resume = False
        load_run = -1
        checkpoint = -1
