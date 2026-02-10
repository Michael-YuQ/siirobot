from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_flat_config import GO2FlatCfg, GO2FlatCfgPPO
from legged_gym.envs.go2.go2_dr_baseline_config import GO2DRBaselineCfg, GO2DRBaselineCfgPPO
from legged_gym.envs.go2.go2_paired_config import GO2PAIREDCfg, GO2PAIREDCfgPPO
from legged_gym.envs.go2.go2_remidi_config import GO2ReMiDiCfg, GO2ReMiDiCfgPPO
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register("go2_flat", LeggedRobot, GO2FlatCfg(), GO2FlatCfgPPO())
task_registry.register("go2_dr_baseline", LeggedRobot, GO2DRBaselineCfg(), GO2DRBaselineCfgPPO())
task_registry.register("go2_paired", LeggedRobot, GO2PAIREDCfg(), GO2PAIREDCfgPPO())
task_registry.register("go2_remidi", LeggedRobot, GO2ReMiDiCfg(), GO2ReMiDiCfgPPO())
