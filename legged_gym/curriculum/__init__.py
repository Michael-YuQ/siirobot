"""
Curriculum Learning Module for Legged Robot Training
包含 PAIRED, ReMiDi 和 AT-PC 对抗课程学习算法实现
"""

from .generator import TerrainGenerator
from .parametric_generator import ParametricTerrainGenerator, ParametricConditionBuilder
from .paired_trainer import PAIREDTrainer
from .regret_buffer import RegretBuffer, MultiLevelBuffer
from .remidi import ReMiDiTrainer, TrajectoryNoveltyChecker
from .adversarial_trainer import AdversarialCurriculumTrainer

__all__ = [
    'TerrainGenerator',
    'ParametricTerrainGenerator',
    'ParametricConditionBuilder',
    'PAIREDTrainer',
    'RegretBuffer',
    'MultiLevelBuffer',
    'ReMiDiTrainer',
    'TrajectoryNoveltyChecker',
    'AdversarialCurriculumTrainer',
]
